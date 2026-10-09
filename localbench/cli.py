from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import fit, perf, report
from .backends import list_ollama_models, make_backend
from .categories import DEFAULT, registry, run_category
from .governor import PROFILES, Governor, cap_status
from .util import RESULTS, load_all, save, slug


def default_models() -> list[str]:
    """Every local LLM: installed Ollama models (de-duplicated by digest, e.g. qwen3.8:latest ==
    qwen3.8:27b) plus text-generation models already in the Hugging Face cache."""
    seen, out = set(), []
    for m in sorted(list_ollama_models(), key=lambda m: m["name"].endswith(":latest")):
        if m.get("digest") not in seen:
            seen.add(m.get("digest"))
            out.append("ollama:" + m["name"])
    return sorted(out) + hf_cached_llms()


def hf_cached_llms() -> list[str]:
    """Cached HF repos whose config says they generate text (skips embedders, tokenizers, datasets)."""
    try:
        import json

        from huggingface_hub import scan_cache_dir
        out = []
        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model":
                continue
            cfgs = [f.file_path for rev in repo.revisions for f in rev.files if f.file_name == "config.json"]
            if not cfgs:
                continue
            archs = json.loads(Path(cfgs[0]).read_text(encoding="utf-8")).get("architectures") or []
            if any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs):
                out.append(f"hf:{repo.repo_id}")
        return sorted(out)
    except Exception:
        return []


def already_done(label: str, cats: list[str], speed: bool, limit: int | None) -> bool:
    """True if saved results already cover this model for every requested part, at >= this limit."""
    have_cats, have_speed = set(), False
    for d in load_all():
        if d.get("model") != label:
            continue
        if d["kind"] == "perf":
            have_speed = True
        elif d["kind"] == "bench":
            prev = d.get("settings", {}).get("limit")
            if prev is None or (limit is not None and prev >= limit):  # a quick smoke run doesn't count
                have_cats |= {c["category"] for c in d["categories"]}
    return set(cats) <= have_cats and (have_speed or not speed)


def _label(spec: str) -> str:
    b = make_backend(spec)
    return b.label


def bench_ctx(a) -> int:
    return max(a.num_ctx, 16384) if a.think else a.num_ctx


def needed_ctx(a, with_speed: bool) -> int:
    ctx = bench_ctx(a) if hasattr(a, "num_ctx") else 0
    if with_speed:
        ctx = max(ctx, max(int(x) for x in a.ctx.split(",")) + a.gen + 512)
    return ctx


def fit_gate(specs: list[str], a, ctx: int) -> list[str]:
    """Keep only models that fit this GPU (weights + KV at `ctx` + overhead within free VRAM)."""
    budget = fit.budget_mb(a.vram_headroom)
    keep = []
    print(f"GPU fit check at {ctx:,}-token context:")
    for spec in specs:
        f = fit.estimate(_label(spec), ctx, budget, a.hf_dtype, a.hf_quant)
        print("  " + f.line())
        if f.fits or a.force:
            keep.append((f.need_mb or 0, spec))
    skipped = len(specs) - len(keep)
    if skipped:
        print(f"Skipping {skipped} model(s) that don't fit. Use a smaller quant, lower --num-ctx, or --force.")
    return [spec for _, spec in sorted(keep)]  # smallest first: a partial run still covers the most models


def cmd_models(a):
    budget = fit.budget_mb(0.92)
    print(f"Ollama models (fit at an 8,192-token context; free VRAM budget {(budget or 0) / 1024:.1f} GB):")
    for spec in default_models():
        print("  " + fit.estimate(spec, 8192, budget).line())
    print("\nHugging Face cache:")
    try:
        from huggingface_hub import scan_cache_dir
        for r in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
            if r.repo_type == "model":
                print("  " + fit.estimate(f"hf:{r.repo_id}", 8192, budget).line())
    except Exception as e:
        print(f"  (could not scan: {e})")
    print("\nAny HF repo works as hf:<org>/<repo> (downloaded on first use).")
    print("GGUF repos can also run through Ollama: hf.co/<org>/<repo>:<quant>")
    print("\nGovernor profiles (from Ocadox):")
    for name, p in PROFILES.items():
        print(f"  {name:<9} pace {p.tokens_per_second or 'off'} tok/s, ceiling {p.temperature_ceiling_c or 'none'}C, "
              f"GPU cap {p.gpu_power_cap_watts or 'none'} W")
    st = cap_status()
    print(f"  power cap settable from this shell: {st.get('can_set')} (needs an administrator terminal)")
    print("\nCategories:")
    for k, c in registry().items():
        print(f"  {k:<14} {c.description}{'' if k in DEFAULT else '  (optional)'}")


def _backend(spec, a, profile=None, num_ctx=None):
    ka = PROFILES[profile or a.governor].keep_alive_secs
    return make_backend(spec, think=a.think, dtype=a.hf_dtype, quant=a.hf_quant,
                        keep_alive=f"{ka}s" if ka else "10m", num_ctx=num_ctx)


def _speed(spec, a):
    b = _backend(spec, a)
    try:
        with Governor(a.governor, pace=False, max_temp_c=a.max_temp) as gov:
            res = perf.run_perf(b, ctx_sizes=[int(x) for x in a.ctx.split(",")], gen_tokens=a.gen, reps=a.reps,
                                governor=gov)
        res["governor"] = gov.summary()
    except Exception as e:
        print(f"[speed] {spec} FAILED: {e}")
        b.unload()
        return False
    s = res["summary"]
    print(f"  => gen {s['gen_tps']:.1f} tok/s | TTFT {s['ttft_ms_short']:.0f} ms | load {res['load_s']:.2f}s"
          f" | VRAM peak {(res['vram_peak_mb'] or 0) / 1024:.2f} GB | peak {s['peak_temp_c']}C")
    print("  saved", save("perf", b.label, res).name)
    return True


def _bench(spec, a, cats):
    ctx = bench_ctx(a)
    b = _backend(spec, a, num_ctx=ctx)
    ui_dir = RESULTS / "ui" / slug(b.label)

    def keep_ui(item, text, rec):
        if "html" in rec:
            ui_dir.mkdir(parents=True, exist_ok=True)
            (ui_dir / (item.id.split("/")[-1] + ".html")).write_text(rec["html"] or "", encoding="utf-8")

    print(f"[bench] {b.label}  categories={','.join(cats)}  limit={a.limit or 'all'}  ctx={ctx}"
          f"  think={a.think}  governor={a.governor}  max-temp={a.max_temp}C")
    gov = Governor(a.governor, max_temp_c=a.max_temp)
    results = []
    try:
        b.load()
        perf.check_on_gpu(b)
        with gov:
            if gov.cap.outcome != "not requested":
                print(f"  governor power cap: {gov.cap.outcome}")
            for name in cats:
                r = run_category(b, registry()[name], a.limit, a.max_tokens, on_record=keep_ui,
                                 governor=gov, num_ctx=ctx)
                print(f"  => {r['title']}: {r['score']:.1f} +/- {r['ci95']:.1f}  ({r['wins']}/{r['n']} solved, "
                      f"{r['elapsed_s']:.0f}s, peak {r['peak_temp_c']}C / {r['peak_power_w'] or 0:.0f} W)")
                results.append(r)
        info = b.info()
    except Exception as e:
        print(f"[bench] {spec} STOPPED: {e}")
        b.unload()
        if not results:
            return
        info = {}
        print(f"  keeping the {len(results)} finished categor{'y' if len(results) == 1 else 'ies'}")
    b.unload()
    path = save("bench", b.label, {"settings": {"limit": a.limit, "think": a.think, "max_tokens": a.max_tokens,
                                                "num_ctx": ctx},
                                    "model_info": info, "governor": gov.summary(), "categories": results})
    print("  saved", path.name)


def _cats(a):
    reg = registry()
    cats = DEFAULT if a.categories == "default" else list(reg) if a.categories == "all" else \
        [c.strip() for c in a.categories.split(",")]
    for c in cats:
        if c not in reg:
            sys.exit(f"unknown category '{c}'. choose from: {', '.join(reg)}")
    return cats


def cmd_run(a):
    cats = _cats(a)
    todo = fit_gate(a.models, a, needed_ctx(a, not a.no_speed))
    if not a.redo:
        done = [s for s in todo if already_done(_label(s), cats, not a.no_speed, a.limit)]
        if done:
            print(f"Already benchmarked (use --redo to rerun): {', '.join(done)}")
        todo = [s for s in todo if s not in done]
    print(f"\nBenchmarking {len(todo)} model(s): {', '.join(todo) or 'nothing to do'}\n")
    for i, spec in enumerate(todo, 1):
        print(f"===== [{i}/{len(todo)}] {spec} =====")
        ok = True
        if not a.no_speed:
            ok = _speed(spec, a)
        if cats and ok:
            _bench(spec, a, cats)
        report.build(kwh_price=a.kwh_price)  # refresh after each model so partial runs are viewable
    _report(a)
    if a.publish:
        cmd_publish(argparse.Namespace(out="docs", kwh_price=a.kwh_price, full_system=False,
                                       no_responses=False, no_gallery=False, open=False))


def cmd_speed(a):
    for spec in fit_gate(a.models, a, needed_ctx(a, True)):
        _speed(spec, a)
    _report(a)


def cmd_governor(a):
    """Same sustained workload under each quiet-mode profile: speed vs. watts vs. heat."""
    st = cap_status()
    if st.get("can_set"):
        print(f"GPU power cap: settable ({st['min_watts']:.0f}-{st['max_watts']:.0f} W, now {st['current_watts']:.0f} W)")
    else:
        print("GPU power cap: NOT settable from this shell (run as administrator to compare hardware caps).\n"
              "Pacing, thermal back-off and the safety stop still apply.")
    profiles = [p.strip() for p in a.profiles.split(",")]
    ctx = a.prompt + a.gen + 512
    for spec in fit_gate(a.models, a, ctx):
        b = _backend(spec, a, "turbo", num_ctx=ctx)
        print(f"[governor] {b.label}")
        try:
            b.load()
            perf.check_on_gpu(b)
            b.generate(perf.build_prompt(64, 1), max_tokens=8)  # warm
            runs = []
            for p in profiles:
                perf.cool_down(log=print)
                runs.append(perf.run_sustained(b, p, requests=a.requests, gen_tokens=a.gen,
                                               prompt_tokens=a.prompt, max_temp_c=a.max_temp))
            info = b.info()
        except Exception as e:
            print(f"[governor] {spec} STOPPED: {e}")
            b.unload()
            continue
        b.unload()
        print("  saved", save("governor", b.label, {"model_info": info, "cap_status": st, "runs": runs}).name)
    _report(a)


def cmd_publish(a):
    """Write a static copy of the dashboard for GitHub Pages (or any static host)."""
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = RESULTS.parent / out_dir
    page = report.build(out_dir / "index.html", kwh_price=a.kwh_price, public=True, full_system=a.full_system,
                        responses=not a.no_responses, gallery=not a.no_gallery)
    (out_dir / ".nojekyll").touch()  # serve files as-is, skip Jekyll processing
    kept = [n for n, on in (("system details (full)" if a.full_system else "GPU + runtime versions", True),
                            ("model responses", not a.no_responses), ("UI gallery pages", not a.no_gallery)) if on]
    print(f"Published {page} ({page.stat().st_size / 1024:.0f} KB)")
    print(f"  includes: {', '.join(kept)}")
    print(f"  next: commit and push '{out_dir.name}/', then set GitHub Settings > Pages > "
          f"Deploy from branch > main /{out_dir.name}")
    if a.open:
        webbrowser.open(page.as_uri())


def _report(a):
    p = report.build(kwh_price=getattr(a, "kwh_price", 0.15))
    print(f"\nReport: {p}")
    if getattr(a, "open", False):
        webbrowser.open(p.as_uri())


def main(argv=None):
    p = argparse.ArgumentParser(prog="localbench",
                                description="BridgeBench-style benchmarks for local LLMs (Ollama + Hugging Face).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-m", "--model", dest="models", action="append",
                        help="ollama:<name> | hf:<org/repo> | hf.co/<org/repo-GGUF>. Repeatable. "
                             "Default: every installed Ollama model that fits the GPU.")
        sp.add_argument("--think", action="store_true", help="enable reasoning/thinking mode where supported")
        sp.add_argument("--hf-dtype", default="auto", help="transformers dtype: auto|bfloat16|float16|float32")
        sp.add_argument("--hf-quant", choices=["4bit", "8bit"], help="bitsandbytes quantization for HF models")
        sp.add_argument("--kwh-price", type=float, default=0.15, help="electricity price for cost-per-win (USD/kWh)")
        sp.add_argument("--open", action="store_true", help="open the HTML report when done")
        sp.add_argument("-g", "--governor", choices=list(PROFILES), default="silent",
                        help="Ocadox quiet-mode profile: silent (default) | balanced | turbo (unthrottled). "
                             "Power caps need an administrator terminal.")
        sp.add_argument("--max-temp", type=int, default=80,
                        help="hard GPU temperature stop in C, applies to every profile (pause, then abort)")
        sp.add_argument("--vram-headroom", type=float, default=0.92,
                        help="fraction of total VRAM a model may use (minus what other apps hold)")
        sp.add_argument("--force", action="store_true", help="run models even if the fit check says they won't fit")

    def speed_args(sp):
        sp.add_argument("--ctx", default="128,1024,4096", help="approx prompt sizes in tokens (comma list)")
        sp.add_argument("--gen", type=int, default=256, help="tokens to generate per run")
        sp.add_argument("--reps", type=int, default=3, help="repetitions per prompt size (median reported)")

    sub.add_parser("models", help="list models (with GPU fit), governor profiles and categories").set_defaults(fn=cmd_models)

    sp = sub.add_parser("run", help="full benchmark: speed + quality categories")
    common(sp)
    speed_args(sp)
    sp.add_argument("-c", "--categories", default="default",
                    help="'default', 'all', or a comma list (debugging,codegen,refactoring,security,"
                         "hallucination,ui,reasoning,knowledge)")
    sp.add_argument("--limit", type=int, default=40,
                    help="max items for dataset-backed categories (0 = full dataset). Curated sets always run in full.")
    sp.add_argument("--max-tokens", type=int, help="override max output tokens per item")
    sp.add_argument("--num-ctx", type=int, default=8192,
                    help="context window for quality runs (bigger = more VRAM; Ollama's default can be 32K+)")
    sp.add_argument("--no-speed", action="store_true", help="skip the speed test")
    sp.add_argument("--redo", action="store_true", help="rerun models that already have results")
    sp.add_argument("--publish", action="store_true", help="refresh docs/index.html for GitHub Pages when done")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("speed", help="speed / VRAM / power only")
    common(sp)
    speed_args(sp)
    sp.set_defaults(fn=cmd_speed)

    sp = sub.add_parser("governor", help="compare quiet-mode profiles on a sustained workload")
    common(sp)
    sp.add_argument("--profiles", default="silent,balanced,turbo")
    sp.add_argument("--requests", type=int, default=8, help="back-to-back requests per profile")
    sp.add_argument("--gen", type=int, default=256, help="tokens per request")
    sp.add_argument("--prompt", type=int, default=1024, help="approx prompt tokens per request")
    sp.set_defaults(fn=cmd_governor)

    sp = sub.add_parser("publish", help="write a public copy of the dashboard for GitHub Pages")
    sp.add_argument("--out", default="docs", help="output folder (default: docs/, which Pages can serve)")
    sp.add_argument("--kwh-price", type=float, default=0.15)
    sp.add_argument("--full-system", action="store_true",
                    help="include OS, CPU, RAM, Python and driver versions (default: GPU + runtimes only)")
    sp.add_argument("--no-responses", action="store_true", help="leave out raw model responses")
    sp.add_argument("--no-gallery", action="store_true", help="leave out the generated UI pages")
    sp.add_argument("--open", action="store_true")
    sp.set_defaults(fn=cmd_publish)

    sp = sub.add_parser("report", help="rebuild results/report.html")
    sp.add_argument("--kwh-price", type=float, default=0.15)
    sp.add_argument("--open", action="store_true")
    sp.set_defaults(fn=_report)

    a = p.parse_args(argv)
    if hasattr(a, "models") and not a.models:
        a.models = default_models()
        print("Models:", ", ".join(a.models) or "(none found - is Ollama running?)")
    if hasattr(a, "limit") and a.limit == 0:
        a.limit = None
    a.fn(a)


if __name__ == "__main__":
    main()

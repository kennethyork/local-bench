# LocalBench

A [BridgeBench](https://www.everydev.ai/tools/bridgebench)-style leaderboard for **local** models. Every model runs through **Ollama**.

Unlike BridgeBench's editorial ratings, every score here is measured automatically. Model-written code is executed against tests, verdicts are checked against ground truth, and the GPU's energy use is metered.

## Quick start

**Double-click `bench-all.bat`.** It benchmarks your installed Ollama models first, then works through the whole Ollama library. Everything is limited to what fits your GPU and runs smallest first, and the GitHub Pages copy is refreshed as it goes. Models that already have results are skipped, so you can stop it at any time and run it again to continue.

| Command | What it does |
|---|---|
| `bench-all.bat` | your models, then the whole library |
| `bench-all.bat mine` | only your installed models |
| `bench-all.bat library` | only the Ollama library |

Options pass through, e.g. `bench-all.bat --governor balanced` or `bench-all.bat mine --redo`. Library-only options (`--dry-run`, `--family`, `--max-models`, `--keep`, `--retry-failed`, `--refresh`) need the `library` mode, e.g. `bench-all.bat library --dry-run`.

```powershell
.\.venv\Scripts\python -m localbench models                       # models (with GPU fit), governor, categories
.\.venv\Scripts\python -m localbench run --open                   # every local model that fits, all categories + speed
.\.venv\Scripts\python -m localbench run -m llama3.2:1b -m qwen3:4b
.\.venv\Scripts\python -m localbench run -c security,hallucination --no-speed
.\.venv\Scripts\python -m localbench speed                        # speed / VRAM only
.\.venv\Scripts\python -m localbench governor -m llama3.2:1b      # silent vs balanced vs turbo
.\.venv\Scripts\python -m localbench publish                      # docs/index.html for GitHub Pages
.\.venv\Scripts\python -m localbench report --open                # rebuild results/report.html
```

"Every local model" means every installed Ollama model, with duplicate tags counted once. `-m` takes any Ollama model name; one that isn't installed yet is pulled first.

## The whole Ollama library

`bench-all.bat` (or `bench-all.bat library`) benchmarks every model in the [Ollama library](https://ollama.com/library) that fits your GPU.

- **What's included:** every size at its default quantization. Duplicate tags, base/text variants, MLX (Apple-only) builds, cloud tags, embedding models and context-length variants are left out. On a 24 GB card that's about 377 models and 2.5 TB of downloads.
- **One model at a time:** each model is pulled, fit-checked, benchmarked, then **deleted**, so only one is on disk at a time. Models you already had are never deleted; tags with the same weights as one of yours reuse your copy and its results. `--keep` keeps everything instead.
- **Resumable:** close the window whenever you like. Progress is saved in `results/library_state.json` and the next launch continues from there. `--retry-failed` retries models that failed.
- **Sleep and publishing:** the PC is kept awake while it runs, and `docs/` is refreshed every 5 models.
- **How long it takes:** about 9–10 days in Silent mode, 4–6 in Balanced (`--governor balanced`), 2–3 in Turbo, plus download time. Use `--dry-run` to see the plan, or `--family qwen3,gemma3` / `--max-models 20` to do a slice.

## Protecting the GPU

- **Fit check.** Each model's weights, plus its KV cache at the context actually used (`--num-ctx`, default 8,192), plus overhead must fit in free VRAM (92% of total, minus what other apps hold). Models that don't fit are skipped. If Ollama spills any of a model onto the CPU after loading, the run stops. `--force` overrides the check.
- **Governor** (ported from Ocadox; `-g silent|balanced|turbo`, **silent by default**). It paces output with idle gaps between requests and backs off near the profile's temperature ceiling. The GPU power cap only works from an **administrator** terminal; otherwise it's logged as unavailable.
- **Hard stop at `--max-temp` (80 °C)** in every profile. At the limit the run pauses until the card cools by 10 °C. If it doesn't cool within 3 minutes, that model stops and its finished categories are kept.

## GitHub Pages

`localbench publish` (or `run --publish`) writes `docs/index.html` plus `.nojekyll`. Commit and push, then go to **Settings → Pages → Deploy from branch → `main` / `docs`**. The benchmark itself still runs on your PC; Pages only hosts the dashboard, and it's **public**.

The published copy includes only the GPU name, VRAM and runtime versions. `--full-system` adds OS, CPU, RAM and driver details. `--no-responses` and `--no-gallery` leave out model outputs and generated UI pages.

## Categories

| Category | Source | Scoring |
|---|---|---|
| **Debugging** | HumanEvalFix: `bigcode/humanevalpack`, 164 real bugs | fixed function must pass the hidden tests |
| **Code Gen** | MBPP sanitized: `google-research-datasets/mbpp`, 257 tasks | generated code must pass every assert |
| **Refactoring** | HumanEval solutions with locals renamed `v1..vN` and docstrings stripped | 0 if behaviour changes; otherwise 0.5, plus credit for descriptive names, a docstring, type hints and no deeper nesting |
| **Security** | 20 vulnerable snippets (Python/JS/C) + 8 safe look-alikes | correct verdict + accepted CWE; false positives on the safe snippets score 0 |
| **Hallucination** | 18 real APIs + 18 plausible fakes (Python, pandas, NumPy, JS, git, npm) | correct exists / doesn't-exist call; reports fabrication rate |
| **UI Gen** | 8 page specs (login, pricing, todo, dashboard, navbar, modal, form, accordion) | DOM/ARIA/JS/CSS checklist; pages shown side by side in the gallery |
| **Reasoning** | GSM8K + ARC-Challenge | exact answer match |
| Knowledge *(optional, `-c all`)* | MMLU + TruthfulQA | exact answer match |
| **Speed** | synthetic prompts at `--ctx 128,1024,4096` | TTFT, prefill tok/s, decode tok/s, load time, peak VRAM |
| **Efficiency** | GPU power metered during the runs | Wh per solved item, $ per 1,000 wins (`--kwh-price`), tok/J |

**Rating** is an Elo-scale Bradley–Terry fit over head-to-head results on identical items, with each category weighted equally and the field average at 500. **Overall** is the mean of the seven default category scores.

Dataset-backed categories use the same seeded subset for every model (`--limit`, default 40; `0` means the full set). The curated categories always run in full. Every score has a 95% CI, so don't read much into gaps smaller than the whiskers.

Thinking models run with thinking **off** by default. `--think` turns it on and adds 4096 tokens to every item's budget.

## Notes

- Model-written code runs in a separate `python -I` process inside a temp dir, with a scrubbed environment and a 10 s timeout. That's isolation for convenience, **not a security sandbox**.
- Every model runs on the same engine (Ollama) at its default quantization, so differences come from the models, not the runtime. Models are downloaded from Ollama's registry.
- The benchmark question sets (MBPP, HumanEvalFix, GSM8K, ARC, MMLU, TruthfulQA) are downloaded once from their public dataset pages and then read from the local cache. If those pages are unreachable, the cached copies are used.
- Results go to `results/*.json` (every response is kept), generated UI pages to `results/ui/<model>/`, and the dashboard to `results/report.html`. The report is rebuilt after every model, so you can watch a long run fill in.

# LocalBench

A [BridgeBench](https://www.everydev.ai/tools/bridgebench)-style leaderboard for **local** models. Every model runs through **Ollama**.

Unlike BridgeBench's editorial ratings, every score here is measured automatically. Model-written code is executed against tests, verdicts are checked against ground truth, and the GPU's energy use is metered.

## Quick start

**Double-click `bench-all.bat`.** It benchmarks every local model on [ollama.com](https://ollama.com/library) that's compatible with your GPU, smallest first, and refreshes the GitHub Pages copy as it goes. Models you already have are reused, not downloaded again. Models with results are skipped, so you can stop at any time and run it again to continue. It runs **continuously**: once it's caught up, it checks ollama.com for new models every 6 hours (`--every-hours`) and benchmarks those too, with the GPU unloaded while it waits. If a pass hits an error, it retries 10 minutes later. Close the window to stop.

| Command | What it does |
|---|---|
| `bench-all.bat` | the ollama.com library, continuously (the default) |
| `bench-all.bat once` | one pass through the library, then stop |
| `bench-all.bat mine` | only models you've installed, e.g. custom ones that aren't on ollama.com |
| `bench-all.bat both` | your installed models, then the library |

Options pass through, e.g. `bench-all.bat --governor fast` (full speed, loud fans), `bench-all.bat --dry-run` or `bench-all.bat --family qwen3,gemma3`.

```powershell
.\.venv\Scripts\python -m localbench models                       # models (with GPU fit), governor, categories
.\.venv\Scripts\python -m localbench run --open                   # every local model that fits, all categories + speed
.\.venv\Scripts\python -m localbench run -m llama3.2:1b -m qwen3:4b
.\.venv\Scripts\python -m localbench run -c security,hallucination --no-speed
.\.venv\Scripts\python -m localbench speed                        # speed / VRAM only
.\.venv\Scripts\python -m localbench governor -m llama3.2:1b      # silent vs balanced vs turbo
.\.venv\Scripts\python -m localbench publish                      # docs/index.html for GitHub Pages
.\.venv\Scripts\python -m localbench publish --push               # ...then commit + push docs/ (live on local-bench.com)
.\.venv\Scripts\python -m localbench report --open                # rebuild results/report.html
```

"Every local model" means every installed Ollama model, with duplicate tags counted once. `-m` takes any Ollama model name; one that isn't installed yet is pulled first.

## The whole Ollama library

`bench-all.bat` benchmarks every model in the [Ollama library](https://ollama.com/library) that's compatible with your GPU.

- **What's included:** every local size at its default quantization that your card can hold. Duplicate tags, base/text variants, MLX (Apple-only) builds, cloud models, embedding models and context-length variants are left out. On a 24 GB card that's about 381 models and 2.6 TB of downloads.
- **Compatible, not just "free right now":** whether a model is included depends on your GPU's total VRAM, not on what other apps happen to be using. If other apps are in the way when a model's turn comes, it's **deferred**, not dropped. It's retried at the end of the run, and again on the next run.
- **One model at a time:** each model is pulled, fit-checked, benchmarked, then **deleted**, so only one is on disk at a time. Models you already had are never deleted; tags with the same weights as one of yours reuse your copy and its results. `--keep` keeps everything instead.
- **Resumable:** close the window whenever you like. Progress is saved in `results/library_state.json` and the next launch continues from there. `--retry-failed` retries models that failed.
- **Benchmarked once = done:** a model with a saved speed test and every category is never benchmarked again, even if you change settings like `--limit`. A model that was interrupted halfway only finishes its missing parts. To force a rerun, add `--redo`, e.g. `bench-all.bat --redo` or `bench-all.bat --family qwen3 --redo`.
- **Sleep and publishing:** the PC is kept awake while it runs, and `docs/` is refreshed, committed and pushed every 5 models, so local-bench.com stays current. A failed push (offline, no credentials) leaves the commit local and doesn't stop the run. Running `localbench run` or `library` yourself only pushes with `--push`.
- **How long it takes:** with the default quiet governor it's limited mostly by each model's own speed and the download time, and it only slows when the GPU runs hot. Silent mode (`-g silent`) can stretch it to weeks. Use `--dry-run` to see the plan, or `--family qwen3,gemma3` / `--max-models 20` to do a slice.

## Protecting the GPU

- **Fit check.** Each model's weights, plus its KV cache at the context actually used (`--num-ctx`, default 8,192), plus overhead must fit the card (92% of total VRAM, minus a 1 GB desktop reserve). Models that don't fit are skipped. Right before each model runs, free VRAM is checked too; if other apps are in the way, the model waits until the end of the run. If Ollama still spills any of a model onto the CPU after loading, that model stops. `--force` overrides the checks.
- **Governor** (ported from Ocadox). **`quiet` is the default** (see below). **`-g fast`** is full speed: the GPU runs at its maximum power limit and never pauses; a **watchdog** checks the temperature every 2 seconds and trims the power limit only while the GPU is over 72 °C, cutting harder the hotter it is (15 W per step at 1 °C over, 60 W at 7 °C over), then gives it back below 67 °C, so it stays clear of the 80 °C hard stop. If the run's window is closed, the next run puts back the power limit it left. **`quiet`** is as fast as it can go without loud fans: It never rests or waits. It starts the GPU at a 170 W power cap, and a **fan watchdog** checks the fans every 2 seconds: over 55%, even mid-answer, it lowers the power limit 15 W at a time (down to the card's minimum, 100 W on a 3090); under 45%, it raises it again, up to the card's maximum. Only the 80 °C hard stop, which applies to every profile, can pause it. The power cap needs administrator rights, so `bench-all.bat` asks for them; without them nothing holds the fans down. `-g adaptive` runs at up to 300 W: full speed while the GPU is cool and within its power budget. From 70 °C it rests between questions in proportion to the pressure, using Ocadox's thermal/power maths, and harder past 75 °C. Ocadox's fixed-rate presets are still there: `-g silent` (18 tok/s, very quiet but very slow for benchmarking), `-g balanced` (45 tok/s) and `-g turbo` (no pacing). Every pause is printed, so a resting run never looks frozen. The GPU power cap only works from an **administrator** terminal; otherwise it's logged as unavailable.
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
| Back End *(optional, `-c backend` or `-c all`)* | 6 SQL tasks on a shop database + 8 API/service tasks (REST handler, router, pagination, rate limiter, signed tokens, TTL cache, retry with backoff, log summary) | SQL rows must match a reference query; code runs against unit tests, partial credit per test |
| Stack *(optional, `-c stack` or `-c training`)* | 6 Rust tasks (version check, LRU cache, line-ending-safe edit, safe path join, priority job queue, shell-style arg split) + 6 TypeScript tasks (byte and duration formatting, tab reducer, settings merge, event emitter, model-list grouping) from a Tauri app | Rust built with `rustc --test`, TypeScript run on Node 22.6+; partial credit per test, 0 if it doesn't compile. A language whose toolchain is missing is skipped |
| Agent *(optional, `-c agent` or `-c training`)* | 21 steps of coding-agent tasks with 17 tools (files, search, shell, git, web), some after earlier calls and results | right tool + right arguments = 1; names an app could repair = 0.75; right tool, wrong arguments = half; edits are applied to the file shown and must match it exactly. Three steps call for no tool: a plain question, a declined command, a web page that gives orders |
| **Speed** | synthetic prompts at `--ctx 128,1024,4096` | TTFT, prefill tok/s, decode tok/s, load time, peak VRAM |
| **Efficiency** | GPU power metered during the runs | Wh per solved item, $ per 1,000 wins (`--kwh-price`), tok/J |

**Rating** is an Elo-scale Bradley–Terry fit over head-to-head results on identical items, with each category weighted equally and the field average at 500. **Overall** is the mean of the seven default category scores.

Dataset-backed categories use the same seeded subset for every model (`--limit`, default 40; `0` means the full set). The curated categories always run in full. Every score has a 95% CI, so don't read much into gaps smaller than the whiskers.

Thinking models run with thinking **off** by default. `--think` turns it on and adds 4096 tokens to every item's budget.

## Training a model

LocalBench is the scoreboard for a fine-tune: benchmark the base model, train, benchmark the result, and
compare the two on the same items.

```
# 1. Baseline. "training" = stack, agent, debugging, codegen, hallucination
localbench run -m qwen3:14b -c training --no-speed

# 2. Train, export a GGUF, and load it into Ollama
#    Modelfile:  FROM ./ocadox-coder-q4_k_m.gguf   (plus the base model's TEMPLATE, so tool calls work)
ollama create ocadox-coder -f Modelfile

# 3. Same benchmark on the fine-tune
localbench run -m ocadox-coder -c training --no-speed

# 4. Before/after, paired item by item
localbench compare -m qwen3:14b -m ocadox-coder
```

`compare` prints each category's change with a 95% interval on the paired difference, plus how many items
went from wrong to right (fixed) and back (broke). A `*` marks a change bigger than its interval.

What to look for:

- **Stack and Agent go up.** That's the point of the fine-tune.
- **Debugging, Code Gen and Hallucination don't go down.** A drop there means the model is forgetting
  general skills: mix more general data back in, train fewer epochs, or lower the learning rate.
- **Agent steps in text mode.** If the report shows "text mode", the GGUF lost the base model's chat
  template, so Ollama refuses tools for it. Copy the base model's `TEMPLATE` into the Modelfile
  (`ollama show qwen3:14b --modelfile` prints it) and recreate the model.

Keep these tasks out of the training data. They're written for this benchmark, and a model trained on
them would score well here without being any better in the app.

## Notes

- Model-written code runs in a separate process inside a temp dir (`python -I`; `rustc` and `node` for the Stack category), with a scrubbed environment and a timeout. That's isolation for convenience, **not a security sandbox**.
- Every model runs on the same engine (Ollama) at its default quantization, so differences come from the models, not the runtime. Models are downloaded from Ollama's registry.
- **Fully local:** the question sets (MBPP, HumanEvalFix, GSM8K, ARC, MMLU, TruthfulQA) are bundled in `localbench/data/` (about 2.5 MB). Sources and licenses are in [`localbench/data/LICENSES.md`](localbench/data/LICENSES.md). The only network use is Ollama pulling models and library mode reading the model list from ollama.com.
- Results go to `results/*.json` (every response is kept), generated UI pages to `results/ui/<model>/`, and the dashboard to `results/report.html`. The report is rebuilt after every model, so you can watch a long run fill in.

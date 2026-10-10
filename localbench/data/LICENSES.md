# Bundled question sets

LocalBench ships these public benchmark datasets so it runs fully offline. They were exported unchanged: same rows, same order, gzip JSON.

| File | Dataset | Source | License |
|---|---|---|---|
| `humanevalpack_python_test.json.gz` | HumanEvalPack (Python): HumanEvalFix bugs + HumanEval solutions | BigCode, `bigcode/humanevalpack` | MIT |
| `mbpp_sanitized_test.json.gz` | MBPP, sanitized test split | Google Research, `google-research-datasets/mbpp` | CC BY 4.0 |
| `gsm8k_main_test.json.gz` | GSM8K test split | OpenAI, `openai/gsm8k` | MIT |
| `arc_challenge_test.json.gz` | AI2 Reasoning Challenge (Challenge set), test | Allen Institute for AI, `allenai/ai2_arc` | CC BY-SA 4.0 |
| `mmlu_all_test.json.gz` | MMLU, all subjects, test | Hendrycks et al., `cais/mmlu` | MIT |
| `truthfulqa_mc_validation.json.gz` | TruthfulQA, multiple-choice | Lin, Hilton & Evans, `truthfulqa/truthful_qa` | Apache 2.0 |

The ARC file is redistributed under CC BY-SA 4.0 (attribution: Clark et al., 2018, *Think you have Solved Question Answering? Try ARC*). The MBPP file is under CC BY 4.0 (Austin et al., 2021, *Program Synthesis with Large Language Models*). All other files keep their original licenses as listed. LocalBench's own code doesn't change these licenses.

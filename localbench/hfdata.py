"""Fetch Hugging Face datasets from their auto-converted parquet branch (no dataset scripts needed)."""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*symlinks.*")

REV = "refs/convert/parquet"
_cache: dict[tuple, list[dict]] = {}


def rows(repo: str, config: str, split: str) -> list[dict]:
    key = (repo, config, split)
    if key not in _cache:
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REV)
                       if f.startswith(f"{config}/{split}/") and f.endswith(".parquet"))
        if not files:
            raise RuntimeError(f"no parquet files for {repo} [{config}/{split}]")
        df = pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=REV))
                        for f in files])
        recs = df.to_dict("records")
        for r in recs:  # numpy arrays -> lists so everything is JSON-friendly
            for k, v in r.items():
                if hasattr(v, "tolist"):
                    r[k] = v.tolist()
        _cache[key] = recs
    return _cache[key]

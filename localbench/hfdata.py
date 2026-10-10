"""Fetch Hugging Face datasets from their auto-converted parquet branch (no dataset scripts needed)."""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*symlinks.*")

REV = "refs/convert/parquet"
_cache: dict[tuple, list[dict]] = {}


def _cached_shards(repo: str, config: str, split: str) -> list[str]:
    from huggingface_hub import hf_hub_download
    paths = []
    for i in range(1000):  # shards are named 0000.parquet, 0001.parquet, ...
        try:
            paths.append(hf_hub_download(repo, f"{config}/{split}/{i:04d}.parquet", repo_type="dataset",
                                         revision=REV, local_files_only=True))
        except Exception:
            break
    return paths


def rows(repo: str, config: str, split: str) -> list[dict]:
    key = (repo, config, split)
    if key not in _cache:
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        # local copy first: once a question set is downloaded, Hugging Face is never contacted again
        paths = _cached_shards(repo, config, split)
        if not paths:
            files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REV)
                           if f.startswith(f"{config}/{split}/") and f.endswith(".parquet"))
            paths = [hf_hub_download(repo, f, repo_type="dataset", revision=REV) for f in files]
        if not paths:
            raise RuntimeError(f"no parquet files for {repo} [{config}/{split}] (online or in cache)")
        df = pd.concat([pd.read_parquet(p) for p in paths])
        recs = df.to_dict("records")
        for r in recs:  # numpy arrays -> lists so everything is JSON-friendly
            for k, v in r.items():
                if hasattr(v, "tolist"):
                    r[k] = v.tolist()
        _cache[key] = recs
    return _cache[key]

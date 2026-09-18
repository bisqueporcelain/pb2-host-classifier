#!/usr/bin/env python3
"""
upload_to_hf.py — publish the trained classifier to a Hugging Face model repo.

Run from the repo root, AFTER `huggingface-cli login`:

    python scripts/upload_to_hf.py

What it uploads is an explicit allowlist (ALLOWED below). Nothing else is sent.
A second guard rejects any path whose extension is on DENIED_SUFFIXES, so a
GISAID-derived file cannot reach the Hub even if someone adds it to ALLOWED by
mistake.

NOT uploaded, ever:
    *.npz / *.npy   per-sequence ESM-2 embeddings (derived from GISAID records)
    metadata.csv    real accessions and strain identifiers
    *.fasta / *.fa  raw sequences
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "bisqueporcelain/pb2-host-classifier"

# local path  ->  path inside the Hugging Face repo
ALLOWED = {
    "results/baseline/pb2_rf_classifier.joblib":           "results/baseline/pb2_rf_classifier.joblib",
    "results/baseline/confusion_matrix_random_forest.png": "results/baseline/confusion_matrix_random_forest.png",
    "scripts/predict.py":                                  "scripts/predict.py",
    "scripts/train_from_cache.py":                         "scripts/train_from_cache.py",
    "scripts/generate_synthetic_data.py":                  "scripts/generate_synthetic_data.py",
    "requirements.txt":                                    "requirements.txt",
    "data/README.md":                                      "data/README.md",
    "data/sample_accessions.csv":                          "data/sample_accessions.csv",
    "notebooks/pb2_host_prediction.ipynb":                 "notebooks/pb2_host_prediction.ipynb",
    "hf_model_card.md":                                    "README.md",   # model card == repo README
}

# Belt and braces: reject these no matter what ALLOWED says.
DENIED_SUFFIXES = {".npz", ".npy", ".fasta", ".fa", ".pt", ".pth", ".ckpt"}
DENIED_NAMES = {"metadata.csv"}

WEIGHTS = "results/baseline/pb2_rf_classifier.joblib"


def main() -> int:
    root = Path.cwd()

    if not (root / "requirements.txt").exists():
        print("ERROR: run this from the repository root (requirements.txt not found here).")
        return 1

    if not (root / WEIGHTS).exists():
        print(f"ERROR: {WEIGHTS} does not exist.")
        print("Train it first:")
        print("    python scripts/train_from_cache.py --npz <...> --meta <...>")
        return 1

    # Safety guard
    for local in ALLOWED:
        p = Path(local)
        if p.suffix.lower() in DENIED_SUFFIXES or p.name.lower() in DENIED_NAMES:
            print(f"ERROR: refusing to upload {local} — GISAID-derived file type.")
            return 1

    present, missing = {}, []
    for local, remote in ALLOWED.items():
        if (root / local).exists():
            present[local] = remote
        else:
            missing.append(local)

    if missing:
        print("Skipping (not present locally):")
        for m in missing:
            print(f"    {m}")
        print()

    print(f"Uploading {len(present)} files to {REPO_ID}\n")

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)

    for local, remote in present.items():
        size_kb = (root / local).stat().st_size / 1024
        print(f"  {local}  ({size_kb:,.0f} KB)  ->  {remote}")
        api.upload_file(
            path_or_fileobj=str(root / local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
        )

    url = f"https://huggingface.co/{REPO_ID}"
    print(f"\nDone: {url}")
    print()
    print("REMAINING MANUAL STEP — enable gating for GISAID compliance:")
    print(f"    {url}/settings")
    print("    -> Gated model -> set to 'Manual review' (or 'Automatic')")
    return 0


if __name__ == "__main__":
    sys.exit(main())

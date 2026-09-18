#!/usr/bin/env python3
"""
train_from_cache.py — train the Stage 1 RF classifier from pre-computed ESM-2 embeddings.

Skips the 25-minute ESM-2 embedding step by loading a cached .npz directly.

Run from the repo root:
    python scripts/train_from_cache.py \
        --npz  /absolute/path/to/esm2_sequence_embeddings.npz \
        --meta /absolute/path/to/metadata.csv

Writes:
    results/baseline/pb2_rf_classifier.joblib
    results/baseline/confusion_matrix_random_forest.png   (if matplotlib is installed)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
RESULTS_DIR = Path("results")


def load_embeddings(npz_path: str) -> dict:
    """Load a dict of {seq_id: 1280-dim vector} from a .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    # np.savez(**dict) stores each key as a named array
    if hasattr(data, "files"):
        emb = {k: data[k] for k in data.files}
        print(f"  Loaded {len(emb)} embeddings from npz")
        return emb
    # np.save of the dict object (allow_pickle=True)
    obj = data.item()
    if isinstance(obj, dict):
        print(f"  Loaded {len(obj)} embeddings (pickle array)")
        return obj
    raise ValueError(
        "Unrecognised .npz format. Expected np.savez(**embeddings_dict) "
        "or np.save with allow_pickle=True."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train RF host classifier from cached ESM-2 embeddings."
    )
    parser.add_argument("--npz",  required=True, help="Path to esm2_sequence_embeddings.npz")
    parser.add_argument("--meta", required=True, help="Path to metadata.csv")
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("Loading cached embeddings...")
    sequence_embeddings = load_embeddings(args.npz)
    sample_vec = next(iter(sequence_embeddings.values()))
    print(f"  Embedding shape: {sample_vec.shape}")  # expect (1280,)

    print("\nLoading metadata...")
    metadata_df = pd.read_csv(args.meta)
    print(metadata_df["host_type"].value_counts().to_string())

    # ── 2. Build X / y ───────────────────────────────────────────────────────
    seq_ids = list(sequence_embeddings.keys())
    X = np.array([sequence_embeddings[sid] for sid in seq_ids])

    label_dict = dict(zip(metadata_df["seq_id"], metadata_df["host_type"]))
    missing = [sid for sid in seq_ids if sid not in label_dict]
    if missing:
        print(f"\nWARNING: {len(missing)} seq_ids not found in metadata — dropping them")
        seq_ids = [s for s in seq_ids if s in label_dict]
        X = np.array([sequence_embeddings[sid] for sid in seq_ids])

    y = np.array([label_dict[sid] for sid in seq_ids])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    # Expected: avian=0, human=1, mammal=2 — must match the order in predict.py

    # ── 3. Stratified 60 / 20 / 20 split ─────────────────────────────────────
    idx = np.arange(len(seq_ids))
    idx_temp, idx_test = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_enc[idx_temp]
    )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y_enc[idx_train], y_enc[idx_val], y_enc[idx_test]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── 4. Train Random Forest ────────────────────────────────────────────────
    print("\nTraining Random Forest (100 trees)...")
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_val   = rf.predict(X_val)
    y_pred_test  = rf.predict(X_test)

    print(f"Train acc : {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Val acc   : {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Test acc  : {accuracy_score(y_test, y_pred_test):.4f}")
    print()
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # ── 5. Save weights ───────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pb2_rf_classifier.joblib"
    joblib.dump(rf, out_path)
    print(f"Saved: {out_path}")

    # ── 6. Confusion matrix ───────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred_test)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                plt.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]:.2f})",
                         ha="center", va="center", fontsize=9, color="gray")
        plt.title("Frozen ESM-2 + Random Forest — Test Set")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        fig_path = out_dir / "confusion_matrix_random_forest.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")
    except ImportError:
        print("(matplotlib/seaborn not available — confusion matrix plot skipped)")

    print("\nDone. Model written to results/baseline/pb2_rf_classifier.joblib")
    print("Upload that file to Hugging Face — do NOT upload the .npz or metadata.csv.")


if __name__ == "__main__":
    main()

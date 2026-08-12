# Predicting Influenza A Host Species from PB2 Sequences with ESM-2

Three-way host classification — **human / avian / other mammal** — from the amino-acid
sequence of the influenza A PB2 segment, using the ESM-2 protein language model.

Best test accuracy: **90.97%** (chance = 33.3%), from a random forest over *frozen*
ESM-2 embeddings. Fine-tuning the transformer did not improve on this. That negative
result is the most interesting part of the project and is analysed rather than hidden.

---

## Background

PB2 is the canonical host-range determinant in influenza A. Mutations at residues 627 and
701 are among the best-characterised markers of avian → mammalian adaptation, so if any
single genome segment carries host signal in its primary sequence, it is this one.

ESM-2 is a BERT-style masked language model trained on ~65M protein sequences instead of
text. It embeds each amino acid as a 1280-dim contextual vector. The question this project
asks is a transfer-learning one: **how much task-specific adaptation does a protein
language model actually need to solve a host-prediction task?**

## Dataset

| | |
|---|---|
| Source | GISAID EpiFlu |
| Subtype | Influenza A / H5N1 (all 1602 sequences) |
| Segment | PB2 |
| Sequences | 1602 — 534 human, 534 avian, 534 mammal (downsampled to balance) |
| Typical length | 759 residues |
| Split | Stratified 60 / 20 / 20 → 960 train, 321 val, 321 test |

The mammal class spans dairy cattle, swine, felids, and marine mammals — the H5N1 spillover
hosts of the 2024–25 outbreak.

**The sequences themselves are not in this repository.** GISAID's terms prohibit
redistribution, so none are included here. `data/sample_accessions.csv` documents the
expected input format using synthetic records; see [`data/README.md`](data/README.md) for
how to obtain real data with your own GISAID access.

## Methods

Three stages, in increasing order of how much the model is allowed to adapt:

**1. Frozen embeddings + classical classifiers.** Mean-pool ESM-2's final-layer residue
embeddings into one 1280-dim vector per sequence, then fit logistic regression (tests
linear separability) and a random forest (allows nonlinear splits). Nothing in ESM-2
trains.

**2. Frozen embeddings + MLP head.** Same vectors, a learned 1280 → 512 → 3 MLP with
dropout, Adam, and best-checkpoint selection on validation accuracy.

**3. End-to-end fine-tuning.** Unfreeze the top transformer blocks (last 3 of 33, and a
last-8 variant), replace mean-pooling with **learned attention pooling** — motivated by
the fact that host adaptation is driven by a few specific residues that an average over
759 positions would dilute — and add a four-layer classifier head with batch norm.
AdamW (wd 0.01), gradient clipping at 1.0, cosine-annealed LR, early stopping.

## ⚠️ Data disclaimer — read before running

**This repository ships with sample data only.** The results reported below were produced
on real influenza sequences from the GISAID EpiFlu database, which cannot be
redistributed under its access agreement. No real sequences, accessions, or strain
identifiers appear anywhere in this repo.

What that means in practice:

- **The numbers in "Original results" below came from real GISAID data.** They are the
  actual findings of the project.
- **Running the notebook as-shipped will not reproduce them.** It runs on generated
  sample data (`scripts/generate_synthetic_data.py`), which exists so the pipeline is
  executable, not so it is reproducible.
- **Any accuracy you obtain from sample data is a smoke test**, confirming the code runs
  end to end. It reflects signal that was deliberately planted by the generator and says
  nothing about influenza biology. Do not cite it, quote it, or compare it to the table
  below.
- All sample identifiers are prefixed `SYNTH` so the two can never be confused.

To reproduce the original results you need your own GISAID access — see
[`data/README.md`](data/README.md).

## Original results

Produced on the real GISAID dataset: 1602 H5N1 PB2 sequences, 534 per host class,
stratified 60/20/20 split.

| Stage | Model | What trains | Test accuracy |
|---|---|---|---|
| 1 | Frozen ESM-2 + logistic regression | classifier only | 83.80% |
| 1 | **Frozen ESM-2 + random forest** | classifier only | **90.97%** |
| 2 | Frozen ESM-2 + MLP head | 2-layer MLP | 84.11% |
| 3 | Fine-tuned ESM-2 (freeze 30, tune 3) | top 3 layers + head | 89.10% |

Chance is 33.3% on the balanced three-class problem.

Confusion matrices and the full fine-tuning log — both generated from the real data — are
in [`results/`](results/). Those artifacts are the primary evidence for the numbers above,
since the underlying sequences can't be shared.

> **Note on the fine-tuned figure.** `results/finetune/confusion_matrix_esm-2_fine-tuned.png`
> sums to 289/321 = 90.03%, which does not match the 89.10% in the table. The two come from
> different runs: the table and `training_log_freeze30.txt` are the freeze-30 configuration
> (last 3 layers trained), while the saved figure is from a freeze-25 run (last 8 layers).
> Both are reported rather than silently reconciled. The gap between them — ~0.9 points, or
> three test sequences — is itself a decent illustration of the run-to-run variance that
> makes the single-split results below untrustworthy as a ranking.

### Why didn't fine-tuning win?

1. **Too little data.** 960 training sequences against ~60M trainable parameters even with
   30 of 33 layers frozen. The training log shows accuracy pinned at ~0.82 for seven
   consecutive epochs before moving — underfitting, then late memorisation.
2. **The signal may already be in the frozen features.** If ESM-2's pretrained
   representation of residues 627/701 already differs by host, a random forest can read it
   straight out of the pooled vector and fine-tuning has little left to add.
3. **Random forests suit this regime.** 1602 samples × 1280 features is a sample-to-
   dimension ratio where axis-aligned ensembles do well and gradient-trained deep heads
   struggle.

### Limitations

Stated plainly, because they bound how much the numbers above mean:

- **Single split, single seed.** All results come from one 60/20/20 split at
  `random_state=42`. On 321 test sequences, 89.10% vs 90.97% is a difference of ~6
  sequences — inside plausible seed variation. **The model ranking is not statistically
  established.**
- **No sequence-identity deduplication.** Influenza sequences from one outbreak are
  near-identical; if similar sequences straddle the train/test boundary, every accuracy
  here is inflated. This is the biggest threat to validity.
- **Truncated sequences retained** — partial submissions down to 32 residues, and
  truncation rate may correlate with host.
- **H5N1 only** — no evidence of generalisation to other subtypes.

### Next steps

1. 5-fold stratified CV across seeds, reporting mean ± std.
2. Cluster at 95% identity (CD-HIT / MMseqs2) and split by cluster, not by sequence.
3. Plot attention-pooling weights against residue position — do they land on 627 and 701?
4. Trivial baseline: how far do positions 627 + 701 alone get you, one-hot encoded?
5. Drop sequences under ~700 residues and re-run.

## Repository layout

```
pb2-host-prediction/
├── notebooks/
│   └── pb2_host_prediction.ipynb   # full pipeline, preprocessing → fine-tuning
├── scripts/
│   └── generate_synthetic_data.py  # synthetic stand-in so the pipeline is runnable
├── data/
│   ├── sample_accessions.csv       # synthetic records showing expected format
│   └── README.md                   # data provenance + how to obtain real data
├── results/                        # generated from the REAL data — the evidence
│   ├── baseline/                   # LR + RF confusion matrices, contact maps, summary
│   └── finetune/                   # fine-tuned + MLP head matrices, training log
└── requirements.txt
```

## Running it without GISAID access

Because the sequences can't be redistributed, the repo ships a generator that produces a
synthetic dataset with the same shape, header format, and class balance:

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
```

Then set `USE_SYNTHETIC_DATA = True` in the notebook's config cell and run top to bottom.

**These are not the results in this README.** The sequences are generated, not observed.
Host signal is deliberately planted at the residues that drive real PB2 adaptation —
notably 627 (the classic E627K avian→human marker) and 701 — with sub-100% penetrance, so
the task is learnable but not trivially separable. Purely random sequences would leave
every classifier at chance and make the pipeline look broken; this instead exercises the
full code path. Any accuracy obtained this way is a smoke test, not a finding, and every
synthetic header is prefixed `SYNTH` so the two datasets can never be confused.

## Reproducing the real results

```bash
pip install -r requirements.txt
```

Obtain the FASTA files from GISAID (see `data/README.md`), place the balanced file at
`data/all_pb2_sequences_labeled_balanced.fasta`, leave `USE_SYNTHETIC_DATA = False`, then
run the notebook top to bottom.

The two expensive stages are gated behind flags — set `RUN_ATTENTION_ANALYSIS = True` and
`RUN_FINETUNING = True` to enable them. A GPU is required: embedding extraction over 1602
sequences is the bottleneck, and fine-tuning the 650M checkpoint needs ~16GB VRAM at
`batch_size=4`. Originally run on Colab (T4 / A100) and an HPC node.

## Acknowledgements

Sequence data from the **GISAID EpiFlu** database. We gratefully acknowledge the
originating and submitting laboratories who share sequences through GISAID.

ESM-2: Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a
language model*, Science (2023). Model: `esm2_t33_650M_UR50D` via
[facebookresearch/esm](https://github.com/facebookresearch/esm).

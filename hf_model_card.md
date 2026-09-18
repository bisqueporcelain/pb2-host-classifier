---
license: cc-by-nc-4.0
extra_gated_heading: "Request Access to PB2 Host Classifier"
extra_gated_prompt: "By downloading this model you confirm that you have independently agreed to the GISAID Terms of Use and that you will use this model only for non-commercial research purposes consistent with those terms."
---
- en
tags:
- biology
- protein
- influenza
- esm2
- host-prediction
- bioinformatics
datasets:
- gisaid
model-index:
- name: pb2-host-classifier
  results:
  - task:
      type: text-classification
      name: Protein sequence classification
    dataset:
      name: GISAID EpiFlu (H5N1, PB2 segment, 1602 sequences)
      type: gisaid
    metrics:
    - type: accuracy
      value: 0.9097
      name: Test accuracy (frozen ESM-2 + random forest, 60/20/20 split)
---

# pb2-host-classifier

**Task:** Given the amino-acid sequence of the influenza A PB2 polymerase subunit,
predict whether the virus was isolated from a **human**, **avian**, or **other mammal**
host.

**Best model:** A random forest trained on frozen mean-pooled ESM-2 embeddings reaches
**90.97% test accuracy** on a balanced three-class problem (chance = 33.3%).
Fine-tuning the transformer did not improve on this—that negative result is the most
interesting finding of the project, and the repository analyses it directly.

---

## ⚠️ Data and access notice

This model was trained on influenza A / H5N1 PB2 sequences from the
[GISAID EpiFlu](https://gisaid.org) database. GISAID's Terms of Use prohibit
redistribution of sequences and direct derivatives.

**By downloading this model you confirm that you have independently agreed to the
[GISAID Terms of Use](https://gisaid.org/terms-of-use/) and that you will use this
model only for non-commercial research purposes consistent with those terms.**

No sequences, accessions, strain identifiers, or per-sequence embeddings are
included in this repository or in the model weights. The random forest encodes
split thresholds over mean-pooled 1280-dimensional ESM-2 representations; the
original sequences are not recoverable from it.

---

## Quick start

```bash
git clone https://huggingface.co/bisqueporcelain/pb2-host-classifier
cd pb2-host-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/predict.py MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTSGRQEKNPSLRMKWMMAMKY...
```

Or as a library:

```python
from scripts.predict import predict_host

result = predict_host("MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTS...")
# {
#   "host": "human",
#   "probabilities": {"avian": 0.03, "human": 0.91, "mammal": 0.06},
#   "sequence_length": 759
# }
```

The model file (`results/baseline/pb2_rf_classifier.joblib`) is checked in here.
ESM-2 (`esm2_t33_650M_UR50D`, ~2.5 GB) is downloaded automatically by `fair-esm`
on first run and cached under `~/.cache/torch/hub/checkpoints/`.

---

## Model details

| | |
|---|---|
| Architecture | Random forest (scikit-learn, 100 trees) over frozen mean-pooled ESM-2 embeddings |
| Base model | `esm2_t33_650M_UR50D` (Meta / FAIR, 650M parameters) |
| Input | Single-letter amino-acid sequence; mean-pooled to 1280-dim vector |
| Output | Three-class softmax: `human`, `avian`, `mammal` |
| Training data | 1602 H5N1 PB2 sequences from GISAID EpiFlu, 534 per class (downsampled to balance) |
| Split | Stratified 60 / 20 / 20, `random_state=42` |

### Results

| Stage | Model | Test accuracy |
|---|---|---|
| 1 | Frozen ESM-2 + logistic regression | 83.80% |
| 1 | **Frozen ESM-2 + random forest** | **90.97%** |
| 2 | Frozen ESM-2 + MLP head | 84.11% |
| 3 | Fine-tuned ESM-2 (freeze 30/33 layers) | 89.10% |

Chance = 33.3%.

### Why the random forest wins

Fine-tuning 60M+ transformer parameters on 960 training sequences produces
underfitting, not adaptation. The pretrained ESM-2 representation of key
adaptation residues (especially 627 and 701) already differs by host—a random
forest can read that directly from the pooled vector, leaving little for
gradient-based fine-tuning to add. See the notebook discussion section for the
full analysis.

---

## Limitations

These bound how much the numbers above mean.

- **Single split, single seed.** Every result comes from one 60/20/20 split at
  `random_state=42`. On 321 test sequences the gap between the top two models is
  ~6 sequences—well inside plausible seed variation. **The ranking is not
  statistically established** without cross-validation.
- **No sequence-identity deduplication.** Influenza sequences from the same outbreak
  are near-identical. If similar sequences straddle the train/test boundary, all
  accuracies here are inflated. Clustering at 95% identity before splitting is the
  first thing to fix.
- **H5N1 only.** No evidence of generalisation to other subtypes.
- **"Mammal" is heterogeneous**—dairy cattle, swine, felids, marine mammals with
  different adaptation histories pooled into one class.
- **Partial sequences retained.** Submissions as short as 32 residues remain;
  truncation rate may correlate with host class.

---

## Reproducing results

Weights were generated from real GISAID data (requires your own GISAID access).
The repository ships a synthetic-data generator so the full pipeline is runnable
without GISAID access:

```bash
python scripts/generate_synthetic_data.py
# then in the notebook: USE_SYNTHETIC_DATA = True
```

Synthetic results exercise the code path only. They are not the numbers in this
card.

See [`data/README.md`](data/README.md) for instructions on obtaining real data and
[`notebooks/pb2_host_prediction.ipynb`](notebooks/pb2_host_prediction.ipynb) for
the full training pipeline.

---

## Citation / acknowledgements

Sequence data from the **GISAID EpiFlu** database. We gratefully acknowledge the
originating and submitting laboratories.

- **ESM-2:** Lin et al., *Evolutionary-scale prediction of atomic-level protein
  structure with a language model*, Science (2023).
  [`facebookresearch/esm`](https://github.com/facebookresearch/esm)
- **scikit-learn:** Pedregosa et al., JMLR (2011).

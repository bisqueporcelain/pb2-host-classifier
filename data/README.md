# Data

## What's in this folder

This repository contains no real sequence data. The original analysis used influenza
A PB2 sequences from the GISAID EpiFlu database, whose access agreement prohibits
redistribution. No GISAID sequences, accessions, or strain identifiers are included here.

| File | What it is |
|---|---|
| `sample_accessions.csv` | 1602 **synthetic** accession records, matching the real dataset's shape and class balance. All IDs are prefixed `SYNTH`. |

## `sample_accessions.csv`

1602 rows, 534 per host class:

| Column | Description |
|---|---|
| `epi_id` | Synthetic sequence ID (`SYNTH0000001`) |
| `epi_isl` | Synthetic isolate ID (`SYNTH_ISL_00000001`) |
| `strain` | Synthetic strain name (`A/Synthetic_dairy_cow/SYN/000123/2024`) |
| `subtype` | `A_/_H5N1` throughout |
| `host_type` | `human`, `avian`, or `mammal` |
| `sequence_length` | Residue count |

It documents the expected input format only and does not contain real sequence data.

## Generating runnable data

To execute the notebook end to end:

```bash
python scripts/generate_synthetic_data.py
```

This writes `data/synthetic_pb2_sequences.fasta`: 1602 generated sequences with the same
header format, length distribution, and class balance as the original dataset. Set
`USE_SYNTHETIC_DATA = True` in the notebook config cell to use it.

Host signal is planted at the residues that drive real PB2 adaptation (627, 701, and
three others) with sub-100% penetrance, so the classifiers learn a genuine signal rather
than performing at chance. Accuracy obtained this way tests the code path only and
carries no scientific meaning; see the disclaimer in the main README.

## Using real data

If you have GISAID access and want to reproduce the original numbers:

1. Register at [gisaid.org](https://gisaid.org) and accept the Database Access Agreement.
2. In **EpiFlu → Search**, filter to influenza A, H5N1, **PB2** segment.
3. Download three FASTA files, one per host class (human / avian / other mammal).
4. Run section 1 of the notebook (`add_host_label` → `concatenate_labeled` →
   `downsample_fasta`) to label, merge, and balance them to 534 per class.
5. Save the result as `data/all_pb2_sequences_labeled_balanced.fasta` and leave
   `USE_SYNTHETIC_DATA = False`.

Expected header format:

```
>SEQID|PB2|A/strain/name/here/2024|ISOLATE_ID|A_/_H5N1|HOST_0
```

where `HOST_0` = human, `HOST_1` = avian, `HOST_2` = mammal.

## Do not commit real data

`.gitignore` excludes `*.fasta`, `*.npz`, and `metadata.csv` from this directory, so
downloaded real sequences are not committed by accident. Do not override this:
publishing GISAID sequences violates the access agreement.

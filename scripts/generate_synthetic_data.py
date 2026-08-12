#!/usr/bin/env python3
"""
Generate a SYNTHETIC stand-in for the GISAID PB2 dataset.

Why this exists
---------------
The real sequences come from GISAID EpiFlu and cannot be redistributed (see
data/README.md). Without them the notebook cannot run at all, which makes the
repository read-only for anyone who clones it. This script produces a fake FASTA
with the same shape, header format, and class balance so the full pipeline is
executable end to end.

What this is NOT
----------------
This does not reproduce the results in the README. The sequences are generated,
not observed. Accuracies obtained on synthetic data are a smoke test that the
code path works -- they say nothing about influenza biology and should never be
cited as findings.

How the signal is planted
-------------------------
Purely random sequences would carry no host signal, so every classifier would sit
at chance (33%) and the pipeline would look broken. Instead we plant differences
at positions that genuinely drive PB2 host adaptation, most notably residue 627
(the classic E627K avian->human marker) and 701. Penetrance is set below 100% so
the task is learnable but not trivially separable, which is roughly the regime the
real data occupies.

Every generated header is prefixed SYNTH so synthetic data can never be silently
confused with the real thing.

Usage
-----
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-per-host 100 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Background amino-acid frequencies, roughly UniProt-wide.
AA_FREQ = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
    "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
    "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
}
AMINO_ACIDS = list(AA_FREQ)
AA_WEIGHTS = [AA_FREQ[a] for a in AMINO_ACIDS]

PB2_LENGTH = 759
HOST_CODES = {"human": 0, "avian": 1, "mammal": 2}

# Position (1-indexed) -> {host: (residue, penetrance)}
#
# 627 and 701 are the well-characterised mammalian-adaptation markers. The others
# are invented positions included so the task is not solvable from two residues
# alone -- they make the synthetic problem structurally similar to the real one.
SIGNAL_POSITIONS = {
    627: {"avian": ("E", 0.92), "human": ("K", 0.88), "mammal": ("K", 0.72)},
    701: {"avian": ("D", 0.90), "human": ("N", 0.70), "mammal": ("N", 0.80)},
    271: {"avian": ("T", 0.85), "human": ("A", 0.75), "mammal": ("A", 0.65)},
    590: {"avian": ("G", 0.80), "human": ("S", 0.70), "mammal": ("S", 0.78)},
    591: {"avian": ("Q", 0.82), "human": ("Q", 0.60), "mammal": ("R", 0.68)},
}

# Truncated submissions exist in the real data; mirror that so length-handling
# code (masking, pooling) is exercised rather than silently assumed away.
TRUNCATION_RATE = 0.03
MIN_TRUNCATED_LENGTH = 120

STRAIN_HINTS = {
    "human": ["Synthetic_human"],
    "avian": ["Synthetic_chicken", "Synthetic_duck", "Synthetic_tern"],
    "mammal": ["Synthetic_dairy_cow", "Synthetic_swine", "Synthetic_cat"],
}


def make_backbone(rng: random.Random) -> list[str]:
    """A shared 'consensus' PB2 backbone, so sequences resemble one protein family."""
    return rng.choices(AMINO_ACIDS, weights=AA_WEIGHTS, k=PB2_LENGTH)


def make_sequence(backbone: list[str], host: str, rng: random.Random,
                  drift_rate: float = 0.04) -> str:
    """One synthetic PB2: shared backbone + random drift + planted host signal."""
    seq = list(backbone)

    # Neutral drift so sequences are not identical to each other.
    for i in range(len(seq)):
        if rng.random() < drift_rate:
            seq[i] = rng.choices(AMINO_ACIDS, weights=AA_WEIGHTS, k=1)[0]

    # Planted host-associated residues, applied with sub-100% penetrance.
    for position, host_map in SIGNAL_POSITIONS.items():
        residue, penetrance = host_map[host]
        if rng.random() < penetrance:
            seq[position - 1] = residue

    if rng.random() < TRUNCATION_RATE:
        seq = seq[:rng.randint(MIN_TRUNCATED_LENGTH, PB2_LENGTH)]

    return "".join(seq)


def generate(output_path: Path, n_per_host: int, seed: int) -> None:
    rng = random.Random(seed)
    backbone = make_backbone(rng)

    records = []
    counter = 1
    for host, host_code in HOST_CODES.items():
        for _ in range(n_per_host):
            seq = make_sequence(backbone, host, rng)
            hint = rng.choice(STRAIN_HINTS[host])
            year = rng.randint(2021, 2025)
            header = (
                f"SYNTH{counter:07d}|PB2|"
                f"A/{hint}/SYN/{counter:06d}/{year}|"
                f"SYNTH_ISL_{counter:08d}|A_/_H5N1|HOST_{host_code}"
            )
            records.append((header, seq))
            counter += 1

    rng.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    lengths = [len(s) for _, s in records]
    print(f"Wrote {len(records)} SYNTHETIC sequences to {output_path}")
    print(f"  {n_per_host} per host (human / avian / mammal)")
    print(f"  Length: min {min(lengths)}, max {max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.1f}")
    print(f"  Seed: {seed}")
    print()
    print("  These sequences are generated, not observed. Any accuracy obtained")
    print("  from them is a smoke test of the code path, not a scientific result.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PB2 sequences for testing the pipeline.")
    parser.add_argument("--n-per-host", type=int, default=534,
                        help="Sequences per host class (default: 534, matching the real data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "data" / "synthetic_pb2_sequences.fasta",
                        help="Output FASTA path")
    args = parser.parse_args()

    generate(args.output, args.n_per_host, args.seed)


if __name__ == "__main__":
    main()

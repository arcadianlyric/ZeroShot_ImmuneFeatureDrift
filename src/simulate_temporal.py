"""
Simulate longitudinal temporal data from a single count table.

Takes chr22 real PBMC count table (2024 baseline) and generates
2025 / 2026 samples by adding biologically plausible temporal drift:
  - Global expression noise (Poisson resampling)
  - Immune-aging signal: gradual decline in key immune genes
  - Stochastic gene-level drift (fold-change perturbation)

Usage:
  python simulate_temporal.py \
      --input data/chr22_gencode_counts.txt \
      --output-dir data/longitudinal \
      --format featurecounts
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Immune-relevant genes expected to show aging-related changes
IMMUNE_DECLINE_GENES = {
    # T-cell related - gradual decline with age
    "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4",
    "TCF7", "LEF1", "CCR7", "IL7R", "CD27", "CD28",
    # Naive T-cell markers
    "SELL", "PTPRC",  # CD62L, CD45
}

IMMUNE_INCREASE_GENES = {
    # Inflammatory / senescence markers - increase with age
    "KLRG1", "TIGIT", "LAG3", "GZMB", "PRF1", "NKG7",
    "FCGR3A",  # CD16
    "S100A8", "S100A9", "S100A12",  # Alarmin / inflammation
}


def simulate_temporal_samples(
    count_table_path: str,
    output_dir: str = "./data/longitudinal",
    format: str = "auto",
    years: list = None,
    drift_scale: float = 0.08,
    seed: int = 42,
) -> dict:
    """
    Generate longitudinal samples from a single count table.

    Args:
        count_table_path: Path to featureCounts or CSV count table (single sample).
        output_dir: Directory to write per-sample CSV files.
        format: "featurecounts", "csv", or "auto".
        years: List of year labels (default: ["2024", "2025", "2026"]).
        drift_scale: Per-year fold-change noise scale (log-normal sigma).
        seed: Random seed for reproducibility.

    Returns:
        Dict[sample_id -> DataFrame(co_barcode_cluster_id, gene_id, count)]
    """
    if years is None:
        years = ["2024", "2025", "2026"]

    np.random.seed(seed)
    path = Path(count_table_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load count table ---
    with open(path, 'r') as f:
        first_line = f.readline()

    if format == "auto":
        format = "featurecounts" if first_line.startswith("#") else "csv"

    if format == "featurecounts":
        df = pd.read_csv(path, sep='\t', comment='#')
        meta_cols = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length']
        sample_cols = [c for c in df.columns if c not in meta_cols]
        genes = df['Geneid'].values
        base_counts = df[sample_cols[0]].values.astype(float)
    else:
        sep = '\t' if '\t' in first_line else ','
        df = pd.read_csv(path, sep=sep, index_col=0)
        genes = df.index.values
        base_counts = df.iloc[:, 0].values.astype(float)

    # Filter to non-zero genes
    mask = base_counts > 0
    genes = genes[mask]
    base_counts = base_counts[mask]
    logger.info(f"Loaded {len(genes)} non-zero genes from {path.name}")

    # --- Categorize genes ---
    decline_idx = np.array([g in IMMUNE_DECLINE_GENES for g in genes])
    increase_idx = np.array([g in IMMUNE_INCREASE_GENES for g in genes])
    n_decline = decline_idx.sum()
    n_increase = increase_idx.sum()
    logger.info(f"Immune-decline genes found: {n_decline}, increase genes found: {n_increase}")

    samples_data = {}

    for i, year in enumerate(years):
        years_elapsed = i  # 0 for baseline

        if years_elapsed == 0:
            # Baseline: use original counts directly
            counts = base_counts.copy()
        else:
            # 1) Global stochastic drift: log-normal fold-change per gene
            fold_changes = np.random.lognormal(
                mean=0, sigma=drift_scale * years_elapsed, size=len(genes)
            )

            # 2) Immune-aging signal: decline genes lose ~5-10% per year
            aging_decline = np.ones(len(genes))
            aging_decline[decline_idx] = max(0.3, 1.0 - 0.07 * years_elapsed)

            # 3) Inflammatory increase: ~5-8% per year
            aging_increase = np.ones(len(genes))
            aging_increase[increase_idx] = 1.0 + 0.06 * years_elapsed

            # Combine: base * fold_change * aging
            expected = base_counts * fold_changes * aging_decline * aging_increase

            # 4) Poisson resampling for count-level noise
            expected = np.clip(expected, 0, None)
            counts = np.random.poisson(lam=expected.astype(float))

        # Build pipeline-compatible DataFrame
        counts_int = counts.astype(int)
        nonzero = counts_int > 0
        sample_df = pd.DataFrame({
            'co_barcode_cluster_id': genes[nonzero],
            'gene_id': genes[nonzero],
            'count': counts_int[nonzero],
        })

        # Save to CSV
        csv_path = output_path / f"sample_{year}.csv"
        sample_df.to_csv(csv_path, index=False)
        samples_data[year] = sample_df
        logger.info(f"  {year}: {len(sample_df)} genes, "
                     f"total counts = {counts_int[nonzero].sum():,}")

    # Also save a combined featureCounts-like table for reference
    combined = pd.DataFrame({'Geneid': genes})
    for year in years:
        s = samples_data[year].set_index('gene_id')['count']
        combined[year] = combined['Geneid'].map(s).fillna(0).astype(int)
    combined.to_csv(output_path / "combined_counts.tsv", sep='\t', index=False)
    logger.info(f"Saved combined count table to {output_path / 'combined_counts.tsv'}")

    return samples_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate longitudinal temporal data from a single count table"
    )
    parser.add_argument("--input", required=True, help="Path to count table")
    parser.add_argument("--output-dir", default="./data/longitudinal")
    parser.add_argument("--format", default="auto", choices=["auto", "featurecounts", "csv"])
    parser.add_argument("--drift-scale", type=float, default=0.08,
                        help="Per-year log-normal drift sigma (default: 0.08)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    simulate_temporal_samples(
        count_table_path=args.input,
        output_dir=args.output_dir,
        format=args.format,
        drift_scale=args.drift_scale,
        seed=args.seed,
    )

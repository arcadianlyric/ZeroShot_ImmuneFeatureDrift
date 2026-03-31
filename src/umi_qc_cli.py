#!/usr/bin/env python3
"""
Standalone UMI / Co-barcode QC Tool

Runs ML-based quality control (Isolation Forest) on UMI or stLFR co-barcode
count distributions. Detects PCR bias, optical duplicates, and contamination
artifacts without hard-coded thresholds.

Usage examples:
  # Single sample (CSV with columns: co_barcode_cluster_id, count)
  python umi_qc_cli.py --input counts.csv --output qc_results.json

  # Single sample from featureCounts output
  python umi_qc_cli.py --input feature_count.txt --format featurecounts --output qc_results.json

  # Multiple samples (directory of CSVs or a multi-column count table)
  python umi_qc_cli.py --input-dir ./samples/ --output-dir ./qc_results/

  # Multi-sample featureCounts (each column = one sample)
  python umi_qc_cli.py --input feature_count.txt --format featurecounts --output-dir ./qc_results/

  # Adjust contamination threshold
  python umi_qc_cli.py --input counts.csv --contamination 0.05 --output qc_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from same directory
sys.path.insert(0, str(Path(__file__).parent))

from co_barcode_qc import CoBarcodeQCAnalyzer
from stlfr_preprocess import load_count_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("umi-qc")


def load_single_sample(input_path: str, format: str = "auto") -> pd.DataFrame:
    """
    Load a single sample's UMI/co-barcode count data.

    Supported formats:
      - CSV/TSV with columns [co_barcode_cluster_id, count] (or [gene_id, count])
      - featureCounts output (first sample column used)
      - Two-column file: name + count (auto-detect)
    """
    path = Path(input_path)

    with open(path, 'r') as f:
        first_line = f.readline()

    if format == "auto":
        if first_line.startswith("#") or "featureCounts" in first_line:
            format = "featurecounts"
        else:
            format = "csv"

    if format == "featurecounts":
        # Use load_count_table to parse; take the first sample
        samples = load_count_table(input_path, format="featurecounts")
        if not samples:
            raise ValueError(f"No samples found in {input_path}")
        first_sid = list(samples.keys())[0]
        df = samples[first_sid]
        logger.info(f"Loaded sample '{first_sid}' from featureCounts ({len(df)} genes)")
        return df

    # CSV/TSV
    sep = '\t' if '\t' in first_line else ','
    df = pd.read_csv(path, sep=sep)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('co_barcode_cluster_id', 'cluster_id', 'umi', 'barcode', 'gene_id', 'geneid', 'gene'):
            col_map[c] = 'co_barcode_cluster_id'
        elif cl in ('count', 'counts', 'read_count', 'reads'):
            col_map[c] = 'count'
    df = df.rename(columns=col_map)

    if 'co_barcode_cluster_id' not in df.columns:
        # Use first column as ID
        df = df.rename(columns={df.columns[0]: 'co_barcode_cluster_id'})
    if 'count' not in df.columns:
        # Use second (or last numeric) column as count
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df.rename(columns={numeric_cols[0]: 'count'})
        else:
            raise ValueError(f"Cannot find a numeric 'count' column in {input_path}")

    df = df[['co_barcode_cluster_id', 'count']].dropna()
    df['count'] = df['count'].astype(int)
    df = df[df['count'] > 0].reset_index(drop=True)

    logger.info(f"Loaded {len(df)} entries from {path}")
    return df


def run_single_qc(
    input_path: str,
    output_path: str,
    format: str = "auto",
    contamination: float = 0.1,
    method: str = "isolation_forest",
):
    """Run QC on a single sample file."""
    df = load_single_sample(input_path, format=format)
    sample_id = Path(input_path).stem

    qc = CoBarcodeQCAnalyzer(
        method=method,
        contamination=contamination,
    )
    qc.fit(df['count'].values.reshape(-1, 1))

    stats = qc.analyze_sample(df, sample_id)
    high_abundance = qc.detect_high_abundance_clusters(df)
    randomness = qc.check_randomness(df)

    result = {
        "sample_id": sample_id,
        "input_file": str(input_path),
        "distribution_stats": stats,
        "high_abundance_clusters": high_abundance[:10],
        "randomness_test": randomness,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary to stdout
    _print_summary(result)

    return result


def run_multi_qc(
    input_source: str,
    output_dir: str,
    format: str = "auto",
    contamination: float = 0.1,
    method: str = "isolation_forest",
    glob_pattern: str = "*.csv",
):
    """
    Run QC on multiple samples.
    input_source can be:
      - A directory of per-sample CSV files
      - A multi-column featureCounts file
    """
    input_path = Path(input_source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all samples
    if input_path.is_dir():
        samples_data = {}
        for fp in sorted(input_path.glob(glob_pattern)):
            df = load_single_sample(str(fp), format=format)
            samples_data[fp.stem] = df
    elif input_path.is_file():
        samples_data = load_count_table(str(input_path), format=format)
    else:
        raise FileNotFoundError(f"Input not found: {input_source}")

    if not samples_data:
        raise ValueError(f"No samples loaded from {input_source}")

    logger.info(f"Loaded {len(samples_data)} samples for joint QC")

    # Fit shared model for cross-sample consistency
    qc = CoBarcodeQCAnalyzer(method=method, contamination=contamination)
    qc.fit_longitudinal(samples_data)

    all_results = []
    for sample_id, df in samples_data.items():
        stats = qc.analyze_sample(df, sample_id)
        high_abundance = qc.detect_high_abundance_clusters(df)
        randomness = qc.check_randomness(df)

        result = {
            "sample_id": sample_id,
            "distribution_stats": stats,
            "high_abundance_clusters": high_abundance[:10],
            "randomness_test": randomness,
        }
        all_results.append(result)

        with open(output_dir / f"qc_{sample_id}.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # Write combined results
    with open(output_dir / "qc_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Write summary CSV
    summary_rows = []
    for r in all_results:
        s = r["distribution_stats"]
        summary_rows.append({
            "sample_id": s["sample_id"],
            "total_clusters": s["total_co_barcode_clusters"],
            "outlier_count": s["outlier_count"],
            "outlier_fraction": f"{s['outlier_fraction']:.3f}",
            "cv": f"{s['cv']:.2f}",
            "passed": s["passed"],
            "is_random": r["randomness_test"]["is_random"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "qc_summary.csv", index=False)

    # Print summary
    print("\n=== UMI QC Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {output_dir}/")

    return all_results


def _print_summary(result: dict):
    """Pretty-print a single sample QC result."""
    s = result["distribution_stats"]
    r = result["randomness_test"]
    h = result.get("high_abundance_clusters", [])

    status = "PASS" if s["passed"] else "FAIL"
    print(f"\n{'='*50}")
    print(f"UMI QC Report: {result['sample_id']}  [{status}]")
    print(f"{'='*50}")
    print(f"  Total clusters:    {s['total_co_barcode_clusters']}")
    print(f"  Outlier count:     {s['outlier_count']} ({s['outlier_fraction']:.1%})")
    print(f"  Mean count:        {s['mean_count']:.1f}")
    print(f"  Median count:      {s['median_count']:.1f}")
    print(f"  CV:                {s['cv']:.2f}")
    print(f"  Randomness test:   {'RANDOM' if r['is_random'] else 'NON-RANDOM'} "
          f"(p={r['p_value_randomness']:.4f})")
    if h:
        print(f"  Top high-abundance clusters:")
        for entry in h[:3]:
            print(f"    {entry['co_barcode_cluster_id']}: "
                  f"count={entry['count']}, "
                  f"fold_change={entry['fold_change']:.0f}x, "
                  f"z_score={entry['z_score']:.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="umi-qc",
        description="Standalone ML-based UMI / Co-barcode QC Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sample CSV
  python umi_qc_cli.py --input counts.csv --output qc_report.json

  # Single featureCounts file (first sample column)
  python umi_qc_cli.py --input feature_count.txt --format featurecounts --output qc.json

  # Multi-sample: directory of CSVs
  python umi_qc_cli.py --input-dir ./samples/ --output-dir ./qc/

  # Multi-sample: multi-column featureCounts
  python umi_qc_cli.py --input feature_count.txt --format featurecounts --output-dir ./qc/

  # Stricter threshold (flag top 5% as outliers)
  python umi_qc_cli.py --input counts.csv --contamination 0.05 --output qc.json
        """,
    )

    # Input (mutually exclusive: single file vs directory)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", dest="input_file",
                             help="Single input file (CSV, TSV, or featureCounts output)")
    input_group.add_argument("--input-dir", dest="input_dir",
                             help="Directory of per-sample count files")

    # Output
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file (single-sample mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (multi-sample mode)")

    # Options
    parser.add_argument("--format", "-f", default="auto",
                        choices=["auto", "featurecounts", "csv"],
                        help="Input format (default: auto-detect)")
    parser.add_argument("--contamination", "-c", type=float, default=0.1,
                        help="Expected outlier fraction for Isolation Forest (default: 0.1)")
    parser.add_argument("--method", "-m", default="isolation_forest",
                        choices=["isolation_forest", "autoencoder"],
                        help="QC method (default: isolation_forest)")
    parser.add_argument("--glob", default="*.csv",
                        help="Glob pattern for input directory (default: *.csv)")

    args = parser.parse_args()

    if args.input_file:
        # Single-sample or multi-sample from one file
        if args.output_dir:
            # Multi-sample mode (featureCounts with multiple columns)
            run_multi_qc(
                input_source=args.input_file,
                output_dir=args.output_dir,
                format=args.format,
                contamination=args.contamination,
                method=args.method,
            )
        else:
            output = args.output or f"qc_{Path(args.input_file).stem}.json"
            run_single_qc(
                input_path=args.input_file,
                output_path=output,
                format=args.format,
                contamination=args.contamination,
                method=args.method,
            )
    elif args.input_dir:
        output_dir = args.output_dir or "./qc_results"
        run_multi_qc(
            input_source=args.input_dir,
            output_dir=output_dir,
            format=args.format,
            contamination=args.contamination,
            method=args.method,
            glob_pattern=args.glob,
        )


if __name__ == "__main__":
    main()

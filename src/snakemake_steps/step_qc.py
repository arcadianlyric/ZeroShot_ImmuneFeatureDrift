"""Snakemake step: Load data + ML-based Co-barcode QC."""
import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stlfr_preprocess import load_count_table, generate_mock_stlfr_count_table
from co_barcode_qc import CoBarcodeQCAnalyzer

# --- Load data ---
count_table = snakemake.params.get("count_table")
count_format = snakemake.params.get("count_format", "auto")
data_dir = snakemake.params.get("data_dir", "data/mock")

if count_table and Path(count_table).exists():
    samples_data = load_count_table(count_table, format=count_format)
else:
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("sample_*.csv"))
    if csv_files:
        import pandas as pd
        samples_data = {
            fp.stem.split("_")[1]: pd.read_csv(fp)
            for fp in sorted(csv_files)
        }
    else:
        samples_data = generate_mock_stlfr_count_table(output_dir=data_dir)

# --- Run QC ---
contamination = snakemake.params.get("contamination", 0.1)
method = snakemake.params.get("method", "isolation_forest")

qc = CoBarcodeQCAnalyzer(method=method, contamination=contamination)
qc.fit_longitudinal(samples_data)

all_results = []
for sample_id, df in samples_data.items():
    stats = qc.analyze_sample(df, sample_id)
    high_abundance = qc.detect_high_abundance_clusters(df)
    randomness = qc.check_randomness(df)
    all_results.append({
        "sample_id": sample_id,
        "distribution_stats": stats,
        "high_abundance_clusters": high_abundance[:10],
        "randomness_test": randomness,
    })

# --- Filter outliers ---
filtered = {}
for result in all_results:
    sid = result["sample_id"]
    df = samples_data[sid].copy()
    outlier_ids = result["distribution_stats"].get("outlier_clusters", [])
    if outlier_ids:
        df = df[~df['co_barcode_cluster_id'].isin(outlier_ids)]
    filtered[sid] = df

# --- Save outputs ---
output_dir = Path(snakemake.output.qc_summary).parent
output_dir.mkdir(parents=True, exist_ok=True)

with open(snakemake.output.qc_results, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

summary_df = qc.analyze_longitudinal(samples_data)
summary_df.to_csv(snakemake.output.qc_summary, index=False)

with open(snakemake.output.filtered_data, 'wb') as f:
    pickle.dump(filtered, f)

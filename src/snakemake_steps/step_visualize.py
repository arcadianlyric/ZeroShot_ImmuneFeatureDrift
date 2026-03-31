"""Snakemake step: Drift analysis + visualization."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion_viz import DriftAnalyzer
from deconvolution_prep import CibersortPrep

embeddings = pd.read_csv(snakemake.input.embeddings, index_col=0)
features_df = pd.read_csv(snakemake.input.features)

output_dir = Path(snakemake.output.drift_metrics).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Parse CIBERSORTx results
cibersort_prep = CibersortPrep()
proportions = cibersort_prep.parse_results(snakemake.input.proportions)

analyzer = DriftAnalyzer(output_dir=str(output_dir))

drift_metrics = analyzer.calculate_drift_metrics(embeddings)
analyzer.plot_embedding_trajectory(embeddings)
analyzer.plot_cell_proportions(proportions)
analyzer.plot_splicing_fingerprint(features_df)

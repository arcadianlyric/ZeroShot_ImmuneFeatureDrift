"""Snakemake step: Evaluation / Benchmarking."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import run_full_evaluation

features_df = pd.read_csv(snakemake.input.features)
embeddings = pd.read_csv(snakemake.input.embeddings, index_col=0)

output_dir = Path(snakemake.output.report).parent
bootstrap_n = snakemake.params.get("bootstrap_n", 100)

run_full_evaluation(
    features_df=features_df,
    scgpt_embeddings=embeddings,
    output_dir=str(output_dir),
    bootstrap_n=bootstrap_n,
)

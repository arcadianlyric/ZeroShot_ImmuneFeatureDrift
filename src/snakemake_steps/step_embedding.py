"""Snakemake step: scGPT zero-shot embedding extraction."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scgpt_embedding import ZeroShotScGPTExtractor

features_df = pd.read_csv(snakemake.input.features)

scgpt = ZeroShotScGPTExtractor(
    model_dir=snakemake.params.get("model_dir", "models/scgpt_human_blood"),
)
max_genes = snakemake.params.get("max_genes", 2000)
embeddings = scgpt.get_embeddings(features_df, max_genes=max_genes)

Path(snakemake.output.embeddings).parent.mkdir(parents=True, exist_ok=True)
embeddings.to_csv(snakemake.output.embeddings)

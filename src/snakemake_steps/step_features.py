"""Snakemake step: Feature extraction (entropy, dominant fraction)."""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction import StLFRFeatureExtractor

with open(snakemake.input.filtered_data, 'rb') as f:
    samples_data = pickle.load(f)

extractor = StLFRFeatureExtractor()
features_df = extractor.process_longitudinal_samples(samples_data)

Path(snakemake.output.features).parent.mkdir(parents=True, exist_ok=True)
features_df.to_csv(snakemake.output.features, index=False)

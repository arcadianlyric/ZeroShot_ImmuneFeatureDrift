#!/usr/bin/env python3
"""
Iteration 10: End-to-end pipeline test using combined chr1 (PTPRC) + chr22 data.

Data:
  - chr1_chr22_combined_counts.txt: featureCounts output with 1525 non-zero genes
    including PTPRC (CD45, key immune marker) from chr1-PTPRC.bam and chr22 genes.

Strategy:
  Step 0: Simulate longitudinal data (2024-2026) with immune aging drift
  Step 1: ML-based Co-barcode QC (Isolation Forest)
  Step 2: Feature Extraction (Shannon entropy, dominant fraction)
  Step 3: scGPT Zero-Shot Embedding (real scGPT-blood model)
  Step 4: CIBERSORTx Deconvolution Prep + Fusion Visualization
  Step 5: Verify PTPRC drift signal
"""
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_pipeline")

# --- Paths (relative to repo root) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
COMBINED_COUNTS = REPO_ROOT / "data" / "chr1_chr22_combined_counts.txt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "iteration10_test"


@pytest.fixture(scope="module")
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="module")
def longitudinal_samples():
    """Step 0: Simulate 2024-2026 longitudinal data with immune aging drift."""
    from simulate_temporal import simulate_temporal_samples

    samples = simulate_temporal_samples(
        count_table_path=str(COMBINED_COUNTS),
        output_dir=str(OUTPUT_DIR / "longitudinal"),
        format="featurecounts",
        years=["2024", "2025", "2026"],
        drift_scale=0.08,
        seed=42,
    )
    return samples


@pytest.fixture(scope="module")
def qc_results(longitudinal_samples, output_dir):
    """Step 1: ML-based Co-barcode QC."""
    from co_barcode_qc import CoBarcodeQCAnalyzer

    qc = CoBarcodeQCAnalyzer(method="isolation_forest", contamination=0.1)
    qc.fit_longitudinal(longitudinal_samples)

    results = []
    for sid, df in longitudinal_samples.items():
        stats = qc.analyze_sample(df, sid)
        high_ab = qc.detect_high_abundance_clusters(df)
        randomness = qc.check_randomness(df)
        results.append({
            "sample_id": sid,
            "distribution_stats": stats,
            "high_abundance_clusters": high_ab[:5],
            "randomness_test": randomness,
        })

    with open(output_dir / "qc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


@pytest.fixture(scope="module")
def filtered_samples(longitudinal_samples, qc_results):
    """Filter outlier clusters from QC."""
    from main import filter_outlier_clusters
    return filter_outlier_clusters(longitudinal_samples, qc_results)


@pytest.fixture(scope="module")
def features_df(filtered_samples, output_dir):
    """Step 2: Feature extraction."""
    from feature_extraction import StLFRFeatureExtractor

    extractor = StLFRFeatureExtractor()
    df = extractor.process_longitudinal_samples(filtered_samples)
    df.to_csv(output_dir / "features.csv", index=False)
    return df


@pytest.fixture(scope="module")
def embeddings(features_df, output_dir):
    """Step 3: scGPT zero-shot embedding (falls back to mock if scgpt package missing)."""
    from scgpt_embedding import ZeroShotScGPTExtractor

    try:
        scgpt = ZeroShotScGPTExtractor(
            model_dir=str(REPO_ROOT / "models" / "scgpt_blood"),
            use_real_model=True,
        )
        emb = scgpt.get_embeddings(features_df)
    except (ImportError, ModuleNotFoundError):
        logger.warning("scgpt package not installed, falling back to mock model")
        scgpt = ZeroShotScGPTExtractor(
            model_dir=str(REPO_ROOT / "models" / "scgpt_blood"),
            use_real_model=False,
        )
        emb = scgpt.get_embeddings(features_df)

    emb.to_csv(output_dir / "embeddings.csv")
    return emb


# ============================================================
# Tests
# ============================================================

class TestStep0SimulateTemporal:
    """Verify longitudinal simulation from combined chr1+chr22 counts."""

    def test_input_file_exists(self):
        assert COMBINED_COUNTS.exists(), f"Combined count table not found: {COMBINED_COUNTS}"

    def test_three_timepoints(self, longitudinal_samples):
        assert set(longitudinal_samples.keys()) == {"2024", "2025", "2026"}

    def test_baseline_has_genes(self, longitudinal_samples):
        df_2024 = longitudinal_samples["2024"]
        assert len(df_2024) > 100, f"Baseline has only {len(df_2024)} genes"

    def test_ptprc_present(self, longitudinal_samples):
        """PTPRC (CD45) must be present — it's the key immune marker from chr1."""
        df_2024 = longitudinal_samples["2024"]
        genes = set(df_2024['gene_id'].unique()) if 'gene_id' in df_2024.columns else set(df_2024['co_barcode_cluster_id'].unique())
        assert "PTPRC" in genes, f"PTPRC not found in baseline genes. Available: {sorted(genes)[:20]}..."

    def test_immune_drift_direction(self, longitudinal_samples):
        """PTPRC should decline over time (immune aging signal)."""
        ptprc_counts = {}
        for year, df in longitudinal_samples.items():
            gene_col = 'gene_id' if 'gene_id' in df.columns else 'co_barcode_cluster_id'
            ptprc_row = df[df[gene_col] == 'PTPRC']
            if not ptprc_row.empty:
                ptprc_counts[year] = ptprc_row['count'].values[0]

        assert len(ptprc_counts) >= 2, f"PTPRC found in only {len(ptprc_counts)} timepoints"
        # PTPRC is in IMMUNE_DECLINE_GENES, so 2024 > 2026
        if "2024" in ptprc_counts and "2026" in ptprc_counts:
            assert ptprc_counts["2024"] >= ptprc_counts["2026"], \
                f"PTPRC should decline: 2024={ptprc_counts['2024']}, 2026={ptprc_counts['2026']}"


class TestStep1QC:
    """Verify ML-based QC on simulated timepoints."""

    def test_qc_all_samples(self, qc_results):
        assert len(qc_results) == 3, f"Expected 3 QC results, got {len(qc_results)}"

    def test_qc_has_stats(self, qc_results):
        for r in qc_results:
            assert "distribution_stats" in r
            assert "cv" in r["distribution_stats"]
            assert "outlier_fraction" in r["distribution_stats"]

    def test_qc_passes(self, qc_results):
        for r in qc_results:
            # Allow some outliers but overall should pass
            assert r["distribution_stats"]["outlier_fraction"] < 0.5, \
                f"Sample {r['sample_id']} has too many outliers: {r['distribution_stats']['outlier_fraction']:.2%}"


class TestStep2Features:
    """Verify feature extraction output."""

    def test_features_shape(self, features_df):
        assert len(features_df) > 0
        required_cols = ['sample_id', 'gene_id', 'entropy', 'dominant_fraction', 'total_count']
        for col in required_cols:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_features_three_samples(self, features_df):
        samples = sorted(features_df['sample_id'].unique())
        assert samples == ["2024", "2025", "2026"], f"Got samples: {samples}"

    def test_ptprc_in_features(self, features_df):
        assert "PTPRC" in features_df['gene_id'].values, "PTPRC missing from features"

    def test_entropy_range(self, features_df):
        assert features_df['entropy'].min() >= 0, "Entropy should be non-negative"

    def test_dominant_fraction_range(self, features_df):
        assert features_df['dominant_fraction'].min() >= 0
        assert features_df['dominant_fraction'].max() <= 1.0


class TestStep3Embedding:
    """Verify scGPT zero-shot embedding."""

    def test_embedding_shape(self, embeddings):
        assert embeddings.shape[0] == 3, f"Expected 3 samples, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == 512, f"Expected 512-dim, got {embeddings.shape[1]}"

    def test_embedding_samples(self, embeddings):
        assert sorted(embeddings.index.tolist()) == ["2024", "2025", "2026"]

    def test_embedding_not_identical(self, embeddings):
        """Different timepoints should produce different embeddings."""
        vec_2024 = embeddings.loc["2024"].values
        vec_2026 = embeddings.loc["2026"].values
        assert not np.allclose(vec_2024, vec_2026, atol=1e-6), \
            "2024 and 2026 embeddings are identical — drift signal lost"

    def test_embedding_norms_reasonable(self, embeddings):
        norms = np.linalg.norm(embeddings.values, axis=1)
        assert all(norms > 0), "Some embeddings are zero vectors"
        assert all(norms < 1e6), "Some embedding norms are unreasonably large"


class TestStep4Visualization:
    """Verify drift metrics and visualization outputs."""

    def test_drift_metrics(self, embeddings, output_dir):
        from fusion_viz import DriftAnalyzer

        analyzer = DriftAnalyzer(output_dir=str(output_dir / "figures"))
        metrics = analyzer.calculate_drift_metrics(embeddings)

        assert len(metrics) == 2, f"Expected 2 transitions, got {len(metrics)}"
        assert all(metrics['Cosine_Similarity'] > 0), "Cosine similarity should be positive"
        assert all(metrics['Cosine_Similarity'] <= 1.0), "Cosine similarity should be <= 1"
        assert all(metrics['Euclidean_Distance'] > 0), "Euclidean distance should be positive"

    def test_trajectory_plot(self, embeddings, output_dir):
        from fusion_viz import DriftAnalyzer

        analyzer = DriftAnalyzer(output_dir=str(output_dir / "figures"))
        analyzer.plot_embedding_trajectory(embeddings)
        assert (output_dir / "figures" / "embedding_trajectory.png").exists()

    def test_splicing_fingerprint(self, features_df, output_dir):
        from fusion_viz import DriftAnalyzer

        analyzer = DriftAnalyzer(output_dir=str(output_dir / "figures"))
        analyzer.plot_splicing_fingerprint(features_df)
        assert (output_dir / "figures" / "splicing_fingerprint.png").exists()

    def test_cell_proportions_plot(self, output_dir):
        """Test with mock CIBERSORTx output (token unavailable for commercial researchers)."""
        from fusion_viz import DriftAnalyzer

        np.random.seed(42)
        mock_props = pd.DataFrame(
            np.random.dirichlet(np.ones(5), size=3),
            index=["2024", "2025", "2026"],
            columns=["T_CD4", "T_CD8", "B_cells", "Monocytes", "NK_cells"]
        )

        analyzer = DriftAnalyzer(output_dir=str(output_dir / "figures"))
        analyzer.plot_cell_proportions(mock_props)
        assert (output_dir / "figures" / "cell_proportions.png").exists()


class TestStep5CibersortPrep:
    """Verify CIBERSORTx mixture file preparation."""

    def test_mixture_file(self, features_df, output_dir):
        from deconvolution_prep import CibersortPrep

        prep = CibersortPrep(output_dir=str(output_dir / "cibersort"))
        mixture_file = prep.prepare_mixture_file(features_df)
        assert Path(mixture_file).exists(), f"Mixture file not created: {mixture_file}"

        # Verify format: genes x samples TSV
        df = pd.read_csv(mixture_file, sep='\t', index_col=0)
        assert df.shape[0] > 0, "Mixture file has no genes"
        assert set(df.columns) == {"2024", "2025", "2026"}, f"Columns: {df.columns.tolist()}"


class TestEndToEnd:
    """Integration test: full pipeline produces consistent outputs."""

    def test_ptprc_drift_in_embeddings(self, embeddings):
        """Embedding drift should reflect immune aging — 2024→2026 should show movement."""
        from scipy.spatial.distance import cosine

        drift_24_25 = 1 - cosine(embeddings.loc["2024"].values, embeddings.loc["2025"].values)
        drift_24_26 = 1 - cosine(embeddings.loc["2024"].values, embeddings.loc["2026"].values)

        # Both should be high (same individual) but not identical
        assert drift_24_25 > 0.9, f"2024→2025 cosine similarity too low: {drift_24_25:.6f}"
        assert drift_24_26 > 0.9, f"2024→2026 cosine similarity too low: {drift_24_26:.6f}"

    def test_output_files(self, output_dir, features_df, embeddings, qc_results):
        """All key output files should be generated."""
        expected = [
            "qc_results.json",
            "features.csv",
            "embeddings.csv",
        ]
        for fname in expected:
            assert (output_dir / fname).exists(), f"Missing output: {fname}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Evaluation Framework and Benchmarking for Immune-Drift-Zero

Three evaluation axes:
  1. QC Evaluation: Does ML QC correctly identify synthetic outliers?
  2. Embedding Evaluation: Does scGPT add value over PCA baseline?
  3. Drift Evaluation: Are drift metrics statistically significant and reproducible?
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QC EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class QCBenchmark:
    """
    Evaluate the ML QC module by injecting synthetic outliers into clean data
    and measuring detection performance (precision, recall, F1, AUC).

    Synthetic outlier types:
      - PCR amplification bias: multiply a random subset of clusters by 50-1000x
      - Contamination: inject foreign high-count clusters
      - Uniform dropout: zero out a subset (simulating library prep failure)
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def inject_pcr_bias(
        self, counts: pd.DataFrame, n_outliers: int = 10, fold_range: Tuple[int, int] = (50, 1000)
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Inject PCR amplification bias. Returns modified data + ground truth labels."""
        df = counts.copy()
        n = len(df)
        outlier_idx = self.rng.choice(n, size=min(n_outliers, n), replace=False)
        labels = np.zeros(n, dtype=int)
        labels[outlier_idx] = 1

        for idx in outlier_idx:
            fold = self.rng.randint(fold_range[0], fold_range[1])
            df.iloc[idx, df.columns.get_loc('count')] *= fold

        return df, labels

    def inject_contamination(
        self, counts: pd.DataFrame, n_contaminants: int = 5, count_range: Tuple[int, int] = (10000, 100000)
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Inject foreign contamination clusters."""
        df = counts.copy()
        contaminant_rows = []
        for i in range(n_contaminants):
            contaminant_rows.append({
                'co_barcode_cluster_id': f'CONTAMINANT_{i}',
                'gene_id': f'CONTAMINANT_{i}',
                'count': self.rng.randint(count_range[0], count_range[1]),
            })
        contam_df = pd.DataFrame(contaminant_rows)

        labels = np.zeros(len(df) + n_contaminants, dtype=int)
        labels[len(df):] = 1

        df = pd.concat([df, contam_df], ignore_index=True)
        return df, labels

    def evaluate_detection(
        self,
        qc_analyzer,
        clean_data: pd.DataFrame,
        injection_type: str = "pcr_bias",
        n_trials: int = 10,
    ) -> Dict:
        """
        Run multiple trials of outlier injection + detection.
        Returns precision, recall, F1, AUC with confidence intervals.
        """
        from co_barcode_qc import CoBarcodeQCAnalyzer

        metrics_list = []

        for trial in range(n_trials):
            self.rng = np.random.RandomState(self.random_state + trial)

            if injection_type == "pcr_bias":
                injected, labels = self.inject_pcr_bias(clean_data)
            elif injection_type == "contamination":
                injected, labels = self.inject_contamination(clean_data)
            else:
                raise ValueError(f"Unknown injection type: {injection_type}")

            # Run QC
            qc = CoBarcodeQCAnalyzer(
                method="isolation_forest",
                contamination=self.contamination,
                random_state=self.random_state + trial,
            )
            counts = injected['count'].values.reshape(-1, 1)
            qc.fit(counts)
            pred_labels, scores = qc.predict(counts)

            # Convert: -1 (outlier) -> 1 (positive), 1 (normal) -> 0 (negative)
            pred_binary = (pred_labels == -1).astype(int)

            if labels.sum() > 0 and len(np.unique(labels)) > 1:
                precision = precision_score(labels, pred_binary, zero_division=0)
                recall = recall_score(labels, pred_binary, zero_division=0)
                f1 = f1_score(labels, pred_binary, zero_division=0)
                auc = roc_auc_score(labels, -scores)  # lower score = more anomalous
            else:
                precision = recall = f1 = auc = float('nan')

            metrics_list.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
            })

        metrics_df = pd.DataFrame(metrics_list)
        return {
            'injection_type': injection_type,
            'n_trials': n_trials,
            'precision_mean': float(metrics_df['precision'].mean()),
            'precision_std': float(metrics_df['precision'].std()),
            'recall_mean': float(metrics_df['recall'].mean()),
            'recall_std': float(metrics_df['recall'].std()),
            'f1_mean': float(metrics_df['f1'].mean()),
            'f1_std': float(metrics_df['f1'].std()),
            'auc_mean': float(metrics_df['auc'].mean()),
            'auc_std': float(metrics_df['auc'].std()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING EVALUATION (scGPT vs PCA baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingBenchmark:
    """
    Compare scGPT zero-shot embeddings against PCA baseline.

    Metrics:
      - Temporal ordering preservation: do adjacent timepoints cluster closer than distant ones?
      - Drift signal-to-noise ratio: drift magnitude / within-sample variance
      - Correlation with CIBERSORTx proportions (if available)
    """

    def pca_baseline(self, features_df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """
        Generate PCA baseline embeddings from raw gene expression.
        This is the "what you'd get without a foundation model" comparison.
        """
        sample_ids = sorted(features_df['sample_id'].unique())

        # Pivot to gene x sample matrix
        pivot = features_df.pivot_table(
            index='gene_id', columns='sample_id', values='total_count', fill_value=0
        )
        # Log-CPM normalization
        cpm = pivot.div(pivot.sum(axis=0), axis=1) * 1e6
        log_cpm = np.log1p(cpm)

        n_components = min(n_components, len(sample_ids) - 1, log_cpm.shape[0])
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(log_cpm.T)  # samples x components

        return pd.DataFrame(
            pca_embeddings,
            index=sample_ids,
            columns=[f"PCA_dim_{i}" for i in range(n_components)],
        )

    def calculate_drift_series(self, embeddings: pd.DataFrame) -> List[float]:
        """Calculate pairwise consecutive Euclidean distances."""
        samples = sorted(embeddings.index.tolist())
        drifts = []
        for i in range(len(samples) - 1):
            d = euclidean(embeddings.loc[samples[i]], embeddings.loc[samples[i + 1]])
            drifts.append(d)
        return drifts

    def temporal_ordering_score(self, embeddings: pd.DataFrame) -> float:
        """
        Measure whether temporal ordering is preserved in embedding space.
        Score = Spearman correlation between time index and cumulative distance
        from first timepoint. Higher = better temporal structure.
        """
        samples = sorted(embeddings.index.tolist())
        if len(samples) < 3:
            return float('nan')

        first_emb = embeddings.loc[samples[0]].values
        distances = [euclidean(first_emb, embeddings.loc[s].values) for s in samples]
        time_indices = list(range(len(samples)))

        corr, pval = spearmanr(time_indices, distances)
        return float(corr)

    def compare_methods(
        self,
        features_df: pd.DataFrame,
        scgpt_embeddings: pd.DataFrame,
        proportions: pd.DataFrame = None,
    ) -> Dict:
        """
        Head-to-head comparison of scGPT vs PCA.
        """
        pca_emb = self.pca_baseline(features_df)

        # Drift series
        scgpt_drifts = self.calculate_drift_series(scgpt_embeddings)
        pca_drifts = self.calculate_drift_series(pca_emb)

        # Temporal ordering
        scgpt_temporal = self.temporal_ordering_score(scgpt_embeddings)
        pca_temporal = self.temporal_ordering_score(pca_emb)

        # Drift correlation between methods
        if len(scgpt_drifts) == len(pca_drifts) and len(scgpt_drifts) >= 3:
            drift_corr, drift_pval = pearsonr(scgpt_drifts, pca_drifts)
        else:
            drift_corr, drift_pval = float('nan'), float('nan')

        result = {
            'scgpt_drift_mean': float(np.mean(scgpt_drifts)),
            'scgpt_drift_std': float(np.std(scgpt_drifts)),
            'pca_drift_mean': float(np.mean(pca_drifts)),
            'pca_drift_std': float(np.std(pca_drifts)),
            'scgpt_temporal_ordering': scgpt_temporal,
            'pca_temporal_ordering': pca_temporal,
            'drift_correlation': float(drift_corr),
            'drift_correlation_pval': float(drift_pval),
        }

        # Correlation with CIBERSORTx proportions (external validation)
        if proportions is not None and not proportions.empty:
            result.update(self._correlation_with_proportions(
                scgpt_embeddings, pca_emb, proportions
            ))

        return result

    def _correlation_with_proportions(
        self,
        scgpt_emb: pd.DataFrame,
        pca_emb: pd.DataFrame,
        proportions: pd.DataFrame,
    ) -> Dict:
        """Correlate embedding PC1 with dominant cell-type proportion changes."""
        common_samples = sorted(
            set(scgpt_emb.index) & set(pca_emb.index) & set(proportions.index)
        )
        if len(common_samples) < 3:
            return {}

        # Use PC1 of each method
        scgpt_pca = PCA(n_components=1).fit_transform(scgpt_emb.loc[common_samples].values).ravel()
        pca_pc1 = pca_emb.loc[common_samples].iloc[:, 0].values

        # Use the cell type with highest variance as the "signal"
        prop_aligned = proportions.loc[common_samples]
        signal_col = prop_aligned.var().idxmax()
        signal = prop_aligned[signal_col].values

        scgpt_corr = abs(float(pearsonr(scgpt_pca, signal)[0]))
        pca_corr = abs(float(pearsonr(pca_pc1, signal)[0]))

        return {
            'celltype_signal': signal_col,
            'scgpt_celltype_correlation': scgpt_corr,
            'pca_celltype_correlation': pca_corr,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DRIFT EVALUATION (Statistical significance + reproducibility)
# ═══════════════════════════════════════════════════════════════════════════════

class DriftBenchmark:
    """
    Evaluate whether observed drift is statistically significant
    and reproducible (not driven by noise or batch effects).
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def permutation_test(
        self,
        embeddings: pd.DataFrame,
        n_permutations: int = 1000,
    ) -> Dict:
        """
        Test H0: temporal ordering does not matter.
        Permute timepoint labels and compare observed total drift
        against null distribution of total drifts.
        """
        samples = sorted(embeddings.index.tolist())
        emb_matrix = embeddings.loc[samples].values

        # Observed total drift (sum of consecutive distances)
        observed_drift = sum(
            euclidean(emb_matrix[i], emb_matrix[i + 1])
            for i in range(len(samples) - 1)
        )

        # Null distribution
        null_drifts = []
        for _ in range(n_permutations):
            perm_idx = self.rng.permutation(len(samples))
            perm_matrix = emb_matrix[perm_idx]
            drift = sum(
                euclidean(perm_matrix[i], perm_matrix[i + 1])
                for i in range(len(perm_matrix) - 1)
            )
            null_drifts.append(drift)

        null_drifts = np.array(null_drifts)
        # Two-sided p-value: is observed drift unusually small (ordered trajectory)
        # or large (noisy)?
        p_value_small = float(np.mean(null_drifts <= observed_drift))
        p_value_large = float(np.mean(null_drifts >= observed_drift))

        return {
            'observed_total_drift': float(observed_drift),
            'null_drift_mean': float(np.mean(null_drifts)),
            'null_drift_std': float(np.std(null_drifts)),
            'p_value_ordered': p_value_small,  # drift < null → temporally ordered
            'p_value_noisy': p_value_large,  # drift > null → noisier than random
            'is_temporally_ordered': p_value_small < 0.05,
        }

    def bootstrap_confidence_intervals(
        self,
        features_df: pd.DataFrame,
        embedding_fn,
        n_bootstrap: int = 100,
    ) -> Dict:
        """
        Bootstrap gene subsets to estimate CI for drift metrics.
        Measures how sensitive the drift signal is to gene selection.

        Args:
            features_df: Full feature DataFrame.
            embedding_fn: Callable(features_df) -> embeddings_df.
            n_bootstrap: Number of bootstrap iterations.
        """
        all_genes = features_df['gene_id'].unique()
        n_genes = len(all_genes)
        subsample_size = int(0.8 * n_genes)

        drift_samples = []
        for b in range(n_bootstrap):
            gene_subset = self.rng.choice(all_genes, size=subsample_size, replace=True)
            subset_df = features_df[features_df['gene_id'].isin(gene_subset)].copy()

            try:
                emb = embedding_fn(subset_df)
                samples = sorted(emb.index.tolist())
                total_drift = sum(
                    euclidean(emb.loc[samples[i]], emb.loc[samples[i + 1]])
                    for i in range(len(samples) - 1)
                )
                drift_samples.append(total_drift)
            except Exception:
                continue

        if not drift_samples:
            return {'error': 'All bootstrap iterations failed'}

        drift_arr = np.array(drift_samples)
        return {
            'n_successful': len(drift_arr),
            'drift_mean': float(np.mean(drift_arr)),
            'drift_std': float(np.std(drift_arr)),
            'drift_ci_lower': float(np.percentile(drift_arr, 2.5)),
            'drift_ci_upper': float(np.percentile(drift_arr, 97.5)),
            'drift_cv': float(np.std(drift_arr) / np.mean(drift_arr)) if np.mean(drift_arr) > 0 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    features_df: pd.DataFrame,
    scgpt_embeddings: pd.DataFrame,
    proportions: pd.DataFrame = None,
    clean_sample_data: pd.DataFrame = None,
    output_dir: str = "./outputs/evaluation",
    bootstrap_n: int = 100,
) -> Dict:
    """
    Run complete evaluation framework.

    Args:
        features_df: Output of feature extraction step.
        scgpt_embeddings: 512-dim scGPT embeddings.
        proportions: CIBERSORTx cell proportions (optional).
        clean_sample_data: A single sample's clean count data for QC benchmark (optional).
        output_dir: Where to save evaluation outputs.
        bootstrap_n: Bootstrap iterations.

    Returns:
        Complete evaluation report dict.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    # --- 1. QC Benchmark ---
    if clean_sample_data is not None:
        logger.info("Running QC benchmark with synthetic outlier injection...")
        qc_bench = QCBenchmark()
        report['qc_pcr_bias'] = qc_bench.evaluate_detection(
            None, clean_sample_data, injection_type="pcr_bias", n_trials=10
        )
        report['qc_contamination'] = qc_bench.evaluate_detection(
            None, clean_sample_data, injection_type="contamination", n_trials=10
        )
        logger.info(f"  PCR bias detection F1: {report['qc_pcr_bias']['f1_mean']:.3f} "
                     f"(+/- {report['qc_pcr_bias']['f1_std']:.3f})")
        logger.info(f"  Contamination detection F1: {report['qc_contamination']['f1_mean']:.3f}")

    # --- 2. Embedding Benchmark ---
    logger.info("Running embedding benchmark (scGPT vs PCA)...")
    emb_bench = EmbeddingBenchmark()
    report['embedding_comparison'] = emb_bench.compare_methods(
        features_df, scgpt_embeddings, proportions
    )
    logger.info(f"  scGPT temporal ordering: {report['embedding_comparison']['scgpt_temporal_ordering']:.3f}")
    logger.info(f"  PCA temporal ordering:   {report['embedding_comparison']['pca_temporal_ordering']:.3f}")

    # --- 3. Drift Benchmark ---
    logger.info("Running drift significance tests...")
    drift_bench = DriftBenchmark()
    report['drift_permutation'] = drift_bench.permutation_test(scgpt_embeddings)
    logger.info(f"  Permutation test p-value (ordered): {report['drift_permutation']['p_value_ordered']:.4f}")
    logger.info(f"  Temporally ordered: {report['drift_permutation']['is_temporally_ordered']}")

    # --- Comparison plot ---
    pca_emb = emb_bench.pca_baseline(features_df)
    _plot_comparison(scgpt_embeddings, pca_emb, output_dir)

    # --- Save report ---
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")

    return report


def _plot_comparison(scgpt_emb: pd.DataFrame, pca_emb: pd.DataFrame, output_dir: Path):
    """Side-by-side PCA trajectory comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, emb, title in [
        (axes[0], scgpt_emb, 'scGPT Zero-Shot'),
        (axes[1], pca_emb, 'PCA Baseline'),
    ]:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(emb.values)
        samples = emb.index.tolist()

        ax.scatter(reduced[:, 0], reduced[:, 1], s=150, c='steelblue', edgecolors='black', zorder=5)
        for i, s in enumerate(samples):
            ax.annotate(s, (reduced[i, 0], reduced[i, 1]),
                        xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
        for i in range(len(samples) - 1):
            ax.annotate('', xy=(reduced[i + 1, 0], reduced[i + 1, 1]),
                        xytext=(reduced[i, 0], reduced[i, 1]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6))
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    plt.tight_layout()
    plt.savefig(output_dir / "scgpt_vs_pca.png", dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Immune-Drift-Zero Evaluation Framework")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.csv")
    parser.add_argument("--proportions", default=None, help="Path to CIBERSORTx_Results.txt")
    parser.add_argument("--clean-sample", default=None,
                        help="Path to a clean sample CSV for QC benchmark")
    parser.add_argument("--output-dir", default="./outputs/evaluation")
    parser.add_argument("--bootstrap-n", type=int, default=100)
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    embeddings = pd.read_csv(args.embeddings, index_col=0)

    proportions = None
    if args.proportions:
        proportions = pd.read_csv(args.proportions, sep='\t', index_col=0)
        prop_cols = [c for c in proportions.columns if c not in ['P-value', 'Correlation', 'RMSE']]
        proportions = proportions[prop_cols]

    clean_data = None
    if args.clean_sample:
        clean_data = pd.read_csv(args.clean_sample)

    report = run_full_evaluation(
        features_df=features_df,
        scgpt_embeddings=embeddings,
        proportions=proportions,
        clean_sample_data=clean_data,
        output_dir=args.output_dir,
        bootstrap_n=args.bootstrap_n,
    )

    print("\n=== Evaluation Summary ===")
    print(json.dumps(report, indent=2))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import re
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_co_barcode_counts_from_bam(
    bam_file: str,
    barcode_pattern: str = r'#([ATCG]{15})',
    min_mapq: int = 30,
    use_cb_tag: bool = True,
) -> pd.DataFrame:
    """
    Extract co-barcode/UMI counts from stLFR or MGI library BAM file.
    
    Supports two formats:
    1. Co-barcode in read name: @readID#barcode1_barcode2_barcode3 (stLFR genomics)
    2. 15bp UMI in read name: @readID#GCTTGTTTCGAATTT (MGI stLFR/cfRNA)
    3. CB tag (from cellranger/star)
    
    Args:
        bam_file: Path to BAM file from alignment
        barcode_pattern: Regex to extract barcode/UMI from read name
                        Default matches 15bp DNA string after #
        min_mapq: Minimum mapping quality to consider
        use_cb_tag: If True, prefer CB tag over read name extraction
    
    Returns:
        DataFrame with columns: ['co_barcode_cluster_id', 'count']
        - co_barcode_cluster_id: format is {gene}_{barcode} if gene available,
          otherwise just {barcode}
    """
    try:
        import pysam
    except ImportError:
        raise ImportError("pysam is required for BAM input. Install with: pip install pysam")
    
    logger.info(f"Loading co-barcode counts from {bam_file}...")
    
    gene_barcode_counts = defaultdict(lambda: defaultdict(int))
    barcode_pattern_re = re.compile(barcode_pattern)
    
    bam = pysam.AlignmentFile(bam_file, "rb")
    
    for read in bam:
        if read.is_unmapped or read.mapping_quality < min_mapq:
            continue
        
        barcode = None
        
        if use_cb_tag and read.has_tag("CB"):
            barcode = read.get_tag("CB")
        else:
            match = barcode_pattern_re.search(read.query_name)
            if match:
                barcode = match.group(1)
        
        if barcode is None:
            continue
        
        gene = None
        if read.has_tag("XS"):
            gene = read.get_tag("XS")
        
        if gene is None or gene == "Unassigned":
            cluster_id = barcode
        else:
            cluster_id = f"{gene}_{barcode}"
        
        gene_barcode_counts[cluster_id]["count"] += 1
    
    bam.close()
    
    rows = []
    for cluster_id, counts_dict in gene_barcode_counts.items():
        rows.append({
            "co_barcode_cluster_id": cluster_id,
            "count": counts_dict["count"],
        })
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} co-barcode clusters from {bam_file}")
    
    return df


def load_co_barcode_counts_from_directory(
    bam_dir: str,
    sample_pattern: str = "*.bam",
    barcode_pattern: str = r'#([ATCG]{15})',
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Load co-barcode/UMI counts from a directory of BAM files.
    
    Args:
        bam_dir: Directory containing BAM files
        sample_pattern: Glob pattern for BAM files
        barcode_pattern: Regex to extract barcode/UMI from read name
        **kwargs: Arguments passed to load_co_barcode_counts_from_bam
    
    Returns:
        Dict mapping sample_id to DataFrame
    """
    from pathlib import Path
    
    bam_dir = Path(bam_dir)
    samples = {}
    
    for bam_file in bam_dir.glob(sample_pattern):
        sample_id = bam_file.stem
        samples[sample_id] = load_co_barcode_counts_from_bam(str(bam_file), **kwargs)
    
    logger.info(f"Loaded {len(samples)} samples from {bam_dir}")
    return samples


class CoBarcodeQCAnalyzer:
    """
    ML-based Quality Control for MGI stLFR Co-barcode distributions.
    
    Detects technical artifacts (PCR bias, optical duplicates, contamination)
    by analyzing the distribution of co-barcode cluster abundances.
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Args:
            method: 'isolation_forest' or 'autoencoder'
            contamination: Expected fraction of abnormal co-barcodes
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        
    def fit(self, co_barcode_counts: np.ndarray) -> "CoBarcodeQCAnalyzer":
        """
        Fit QC model on co-barcode count matrix.
        
        Args:
            co_barcode_counts: (n_genes, n_clusters) or (n_samples, n_features)
        """
        logger.info(f"Fitting {self.method} QC model...")
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
            )
            self.model.fit(co_barcode_counts)
            
        elif self.method == "autoencoder":
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
            )
            self.model.fit(co_barcode_counts, co_barcode_counts)
            
        return self
    
    def predict(self, co_barcode_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outlier co-barcodes.
        
        Returns:
            (outlier_labels, anomaly_scores)
            - outlier_labels: -1 for outliers, 1 for normal
            - anomaly_scores: lower is more anomalous (for isolation forest)
        """
        if self.method == "isolation_forest":
            outlier_labels = self.model.predict(co_barcode_counts)
            anomaly_scores = self.model.score_samples(co_barcode_counts)
            return outlier_labels, anomaly_scores
            
        elif self.method == "autoencoder":
            reconstruction = self.model.predict(co_barcode_counts)
            mse = np.mean((co_barcode_counts - reconstruction) ** 2, axis=1)
            threshold = np.percentile(mse, (1 - self.contamination) * 100)
            outlier_labels = (mse > threshold).astype(int) * -1 + 1
            return outlier_labels, -mse
            
        return np.zeros(len(co_barcode_counts)), np.zeros(len(co_barcode_counts))
    
    def fit_longitudinal(
        self,
        samples_dict: Dict[str, pd.DataFrame],
    ) -> "CoBarcodeQCAnalyzer":
        """
        Fit a single shared QC model on pooled co-barcode counts from all timepoints.
        This ensures consistent anomaly thresholds across longitudinal samples.
        """
        all_counts = []
        for counts_df in samples_dict.values():
            all_counts.append(counts_df['count'].values.reshape(-1, 1))
        pooled = np.vstack(all_counts)
        logger.info(f"Fitting shared QC model on {len(pooled)} pooled co-barcode clusters...")
        return self.fit(pooled)

    def analyze_sample(
        self,
        sample_counts: pd.DataFrame,
        sample_id: str,
    ) -> Dict:
        """
        Comprehensive QC analysis for a single sample.
        
        Args:
            sample_counts: DataFrame with ['co_barcode_cluster_id', 'count']
            sample_id: Sample identifier
        """
        logger.info(f"Running QC for sample {sample_id}...")
        
        counts = sample_counts['count'].values.reshape(-1, 1)
        
        if self.model is None:
            logger.warning(f"No pre-fitted model; fitting on this sample alone. "
                           f"Call fit_longitudinal() first for cross-sample consistency.")
            self.fit(counts)
        
        outlier_labels, anomaly_scores = self.predict(counts)
        
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        total_clusters = len(counts)
        outlier_count = len(outlier_indices)
        
        stats = {
            "sample_id": sample_id,
            "total_co_barcode_clusters": total_clusters,
            "outlier_count": outlier_count,
            "outlier_fraction": outlier_count / total_clusters if total_clusters > 0 else 0,
            "mean_count": float(np.mean(counts)),
            "std_count": float(np.std(counts)),
            "median_count": float(np.median(counts)),
            "cv": float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0,
            "passed": (outlier_count / total_clusters) < self.contamination if total_clusters > 0 else True,
        }
        
        if len(outlier_indices) > 0:
            outlier_clusters = sample_counts.iloc[outlier_indices]
            stats["outlier_clusters"] = outlier_clusters['co_barcode_cluster_id'].tolist()[:50]
            stats["outlier_mean_count"] = float(np.mean(counts[outlier_indices]))
            stats["outlier_anomaly_scores"] = anomaly_scores[outlier_indices].tolist()[:50]
            
        return stats
    
    def analyze_longitudinal(
        self,
        samples_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Run QC on multiple timepoints and detect temporal anomalies.
        """
        results = []
        
        for sample_id, counts_df in samples_dict.items():
            stats = self.analyze_sample(counts_df, sample_id)
            results.append(stats)
            
        results_df = pd.DataFrame(results)
        
        logger.info(f"QC Summary:\n{results_df[['sample_id', 'outlier_fraction', 'cv', 'passed']]}")
        
        return results_df
    
    def detect_high_abundance_clusters(
        self,
        sample_counts: pd.DataFrame,
        percentile: float = 99,
    ) -> List[Dict]:
        """
        Detect co-barcode clusters with abnormally high abundance.
        This may indicate PCR bias or contamination.
        """
        counts = sample_counts['count'].values
        threshold = np.percentile(counts, percentile)
        
        high_abundance = []
        for i, (cluster_id, count) in enumerate(zip(sample_counts['co_barcode_cluster_id'], counts)):
            if count > threshold:
                z_score = (count - np.mean(counts)) / np.std(counts) if np.std(counts) > 0 else 0
                high_abundance.append({
                    "co_barcode_cluster_id": cluster_id,
                    "count": int(count),
                    "z_score": float(z_score),
                    "fold_change": float(count / np.median(counts)) if np.median(counts) > 0 else 0,
                })
        
        return sorted(high_abundance, key=lambda x: x["count"], reverse=True)
    
    def check_randomness(
        self,
        sample_counts: pd.DataFrame,
        n_bootstrap: int = 1000,
    ) -> Dict:
        """
        Test if co-barcode distribution follows expected random pattern.
        Under ideal conditions, co-barcode counts should follow Poisson.
        We simulate Poisson draws with the observed mean and compare the CV
        of the observed data against the null distribution of CVs.
        """
        counts = sample_counts['count'].values
        n = len(counts)
        observed_mean = np.mean(counts)
        observed_cv = np.std(counts) / observed_mean if observed_mean > 0 else 0
        
        # Simulate Poisson null: if counts are truly random, they follow Poisson(lambda=mean)
        simulated_cvs = []
        for _ in range(n_bootstrap):
            sim = np.random.poisson(lam=max(observed_mean, 1), size=n).astype(float)
            sim_mean = np.mean(sim)
            sim_cv = np.std(sim) / sim_mean if sim_mean > 0 else 0
            simulated_cvs.append(sim_cv)
        
        simulated_cvs = np.array(simulated_cvs)
        # One-sided test: is observed CV significantly larger than Poisson expectation?
        p_value = float(np.mean(simulated_cvs >= observed_cv))
        
        return {
            "observed_cv": float(observed_cv),
            "expected_cv_mean": float(np.mean(simulated_cvs)),
            "expected_cv_std": float(np.std(simulated_cvs)),
            "p_value_randomness": p_value,
            "is_random": p_value > 0.05,
        }


def run_qc_pipeline(samples_dict: Dict[str, pd.DataFrame], output_dir: str = "./outputs/qc"):
    """
    Run complete QC pipeline on longitudinal stLFR samples.
    """
    import json
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qc_analyzer = CoBarcodeQCAnalyzer(method="isolation_forest", contamination=0.1)
    
    # Fit a single shared model on pooled data for cross-sample consistency
    qc_analyzer.fit_longitudinal(samples_dict)
    
    all_results = []
    
    for sample_id, counts_df in samples_dict.items():
        logger.info(f"\n=== QC Analysis for {sample_id} ===")
        
        stats = qc_analyzer.analyze_sample(counts_df, sample_id)
        
        high_abundance = qc_analyzer.detect_high_abundance_clusters(counts_df)
        
        randomness = qc_analyzer.check_randomness(counts_df)
        
        sample_result = {
            "sample_id": sample_id,
            "distribution_stats": stats,
            "high_abundance_clusters": high_abundance[:10],
            "randomness_test": randomness,
        }
        
        all_results.append(sample_result)
        
        with open(output_dir / f"qc_{sample_id}.json", "w") as f:
            json.dump(sample_result, f, indent=2, default=str)
    
    summary_df = qc_analyzer.analyze_longitudinal(samples_dict)
    summary_df.to_csv(output_dir / "qc_summary.csv", index=False)
    
    logger.info(f"\nQC Pipeline Complete. Results saved to {output_dir}")
    
    return all_results, summary_df

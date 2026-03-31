import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StLFRFeatureExtractor:
    """
    Extracts multi-modal features from MGI stLFR co-barcoded count data.
    Instead of isoforms, we use co-barcode clusters mapped to genes.
    
    Features:
    1. Gene-level counts (for standard expression embedding)
    2. Co-barcode diversity (entropy, dominant cluster fraction, num_clusters)
       - Serves as a proxy for splicing complexity/isoform diversity
    """
    
    def __init__(self, gene_mapping_file: str = None):
        """
        Args:
            gene_mapping_file: CSV with ['co_barcode_cluster_id', 'gene_id'] mapping if needed.
        """
        self.gene_mapping = pd.read_csv(gene_mapping_file) if gene_mapping_file else None
        
    def _get_gene_id(self, cluster_id: str) -> str:
        """Fallback gene ID extraction if no mapping provided."""
        if self.gene_mapping is not None:
            return self.gene_mapping.loc[self.gene_mapping['co_barcode_cluster_id'] == cluster_id, 'gene_id'].values[0]
        # Assume format like "GENE_cluster0"
        return cluster_id.split("_")[0] if "_" in cluster_id else cluster_id

    def extract_features(self, cluster_counts: pd.DataFrame, sample_id: str) -> pd.DataFrame:
        """
        Args:
            cluster_counts: DataFrame with ['co_barcode_cluster_id', 'count']
            sample_id: str, e.g., '2024', '2025', '2026'
        Returns:
            DataFrame with gene-level features: [gene_id, count, entropy, dominant_fraction, num_clusters]
        """
        logger.info(f"Extracting features for sample {sample_id}...")
        
        # Add gene_id
        df = cluster_counts.copy()
        df['gene_id'] = df['co_barcode_cluster_id'].apply(self._get_gene_id)
        
        # Group by gene
        gene_features = []
        for gene_id, group in df.groupby('gene_id'):
            total_count = group['count'].sum()
            num_clusters = len(group)
            
            if total_count == 0:
                continue
                
            # Calculate fractions
            fractions = group['count'] / total_count
            
            # 1. Shannon Entropy of co-barcode cluster distribution
            cluster_entropy = entropy(fractions, base=2) if num_clusters > 1 else 0.0
            
            # 2. Dominant Cluster Fraction
            dominant_fraction = fractions.max()
            
            gene_features.append({
                'sample_id': sample_id,
                'gene_id': gene_id,
                'total_count': total_count,
                'entropy': cluster_entropy,
                'dominant_fraction': dominant_fraction,
                'num_clusters': num_clusters
            })
            
        result_df = pd.DataFrame(gene_features)
        logger.info(f"Extracted features for {len(result_df)} genes.")
        return result_df

    def process_longitudinal_samples(self, samples_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process multiple timepoints and combine."""
        all_features = []
        for sample_id, counts_df in samples_dict.items():
            feats = self.extract_features(counts_df, sample_id)
            all_features.append(feats)
        return pd.concat(all_features, ignore_index=True)

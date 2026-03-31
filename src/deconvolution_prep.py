import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CibersortPrep:
    """
    Prepares bulk RNA expression data for CIBERSORTx deconvolution.
    Format required by CIBERSORTx: Tab-delimited text file (mixture file)
    Columns: Sample IDs
    Rows: Gene Symbols
    Values: Non-log transformed normalized counts (e.g., TPM, FPKM, CPM)
    """
    
    def __init__(self, output_dir: str = "./outputs/cibersort"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_mixture_file(
        self, 
        longitudinal_features: pd.DataFrame, 
        output_filename: str = "mixture_2024_2026.txt"
    ) -> str:
        """
        Converts extracted features into CIBERSORTx format.
        Args:
            longitudinal_features: DataFrame with ['sample_id', 'gene_id', 'total_count']
        """
        logger.info("Preparing mixture file for CIBERSORTx...")
        
        # Pivot the dataframe so that:
        # Rows = gene_id
        # Columns = sample_id
        # Values = total_count
        mixture_df = longitudinal_features.pivot(
            index='gene_id', 
            columns='sample_id', 
            values='total_count'
        ).fillna(0)
        
        # Calculate CPM (Counts Per Million) for CIBERSORTx
        # CIBERSORTx prefers non-log linear space normalized data
        cpm_df = mixture_df.div(mixture_df.sum(axis=0), axis=1) * 1e6
        
        # Formatting for CIBERSORTx
        cpm_df.index.name = "GeneSymbol"
        
        output_path = self.output_dir / output_filename
        cpm_df.to_csv(output_path, sep='\t')
        
        logger.info(f"Successfully generated CIBERSORTx mixture file at {output_path}")
        logger.info(f"Shape: {cpm_df.shape[0]} genes x {cpm_df.shape[1]} samples")
        
        return str(output_path)
    
    def parse_results(self, cibersort_results_file: str) -> pd.DataFrame:
        """
        Parses CIBERSORTx output (CIBERSORTx_Results.txt) for downstream visualization.
        """
        if not Path(cibersort_results_file).exists():
            logger.warning(f"CIBERSORTx results file not found at {cibersort_results_file}")
            return pd.DataFrame()
            
        results = pd.read_csv(cibersort_results_file, sep='\t', index_col=0)
        # Drop metric columns (P-value, Correlation, RMSE) to keep only cell proportions
        proportion_cols = [c for c in results.columns if c not in ['P-value', 'Correlation', 'RMSE']]
        proportions = results[proportion_cols]
        
        return proportions

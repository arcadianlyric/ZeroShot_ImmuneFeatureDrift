import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftAnalyzer:
    """
    Fuses multiple modalities (scGPT embeddings, cell proportions, splicing fingerprints)
    to quantify and visualize intra-individual immune temporal drift.
    """
    
    def __init__(self, output_dir: str = "./outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        sns.set_theme(style="whitegrid")
        
    def calculate_drift_metrics(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Cosine Similarity and Euclidean Distance between consecutive timepoints.
        """
        logger.info("Calculating embedding drift metrics...")
        
        samples = sorted(embeddings.index.tolist())
        metrics = []
        
        for i in range(len(samples) - 1):
            t1 = samples[i]
            t2 = samples[i+1]
            
            vec1 = embeddings.loc[t1].values
            vec2 = embeddings.loc[t2].values
            
            cos_sim = 1 - cosine(vec1, vec2)
            euc_dist = euclidean(vec1, vec2)
            
            metrics.append({
                'Timepoint_Transition': f"{t1} -> {t2}",
                'Cosine_Similarity': cos_sim,
                'Euclidean_Distance': euc_dist
            })
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(self.output_dir / "drift_metrics.csv", index=False)
        return metrics_df

    def plot_embedding_trajectory(self, embeddings: pd.DataFrame):
        """
        Reduce 512-dim scGPT embeddings to 2D (PCA/UMAP) and plot trajectory.
        Since N=3, PCA is more stable and interpretable than UMAP.
        """
        logger.info("Plotting embedding trajectory...")
        
        # Use PCA for N=3
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings.values)
        
        df_plot = pd.DataFrame(
            reduced, 
            index=embeddings.index, 
            columns=['PC1', 'PC2']
        )
        
        plt.figure(figsize=(8, 6))
        
        # Plot points
        sns.scatterplot(
            data=df_plot, x='PC1', y='PC2', 
            s=200, color='b', marker='o', edgecolor='black'
        )
        
        # Add labels
        for i, sample in enumerate(df_plot.index):
            plt.annotate(
                sample, 
                (df_plot.iloc[i]['PC1'], df_plot.iloc[i]['PC2']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold'
            )
            
        # Draw trajectory arrows (scale arrow head to data range)
        x_range = df_plot['PC1'].max() - df_plot['PC1'].min()
        y_range = df_plot['PC2'].max() - df_plot['PC2'].min()
        data_scale = max(x_range, y_range, 1e-6)
        hw = data_scale * 0.02   # head width = 2% of data range
        hl = data_scale * 0.03   # head length = 3% of data range
        lw = data_scale * 0.004  # line width = 0.4% of data range

        for i in range(len(df_plot) - 1):
            x_start = df_plot.iloc[i]['PC1']
            y_start = df_plot.iloc[i]['PC2']
            x_end = df_plot.iloc[i+1]['PC1']
            y_end = df_plot.iloc[i+1]['PC2']

            plt.annotate(
                '', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(
                    arrowstyle='->', color='red', lw=1.5,
                    mutation_scale=12, alpha=0.7,
                ),
            )
            
        plt.title('Individual Immune Drift Trajectory (scGPT Zero-shot)', fontsize=14)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "embedding_trajectory.png", dpi=300)
        plt.close()

    def plot_cell_proportions(self, proportions: pd.DataFrame):
        """
        Plot stacked bar chart of CIBERSORTx cell proportions over time.
        """
        if proportions.empty:
            logger.warning("No cell proportions provided, skipping plot.")
            return
            
        logger.info("Plotting cell proportion drift...")
        
        # Keep top 10 most abundant cell types for clarity
        mean_props = proportions.mean().sort_values(ascending=False)
        top_cells = mean_props.head(10).index
        
        plot_df = proportions[top_cells].copy()
        plot_df['Other'] = 1.0 - plot_df.sum(axis=1)
        
        # Plot stacked bar chart
        ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
        
        plt.title('Immune Cell Proportion Drift (CIBERSORTx)', fontsize=14)
        plt.xlabel('Timepoint', fontsize=12)
        plt.ylabel('Fraction', fontsize=12)
        plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "cell_proportions.png", dpi=300)
        plt.close()

    def plot_splicing_fingerprint(self, features_df: pd.DataFrame, marker_genes: list = None):
        """
        Plot heatmap of isoform diversity (e.g. dominant fraction) for key immune genes.
        """
        logger.info("Plotting splicing fingerprint heatmap...")
        
        if marker_genes is None:
            # Default immune marker genes (PTPRC = CD45, standard HGNC symbols)
            marker_genes = ['PTPRC', 'CD3D', 'CD3E', 'CD8A', 'CD4',
                            'FOXP3', 'IL7R', 'NCAM1', 'NEK7', 'PTPRC-AS1']

        # Filter for marker genes present in the data
        available = set(features_df['gene_id'].unique())
        matched_markers = [g for g in marker_genes if g in available]
        if not matched_markers:
            # Fallback: pick top variable genes
            var_by_gene = features_df.groupby('gene_id')['dominant_fraction'].std()
            matched_markers = var_by_gene.nlargest(10).index.tolist()
            logger.info(f"No marker genes found; using top 10 variable genes instead")

        df_filtered = features_df[features_df['gene_id'].isin(matched_markers)]
        
        if df_filtered.empty:
            logger.warning("No marker genes found in features, skipping splicing plot.")
            return
            
        # Create pivot table for dominant isoform fraction
        pivot_df = df_filtered.pivot(index='gene_id', columns='sample_id', values='dominant_fraction')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_df, 
            cmap='YlOrRd', 
            annot=True, 
            fmt=".2f",
            linewidths=.5,
            cbar_kws={'label': 'Dominant Isoform Fraction'}
        )
        
        plt.title('Immune Splicing Fingerprint (Isoform Switch)', fontsize=14)
        plt.xlabel('Timepoint')
        plt.ylabel('Marker Gene')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "splicing_fingerprint.png", dpi=300)
        plt.close()

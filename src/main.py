import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
import argparse

from stlfr_preprocess import StLFRPreprocessor, generate_mock_stlfr_count_table, load_count_table
from co_barcode_qc import CoBarcodeQCAnalyzer, run_qc_pipeline
from feature_extraction import StLFRFeatureExtractor
from deconvolution_prep import CibersortPrep
from scgpt_embedding import ZeroShotScGPTExtractor
from fusion_viz import DriftAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_outlier_clusters(
    samples_data: dict,
    qc_results: list,
) -> dict:
    """
    Remove outlier co-barcode clusters flagged by QC from each sample.
    This prevents technical artifacts from propagating into downstream
    feature engineering and embedding extraction.
    """
    filtered = {}
    for result in qc_results:
        sid = result["sample_id"]
        df = samples_data[sid].copy()
        outlier_ids = result["distribution_stats"].get("outlier_clusters", [])
        if outlier_ids:
            before = len(df)
            df = df[~df['co_barcode_cluster_id'].isin(outlier_ids)]
            logger.info(f"Sample {sid}: removed {before - len(df)} outlier clusters, "
                        f"{len(df)} remaining")
        filtered[sid] = df
    return filtered


def run_pipeline(
    fastq_manifest: str = None,
    count_table: str = None,
    count_format: str = "auto",
    sample_id_map: dict = None,
    data_dir: str = "./data/mock",
    output_dir: str = "./outputs",
):
    logger.info("Starting Immune-Drift-Zero Pipeline...")
    
    # ========== Step -1: Load Data ==========
    # Priority: --count-table > --fastq-manifest > --data-dir > mock generation
    if count_table and os.path.exists(count_table):
        logger.info(f"Loading in-house count table from {count_table}...")
        samples_data = load_count_table(
            count_table,
            format=count_format,
            sample_id_map=sample_id_map,
        )
    elif fastq_manifest and os.path.exists(fastq_manifest):
        logger.info("Running upstream stLFR preprocessing from raw FASTQ...")
        preprocessor = StLFRPreprocessor()
        manifest = pd.read_csv(fastq_manifest)  # columns: sample_id, fastq_r1, fastq_r2
        samples_data = {}
        for _, row in manifest.iterrows():
            ct = preprocessor.process(
                fastq_r1=row['fastq_r1'],
                fastq_r2=row.get('fastq_r2'),
                output_dir=f"{output_dir}/stlfr_preprocess/{row['sample_id']}",
            )
            samples_data[str(row['sample_id'])] = ct
    else:
        # Load pre-computed count tables or generate mock data
        if not os.path.exists(data_dir) or not list(Path(data_dir).glob("sample_*.csv")):
            logger.info("No existing data found. Generating mock stLFR count tables...")
            samples_data = generate_mock_stlfr_count_table(output_dir=data_dir)
        else:
            samples_data = {
                fp.stem.split("_")[1]: pd.read_csv(fp)
                for fp in sorted(Path(data_dir).glob("sample_*.csv"))
            }
    
    # ========== Step 0: ML-based Co-barcode QC ==========
    logger.info("Step 0: Running ML-based Co-barcode QC...")
    qc_results, qc_summary = run_qc_pipeline(samples_data, output_dir=f"{output_dir}/qc")
    
    all_passed = all(r["distribution_stats"]["passed"] for r in qc_results)
    if not all_passed:
        logger.warning("Some samples have high outlier fraction. Filtering outlier clusters...")
    else:
        logger.info("All samples passed QC.")
    
    # Filter outlier clusters from data before downstream analysis
    samples_data = filter_outlier_clusters(samples_data, qc_results)
    
    # ========== Step 1: Feature Extraction (stLFR Co-barcode diversity) ==========
    extractor = StLFRFeatureExtractor()
    features_df = extractor.process_longitudinal_samples(samples_data)
    
    # ========== Step 2: Path A — CIBERSORTx Deconvolution ==========
    cibersort_prep = CibersortPrep(output_dir=f"{output_dir}/cibersort")
    mixture_file = cibersort_prep.prepare_mixture_file(features_df)
    
    # Mock CIBERSORTx output (replace with real results in production)
    sample_ids = sorted(samples_data.keys())
    mock_props = pd.DataFrame(
        np.random.dirichlet(np.ones(5), size=len(sample_ids)),
        index=sample_ids,
        columns=["T_CD4", "T_CD8", "B_cells", "Monocytes", "NK_cells"]
    )
    cibersort_output = f"{output_dir}/cibersort/CIBERSORTx_Results.txt"
    mock_props.to_csv(cibersort_output, sep='\t')
    proportions = cibersort_prep.parse_results(cibersort_output)
    
    # ========== Step 3: Path B — scGPT Zero-Shot Embedding ==========
    scgpt = ZeroShotScGPTExtractor(
        model_dir="./models/scgpt_blood",
        use_real_model=True,
    )
    embeddings = scgpt.get_embeddings(features_df)
    
    # ========== Step 4: Fusion & Visualization ==========
    analyzer = DriftAnalyzer(output_dir=f"{output_dir}/figures")
    
    drift_metrics = analyzer.calculate_drift_metrics(embeddings)
    logger.info(f"Drift metrics:\n{drift_metrics}")
    
    analyzer.plot_embedding_trajectory(embeddings)
    analyzer.plot_cell_proportions(proportions)
    analyzer.plot_splicing_fingerprint(features_df)
    
    logger.info(f"Pipeline completed successfully! Results in '{output_dir}/'.")
    
    return {
        "qc_summary": qc_summary,
        "features": features_df,
        "embeddings": embeddings,
        "proportions": proportions,
        "drift_metrics": drift_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Immune-Drift-Zero: Zero-Shot Immune Trajectory Monitoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with in-house featureCounts count table (most common):
  python main.py --count-table /path/to/feature_count.txt --output-dir ./outputs/myrun

  # Run with in-house CSV count table + custom sample ID mapping:
  python main.py --count-table counts.csv --count-format csv --output-dir ./outputs

  # Run with raw stLFR FASTQ manifest:
  python main.py --fastq-manifest manifest.csv --output-dir ./outputs

  # Run with mock data (testing):
  python main.py --output-dir ./outputs/test
        """,
    )
    parser.add_argument("--count-table", default=None,
                        help="Path to in-house count table (featureCounts output or CSV/TSV "
                             "with rows=genes, columns=samples). This is the recommended "
                             "entry point for users with existing pipelines.")
    parser.add_argument("--count-format", default="auto", choices=["auto", "featurecounts", "csv"],
                        help="Format of the count table (default: auto-detect)")
    parser.add_argument("--sample-id-map", default=None,
                        help="JSON string mapping column names to sample IDs, "
                             'e.g. \'{"data/2024.bam": "2024", "data/2025.bam": "2025"}\'')
    parser.add_argument("--fastq-manifest", default=None,
                        help="CSV with columns: sample_id, fastq_r1, fastq_r2")
    parser.add_argument("--data-dir", default="./data/mock",
                        help="Directory with pre-computed sample_*.csv count tables")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Output directory (default: ./outputs)")
    args = parser.parse_args()

    # Parse sample-id-map JSON if provided
    sid_map = None
    if args.sample_id_map:
        import json
        sid_map = json.loads(args.sample_id_map)

    run_pipeline(
        count_table=args.count_table,
        count_format=args.count_format,
        sample_id_map=sid_map,
        fastq_manifest=args.fastq_manifest,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

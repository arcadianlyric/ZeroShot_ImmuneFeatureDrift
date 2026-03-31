"""
Immune-Drift-Zero Snakemake Workflow
=====================================

Usage:
  # Full pipeline with in-house count table:
  snakemake --cores 4 --config count_table=data/chr22_gencode_counts.txt

  # Full pipeline with mock data:
  snakemake --cores 4

  # Run only QC step:
  snakemake --cores 4 qc

  # Run only up to feature extraction:
  snakemake --cores 4 features

  # Dry run (show what will be executed):
  snakemake -n

  # Generate DAG visualization:
  snakemake --dag | dot -Tpng > dag.png
"""

import yaml
from pathlib import Path

# ── Load config ──────────────────────────────────────────────────────────────
configfile: "config/config.yaml"

OUTPUT = config.get("output_dir", "outputs")
SRC = "src"


# ── Default target ───────────────────────────────────────────────────────────
rule all:
    input:
        f"{OUTPUT}/qc/qc_summary.csv",
        f"{OUTPUT}/features.csv",
        f"{OUTPUT}/embeddings.csv",
        f"{OUTPUT}/cibersort/CIBERSORTx_Results.txt",
        f"{OUTPUT}/figures/drift_metrics.csv",
        f"{OUTPUT}/figures/embedding_trajectory.png",
        f"{OUTPUT}/figures/cell_proportions.png",


# ── Step 0: Load data + ML-based QC ─────────────────────────────────────────
rule qc:
    """Load input data and run Isolation Forest QC on co-barcode/gene distributions."""
    output:
        qc_summary = f"{OUTPUT}/qc/qc_summary.csv",
        qc_results = f"{OUTPUT}/qc/qc_results.json",
        filtered_data = f"{OUTPUT}/qc/filtered_samples.pkl",
    params:
        count_table = config.get("count_table"),
        count_format = config.get("count_format", "auto"),
        data_dir = config.get("data_dir", "data/mock"),
        contamination = config["qc"]["contamination"],
        method = config["qc"]["method"],
    threads: 1
    script:
        f"{SRC}/snakemake_steps/step_qc.py"


# ── Step 1: Feature extraction ───────────────────────────────────────────────
rule features:
    """Calculate Shannon entropy and dominant cluster fraction per gene."""
    input:
        filtered_data = rules.qc.output.filtered_data,
    output:
        features = f"{OUTPUT}/features.csv",
    script:
        f"{SRC}/snakemake_steps/step_features.py"


# ── Step 2: scGPT zero-shot embedding ────────────────────────────────────────
rule embedding:
    """Extract 512-dim zero-shot embeddings via scGPT-blood foundation model."""
    input:
        features = rules.features.output.features,
    output:
        embeddings = f"{OUTPUT}/embeddings.csv",
    params:
        model_dir = config["scgpt"]["model_dir"],
        max_genes = config["scgpt"]["max_genes"],
    script:
        f"{SRC}/snakemake_steps/step_embedding.py"


# ── Step 3: CIBERSORTx deconvolution prep ────────────────────────────────────
rule deconvolution:
    """Prepare CIBERSORTx mixture file and parse results (or generate mock)."""
    input:
        features = rules.features.output.features,
    output:
        mixture = f"{OUTPUT}/cibersort/mixture_2024_2026.txt",
        results = f"{OUTPUT}/cibersort/CIBERSORTx_Results.txt",
    params:
        use_mock = config["cibersort"].get("use_mock", True),
        real_results = config["cibersort"].get("results_file"),
    script:
        f"{SRC}/snakemake_steps/step_deconvolution.py"


# ── Step 4: Drift analysis + visualization ───────────────────────────────────
rule visualize:
    """Calculate drift metrics and generate trajectory/proportion/fingerprint plots."""
    input:
        embeddings = rules.embedding.output.embeddings,
        proportions = rules.deconvolution.output.results,
        features = rules.features.output.features,
    output:
        drift_metrics = f"{OUTPUT}/figures/drift_metrics.csv",
        trajectory = f"{OUTPUT}/figures/embedding_trajectory.png",
        proportions_plot = f"{OUTPUT}/figures/cell_proportions.png",
    script:
        f"{SRC}/snakemake_steps/step_visualize.py"


# ── Standalone UMI QC (can be run independently) ─────────────────────────────
rule umi_qc_standalone:
    """Run UMI QC as a standalone tool on any count table."""
    input:
        count_table = config.get("count_table", "data/chr22_gencode_counts.txt"),
    output:
        qc_dir = directory(f"{OUTPUT}/standalone_qc/"),
    params:
        contamination = config["qc"]["contamination"],
        format = config.get("count_format", "auto"),
    shell:
        """
        python {SRC}/umi_qc_cli.py \
            --input {input.count_table} \
            --format {params.format} \
            --contamination {params.contamination} \
            --output-dir {output.qc_dir}
        """


# ── Evaluation / Benchmarking (optional) ─────────────────────────────────────
rule benchmark:
    """Run evaluation framework comparing scGPT embedding drift vs PCA baseline."""
    input:
        features = rules.features.output.features,
        embeddings = rules.embedding.output.embeddings,
    output:
        report = f"{OUTPUT}/evaluation/benchmark_report.json",
        comparison_plot = f"{OUTPUT}/evaluation/scgpt_vs_pca.png",
    params:
        bootstrap_n = config["evaluation"]["bootstrap_n"],
    script:
        f"{SRC}/snakemake_steps/step_benchmark.py"

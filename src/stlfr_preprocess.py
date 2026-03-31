"""
stLFR FASTQ Preprocessing Module

Parses raw MGI stLFR co-barcoded FASTQ files and generates
Gene-level Co-barcode Count Tables as input for the downstream pipeline.

Upstream workflow:
  Raw stLFR FASTQ → barcode extraction → alignment (STAR/HISAT2) → co-barcode clustering → count table

This module wraps the key steps or accepts pre-aligned BAM with co-barcode tags.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StLFRPreprocessor:
    """
    Converts raw MGI stLFR FASTQ into Gene-level Co-barcode Count Table.
    
    stLFR reads carry a 3-part co-barcode in the read name or index read:
        @read_id#barcode1_barcode2_barcode3
    Reads sharing the same composite barcode originate from the same long DNA/RNA fragment.
    """
    
    def __init__(
        self,
        reference_genome: str = "./data/references/GRCh38.latest.fa",
        gtf_file: str = "./data/references/gencode.v49.annotation.gtf",
        aligner: str = "hisat2",
        threads: int = 8,
    ):
        self.reference_genome = reference_genome
        self.gtf_file = gtf_file
        self.aligner = aligner
        self.threads = threads
    
    def extract_barcodes_from_fastq(
        self,
        fastq_r1: str,
        fastq_r2: str,
        output_dir: str,
    ) -> Tuple[str, str]:
        """
        Extract stLFR co-barcodes from read names and write barcode-tagged FASTQ.
        
        stLFR read name format: @readID#barcode1_barcode2_barcode3/1
        The composite barcode (barcode1_barcode2_barcode3) identifies the long fragment.
        
        Args:
            fastq_r1: Path to Read 1 FASTQ (gzipped)
            fastq_r2: Path to Read 2 FASTQ (gzipped)
            output_dir: Output directory
        Returns:
            (tagged_r1_path, barcode_stats_path)
        """
        import gzip
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tagged_r1 = output_dir / "tagged_R1.fastq.gz"
        barcode_stats_path = output_dir / "barcode_stats.csv"
        
        barcode_counts = defaultdict(int)
        total_reads = 0
        barcoded_reads = 0
        
        barcode_pattern = re.compile(r'#(\d+_\d+_\d+)')
        
        with gzip.open(fastq_r1, 'rt') as fin, gzip.open(str(tagged_r1), 'wt') as fout:
            while True:
                header = fin.readline().strip()
                if not header:
                    break
                seq = fin.readline().strip()
                plus = fin.readline().strip()
                qual = fin.readline().strip()
                
                total_reads += 1
                
                match = barcode_pattern.search(header)
                if match:
                    barcode = match.group(1)
                    barcode_counts[barcode] += 1
                    barcoded_reads += 1
                    fout.write(f"{header}\tCB:Z:{barcode}\n{seq}\n{plus}\n{qual}\n")
                else:
                    fout.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
        
        stats_df = pd.DataFrame([
            {"barcode": bc, "read_count": cnt}
            for bc, cnt in barcode_counts.items()
        ])
        stats_df.to_csv(barcode_stats_path, index=False)
        
        logger.info(f"Extracted barcodes: {barcoded_reads}/{total_reads} reads "
                     f"({barcoded_reads/total_reads*100:.1f}%), "
                     f"{len(barcode_counts)} unique co-barcodes")
        
        return str(tagged_r1), str(barcode_stats_path)
    
    def align_reads(
        self,
        fastq_r1: str,
        fastq_r2: Optional[str],
        output_dir: str,
    ) -> str:
        """
        Align stLFR reads to reference genome using HISAT2 or STAR.
        
        Returns:
            Path to sorted BAM file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sam_file = output_dir / "aligned.sam"
        bam_file = output_dir / "aligned.sorted.bam"
        
        if self.aligner == "hisat2":
            index_base = self.reference_genome.replace(".fa", "")
            cmd = [
                "hisat2",
                "-x", index_base,
                "-1", fastq_r1,
                "-p", str(self.threads),
                "--dta",
                "-S", str(sam_file),
            ]
            if fastq_r2:
                cmd.extend(["-2", fastq_r2])
        elif self.aligner == "star":
            genome_dir = str(Path(self.reference_genome).parent / "STAR_index")
            cmd = [
                "STAR",
                "--runThreadN", str(self.threads),
                "--genomeDir", genome_dir,
                "--readFilesIn", fastq_r1,
                "--outSAMtype", "SAM",
                "--outFileNamePrefix", str(output_dir / "star_"),
            ]
            if fastq_r2:
                cmd[cmd.index(fastq_r1)] = f"{fastq_r1} {fastq_r2}"
            sam_file = output_dir / "star_Aligned.out.sam"
        else:
            raise ValueError(f"Unsupported aligner: {self.aligner}")
        
        logger.info(f"Running alignment: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Sort and index
        subprocess.run(["samtools", "sort", "-@", str(self.threads),
                         "-o", str(bam_file), str(sam_file)], check=True)
        subprocess.run(["samtools", "index", str(bam_file)], check=True)
        
        logger.info(f"Alignment complete: {bam_file}")
        return str(bam_file)
    
    def generate_co_barcode_count_table(
        self,
        bam_file: str,
        output_dir: str,
    ) -> pd.DataFrame:
        """
        Generate Gene-level Co-barcode Count Table from aligned BAM.
        
        For each read:
          1. Extract gene assignment (from featureCounts or NH tag)
          2. Extract co-barcode (from CB tag or read name)
          3. Group by (gene_id, co_barcode) → count reads
        
        Output schema: (gene_id, co_barcode_cluster_id, count)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run featureCounts to assign reads to genes
        count_file = output_dir / "featureCounts_output.txt"
        assigned_bam = output_dir / "featureCounts_output.txt.featureCounts.bam"
        
        cmd = [
            "featureCounts",
            "-a", self.gtf_file,
            "-o", str(count_file),
            "-R", "BAM",
            "-t", "exon",
            "-g", "gene_name",
            "-T", str(self.threads),
            bam_file,
        ]
        
        logger.info(f"Running featureCounts: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Step 2: Parse assigned BAM to extract (gene, co-barcode) pairs
        import pysam
        
        gene_barcode_counts = defaultdict(lambda: defaultdict(int))
        barcode_pattern = re.compile(r'#(\d+_\d+_\d+)')
        
        bam = pysam.AlignmentFile(str(assigned_bam), "rb")
        for read in bam:
            # Get gene assignment from featureCounts tag
            if not read.has_tag("XS"):
                gene = read.get_tag("XS") if read.has_tag("XS") else None
            else:
                gene = read.get_tag("XS")
            
            if gene is None or gene == "Unassigned":
                continue
            
            # Get co-barcode from CB tag or read name
            if read.has_tag("CB"):
                barcode = read.get_tag("CB")
            else:
                match = barcode_pattern.search(read.query_name)
                barcode = match.group(1) if match else "unknown"
            
            if barcode != "unknown":
                gene_barcode_counts[gene][barcode] += 1
        
        bam.close()
        
        # Step 3: Build count table
        rows = []
        for gene_id, barcodes in gene_barcode_counts.items():
            for barcode, count in barcodes.items():
                rows.append({
                    "gene_id": gene_id,
                    "co_barcode_cluster_id": f"{gene_id}_{barcode}",
                    "count": count,
                })
        
        count_table = pd.DataFrame(rows)
        output_path = output_dir / "co_barcode_count_table.csv"
        count_table.to_csv(output_path, index=False)
        
        logger.info(f"Generated co-barcode count table: {len(count_table)} rows, "
                     f"{count_table['gene_id'].nunique()} genes, "
                     f"{count_table['co_barcode_cluster_id'].nunique()} clusters")
        
        return count_table
    
    def process(
        self,
        fastq_r1: str,
        fastq_r2: Optional[str] = None,
        output_dir: str = "./outputs/stlfr_preprocess",
    ) -> pd.DataFrame:
        """
        Full upstream pipeline: FASTQ → barcode extraction → alignment → count table.
        """
        output_dir_path = Path(output_dir)
        
        logger.info("=== Step 1/3: Extracting stLFR co-barcodes ===")
        tagged_r1, barcode_stats = self.extract_barcodes_from_fastq(
            fastq_r1, fastq_r2, str(output_dir_path / "barcodes")
        )
        
        logger.info("=== Step 2/3: Aligning to reference genome ===")
        bam_file = self.align_reads(
            tagged_r1, fastq_r2, str(output_dir_path / "alignment")
        )
        
        logger.info("=== Step 3/3: Generating co-barcode count table ===")
        count_table = self.generate_co_barcode_count_table(
            bam_file, str(output_dir_path / "counts")
        )
        
        return count_table


def load_count_table(
    count_table_path: str,
    format: str = "auto",
    sample_id_map: Dict[str, str] = None,
    min_count: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Load an in-house or standard count table and convert to the pipeline's
    internal format: Dict[sample_id -> DataFrame(co_barcode_cluster_id, gene_id, count)].

    Supported formats:
      - "featurecounts": featureCounts output (skip comment lines, columns: Geneid, Chr, Start, End, Strand, Length, sample1.bam, ...)
      - "csv": Generic CSV/TSV with rows=genes, columns=samples (first column = gene names)
      - "auto": Auto-detect from file header

    For standard bulk RNA-seq (no co-barcodes), each gene becomes a single
    co-barcode cluster (co_barcode_cluster_id == gene_id). This allows
    the downstream QC and feature extraction to run identically.

    Args:
        count_table_path: Path to count table file.
        format: One of "featurecounts", "csv", "auto".
        sample_id_map: Optional dict mapping column names to desired sample IDs.
                       e.g. {"data/2024.bam": "2024", "data/2025.bam": "2025"}
                       If None, column names are cleaned automatically (strip path + .bam).
        min_count: Drop genes with total count < min_count across all samples.

    Returns:
        Dict[sample_id, DataFrame] where each DataFrame has columns:
        ['co_barcode_cluster_id', 'gene_id', 'count']
    """
    path = Path(count_table_path)
    logger.info(f"Loading count table from {path}...")

    # --- Read raw file, detect format ---
    with open(path, 'r') as f:
        first_line = f.readline()

    if format == "auto":
        if first_line.startswith("#") or first_line.startswith("# Program:featureCounts"):
            format = "featurecounts"
        else:
            format = "csv"
        logger.info(f"Auto-detected format: {format}")

    if format == "featurecounts":
        # featureCounts: skip comment lines (starting with #), header row follows
        df = pd.read_csv(path, sep='\t', comment='#')
        # Standard featureCounts columns: Geneid, Chr, Start, End, Strand, Length, then sample columns
        meta_cols = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length']
        sample_cols = [c for c in df.columns if c not in meta_cols]
        gene_col = 'Geneid'
    elif format == "csv":
        # Try tab first, fall back to comma
        sep = '\t' if '\t' in first_line else ','
        df = pd.read_csv(path, sep=sep, index_col=0)
        sample_cols = list(df.columns)
        gene_col = df.index.name or 'gene_id'
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'Geneid'}, inplace=True)
        gene_col = 'Geneid'
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Found {len(df)} genes, {len(sample_cols)} sample columns: {sample_cols}")

    # --- Build sample_id mapping ---
    if sample_id_map is None:
        sample_id_map = {}
        for col in sample_cols:
            # Strip common suffixes: path + .bam / .sam
            clean = Path(col).stem  # e.g. "data/chr22.bam" -> "chr22"
            sample_id_map[col] = clean

    # --- Filter low-count genes ---
    count_matrix = df[sample_cols].values
    row_totals = count_matrix.sum(axis=1)
    keep = row_totals >= min_count
    df = df[keep].copy()
    logger.info(f"Kept {keep.sum()} genes with total count >= {min_count}")

    # --- Convert to per-sample DataFrames ---
    samples_data = {}
    for col in sample_cols:
        sid = sample_id_map.get(col, col)
        gene_names = df[gene_col].values
        counts = df[col].values.astype(int)

        # For bulk RNA-seq: one "cluster" per gene
        sample_df = pd.DataFrame({
            'co_barcode_cluster_id': gene_names,
            'gene_id': gene_names,
            'count': counts,
        })
        # Drop zero-count rows for this sample
        sample_df = sample_df[sample_df['count'] > 0].reset_index(drop=True)
        samples_data[sid] = sample_df
        logger.info(f"  Sample '{sid}': {len(sample_df)} genes with count > 0")

    return samples_data


def generate_mock_stlfr_count_table(
    n_genes: int = 505,
    n_samples: int = 3,
    output_dir: str = "./data/mock",
) -> Dict[str, pd.DataFrame]:
    """
    Generate mock stLFR co-barcode count tables without running actual alignment.
    Simulates temporal drift for testing the downstream pipeline.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    
    gene_names = [f"GENE_{i}" for i in range(n_genes - 5)] + ["CD45", "TCR", "FOXP3", "IL7R", "CD8A"]
    sample_ids = [str(2024 + i) for i in range(n_samples)]
    
    samples_data = {}
    
    for si, sample_id in enumerate(sample_ids):
        rows = []
        for gene in gene_names:
            n_clusters = np.random.randint(1, 5)
            base_count = np.random.poisson(100)
            drift_factor = 1.0 + si * 0.15
            
            for ci in range(n_clusters):
                cluster_id = f"{gene}_cluster{ci}"
                count = int(np.random.poisson(max(base_count / n_clusters, 1)) * drift_factor)
                
                if gene in ["CD45", "FOXP3"] and ci == 0:
                    count = int(count * max(0.1, 1.0 - si * 0.4))
                
                rows.append({
                    "co_barcode_cluster_id": cluster_id,
                    "gene_id": gene,
                    "count": count,
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/sample_{sample_id}.csv", index=False)
        samples_data[sample_id] = df
    
    logger.info(f"Generated mock data for {len(samples_data)} samples in {output_dir}")
    return samples_data

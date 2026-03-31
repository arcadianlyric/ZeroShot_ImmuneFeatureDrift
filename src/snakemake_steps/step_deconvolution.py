"""Snakemake step: CIBERSORTx deconvolution prep + mock/real results."""
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deconvolution_prep import CibersortPrep

features_df = pd.read_csv(snakemake.input.features)
output_dir = Path(snakemake.output.mixture).parent
output_dir.mkdir(parents=True, exist_ok=True)

cibersort_prep = CibersortPrep(output_dir=str(output_dir))
mixture_file = cibersort_prep.prepare_mixture_file(features_df)

use_mock = snakemake.params.get("use_mock", True)
real_results = snakemake.params.get("real_results")

if not use_mock and real_results and Path(real_results).exists():
    shutil.copy(real_results, snakemake.output.results)
else:
    sample_ids = sorted(features_df['sample_id'].unique())
    mock_props = pd.DataFrame(
        np.random.dirichlet(np.ones(5), size=len(sample_ids)),
        index=sample_ids,
        columns=["T_CD4", "T_CD8", "B_cells", "Monocytes", "NK_cells"],
    )
    mock_props.to_csv(snakemake.output.results, sep='\t')

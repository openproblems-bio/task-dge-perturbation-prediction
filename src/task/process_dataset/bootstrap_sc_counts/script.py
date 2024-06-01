import anndata as ad
import numpy as np

# VIASH START
par = {
    "input": "resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad",
    "output": "output/sc_counts_bootstrapped_*.h5ad",
    "obs_fraction": 0.95,
    "var_fraction": 1
}
# VIASH END

# Load data
input_data = ad.read_h5ad(par["input"])

# Sample indices
obs_ix = np.random.choice(
    input_data.obs_names,
    int(input_data.n_obs * par["obs_fraction"]),
    replace=False
)
var_ix = np.random.choice(
    input_data.var_names,
    int(input_data.n_vars * par["var_fraction"]),
    replace=False
)

# Subset AnnData object
output_data = input_data[obs_ix, var_ix].copy()

# Write output
output_data.write_h5ad(par["output"], compression="gzip")

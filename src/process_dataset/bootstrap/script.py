import anndata as ad
import numpy as np

# VIASH START
par = {
    "input": "resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad",
    "output": "output/sc_counts_bootstrapped_*.h5ad",
    "bootstrap_obs": True,
    "obs_fraction": 1,
    "obs_replace": True,
    "bootstrap_var": False,
    "var_fraction": 1,
    "var_replace": True
}
# VIASH END

# Load data
data = ad.read_h5ad(par["input"])

if par["bootstrap_obs"]:
    # Sample indices
    obs_ix = np.random.choice(
        range(data.n_obs),
        int(data.n_obs * par["obs_fraction"]),
        replace=par["obs_replace"]
    )
    data = data[obs_ix, :]

if par["bootstrap_var"]:
    # Sample indices
    var_ix = np.random.choice(
        range(data.n_vars),
        int(data.n_vars * par["var_fraction"]),
        replace=par["var_replace"]
    )
    data = data[:, var_ix]

# Write output
data.write_h5ad(par["output"], compression="gzip")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

## VIASH START
par = {
    "input": "resources/neurips-2023-data/sc_counts_cleaned.h5ad",
    "output": "resources/neurips-2023-data/pseudobulk.h5ad",
}
## VIASH END

def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    """
    Adapted from this forum post:
    https://discourse.scverse.org/t/group-sum-rows-based-on-jobs-feature/371/4
    """

    # assert pd.api.types.is_categorical_dtype(adata.obs[col])
    assert isinstance(adata.obs[col].dtypes, pd.CategoricalDtype)

    # sum `.X` entries for each unique value in `col`
    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (
            np.broadcast_to(True, adata.n_obs),
            (cat.codes, np.arange(adata.n_obs))
        ),
        shape=(len(cat.categories), adata.n_obs),
    )
    sum_adata = ad.AnnData(
        var=adata.var,
        obs=pd.DataFrame(index=cat.categories),
    )
    if adata.X is not None:
        sum_adata.X = indicator @ adata.X
    for layer in adata.layers:
        sum_adata.layers[layer] = indicator @ adata.layers[layer]

    # copy over `.obs` values that have a one-to-one-mapping with `.obs[col]`
    obs_cols = list(set(adata.obs.columns) - set([col]))

    one_to_one_mapped_obs_cols = []
    nunique_in_col = adata.obs[col].nunique()
    for other_col in obs_cols:
        if len(adata.obs[[col, other_col]].drop_duplicates()) == nunique_in_col:
            one_to_one_mapped_obs_cols.append(other_col)

    joining_df = adata.obs[[col] + one_to_one_mapped_obs_cols].drop_duplicates().set_index(col)
    assert (sum_adata.obs.index == sum_adata.obs.join(joining_df).index).all()
    sum_adata.obs = sum_adata.obs.join(joining_df)
    sum_adata.obs.index.name = col
    sum_adata.obs = sum_adata.obs.reset_index()
    sum_adata.obs.index = sum_adata.obs.index.astype('str')

    return sum_adata


print(">> Load dataset", flush=True)
sc_counts = ad.read_h5ad(par["input"])

print(">> Create pseudobulk dataset", flush=True)
bulk_adata = sum_by(sc_counts, 'plate_well_cell_type')
bulk_adata.obs = bulk_adata.obs.drop(columns=['plate_well_cell_type'])

print(">> Remove samples with no counts", flush=True)
bulk_adata = bulk_adata[bulk_adata.X.todense().sum(axis=1) > 0]

print(">> Save dataset", flush=True)
bulk_adata.write_h5ad(par["output"], compression="gzip")
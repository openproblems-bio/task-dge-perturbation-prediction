import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

## VIASH START
par = {
    "input": "resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad",
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

print(">> Keep only raw counts", flush=True)
sc_counts.X = sc_counts.raw.X
del sc_counts.raw

print(">> Fix splits after reannotation", flush=True)
sc_counts.obs["cell_type_orig_updated"] = sc_counts.obs["cell_type_orig"].apply(lambda x: "T cells" if x.startswith("T ") else x)
sc_counts.obs["sm_cell_type_orig"] = sc_counts.obs["sm_name"].astype(str) + "_" + sc_counts.obs["cell_type_orig_updated"].astype(str)
mapping_to_split = sc_counts.obs.groupby("sm_cell_type_orig")["split"].apply(lambda x: x.unique()[0]).to_dict()
sc_counts.obs["sm_cell_type"] = sc_counts.obs["sm_name"].astype(str) + "_" + sc_counts.obs["cell_type"].astype(str)
sc_counts.obs["split"] = sc_counts.obs["sm_cell_type"].map(mapping_to_split)
sc_counts.obs['control'] = sc_counts.obs['split'].eq("control")

print(">> Create pseudobulk dataset", flush=True)
bulk_adata = sum_by(sc_counts, 'plate_well_celltype_reannotated')
bulk_adata.obs = bulk_adata.obs.drop(columns=['plate_well_celltype_reannotated'])

print(">> Remove samples with no counts", flush=True)
bulk_adata = bulk_adata[bulk_adata.X.todense().sum(axis=1) > 0]

bulk_adata.uns["single_cell_obs"] = sc_counts.obs

print(">> Save dataset", flush=True)
bulk_adata.write_h5ad(par["output"], compression="gzip")
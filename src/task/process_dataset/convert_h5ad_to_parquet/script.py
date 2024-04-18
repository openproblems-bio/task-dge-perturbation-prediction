import anndata as ad
import pandas as pd

## VIASH START
par = {
    "input_train": "resources/neurips-2023-data/de_train.h5ad",
    "input_test": "resources/neurips-2023-data/de_test.h5ad",
    "output_train": "resources/neurips-2023-data/de_train.parquet",
    "output_test": "resources/neurips-2023-data/de_test.parquet",
    "output_id_map": "resources/neurips-2023-data/id_map.csv",
}
## VIASH END


def anndata_to_dataframe(adata):
  metadata = adata.obs[['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control']]

  sign_logfc_pval = pd.DataFrame(
    adata.layers["sign_log10_pval"],
    columns=adata.var_names,
    index=adata.obs.index
  )

  return pd.concat([metadata, sign_logfc_pval], axis=1).reset_index(drop=True)

print(">> Load dataset", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

print(">> Convert AnnData to DataFrame", flush=True)
de_train = anndata_to_dataframe(input_train)
de_test = anndata_to_dataframe(input_test)

print(">> Add 'id' to test", flush=True)
de_test.reset_index(drop=True, inplace=True)
de_test.reset_index(names="id", inplace=True)

print(">> Create id_map data frame", flush=True)
id_map = de_test[["id", "sm_name", "cell_type"]]

print(">> Save data", flush=True)
de_train.to_parquet(par["output_train"])
de_test.to_parquet(par["output_test"])
id_map.to_csv(par["output_id_map"], index=False)

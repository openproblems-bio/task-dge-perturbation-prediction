import anndata as ad
import pandas as pd

## VIASH START
par = {
    "input_train": "resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_train.h5ad",
    "input_test": "resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_test.h5ad",
    "output_train": "resources/neurips-2023-kaggle/de_train.parquet",
    "output_train_h5ad": "resources/neurips-2023-kaggle/de_train.h5ad",
    "output_test": "resources/neurips-2023-kaggle/de_test.parquet",
    "output_test_h5ad": "resources/neurips-2023-kaggle/de_test.h5ad",
    "output_id_map": "resources/neurips-2023-kaggle/id_map.csv",
}
## VIASH END


def anndata_to_dataframe(adata, add_id=False):
  obs_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control']
  if add_id:
    obs_cols = ["id"] + obs_cols

  metadata = adata.obs[obs_cols]

  sign_logfc_pval = pd.DataFrame(
    adata.layers["sign_log10_pval"],
    columns=adata.var_names,
    index=adata.obs.index
  )

  return pd.concat([metadata, sign_logfc_pval], axis=1).reset_index(drop=True)

print(">> Load dataset", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

print(">> Add 'control' column", flush=True)
input_train.obs["control"] = input_train.obs["split"] == "control"
input_test.obs["control"] = input_test.obs["split"] == "control"

print(">> Add 'id' to test", flush=True)
input_test.obs.reset_index(drop=True, inplace=True)
input_test.obs.reset_index(names="id", inplace=True)

print(">> Move X to layers", flush=True)
input_train.layers["sign_log10_pval"] = input_train.X
input_test.layers["sign_log10_pval"] = input_test.X
del input_train.X
del input_test.X

print(">> Convert AnnData to DataFrame", flush=True)
de_train = anndata_to_dataframe(input_train)
de_test = anndata_to_dataframe(input_test, add_id=True)

print(">> Create id_map data frame", flush=True)
id_map = de_test[["id", "sm_name", "cell_type"]]

print(">> Save data", flush=True)
input_train.write(par["output_train_h5ad"], compression=9)
input_test.write(par["output_test_h5ad"], compression=9)
de_train.to_parquet(par["output_train"])
de_test.to_parquet(par["output_test"])
id_map.to_csv(par["output_id_map"], index=False)

import anndata as ad
import numpy as np

## VIASH START
par = {
    "input_train": "resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_train.h5ad",
    "input_test": "resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_test.h5ad",
    "input_single_cell_h5ad": "resources/neurips-2023-raw/sc_counts.h5ad",
    "output_train_h5ad": "resources/neurips-2023-kaggle/de_train.h5ad",
    "output_test_h5ad": "resources/neurips-2023-kaggle/de_test.h5ad",
    "output_id_map": "resources/neurips-2023-kaggle/id_map.csv",
}
## VIASH END

print(">> Load dataset", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])
input_sc = ad.read_h5ad(par["input_single_cell_h5ad"])

print(">> Add 'control' column", flush=True)
input_train.obs["control"] = input_train.obs["split"] == "control"
input_test.obs["control"] = input_test.obs["split"] == "control"

print(">> Add 'id' to test", flush=True)
input_test.obs.reset_index(drop=True, inplace=True)
input_test.obs.reset_index(names="id", inplace=True)

print(">> Move X to layers", flush=True)
input_train.layers["sign_log10_pval"] = input_train.X
input_test.layers["sign_log10_pval"] = input_test.X
clip_val = -np.log10(0.0001)
input_train.layers["clipped_sign_log10_pval"] = np.clip(input_train.X, -clip_val, clip_val)
input_test.layers["clipped_sign_log10_pval"] = np.clip(input_test.X, -clip_val, clip_val)
del input_train.X
del input_test.X

print(">> Store metadata in uns", flush=True)
for key in ["dataset_id", "dataset_name", "dataset_url", "dataset_reference",\
            "dataset_summary", "dataset_description", "dataset_organism"]:
  input_train.uns[key] = par[key]
  input_test.uns[key] = par[key]

new_single_cell_obs = input_sc.obs[input_sc.obs.split.isin(input_train.obs.split.unique())]
input_train.uns["single_cell_obs"] = new_single_cell_obs
input_test.uns["single_cell_obs"] = input_sc.obs

print(">> Create id_map data frame", flush=True)
id_map = input_test.obs[["id", "sm_name", "cell_type"]]

print(">> Save data", flush=True)
input_train.write(par["output_train_h5ad"], compression=9)
input_test.write(par["output_test_h5ad"], compression=9)
id_map.to_csv(par["output_id_map"], index=False)

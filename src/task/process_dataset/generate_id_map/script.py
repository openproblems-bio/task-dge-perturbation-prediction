import anndata as ad

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "id_map": "resources/neurips-2023-data/id_map.csv",
}
## VIASH END

print(">> Load dataset", flush=True)
input_test = ad.read_h5ad(par["input_test"])

print(">> Generate id_map file", flush=True)
id_map = input_test.obs[["sm_name", "cell_type"]]
id_map.reset_index(drop=True, inplace=True)
id_map.reset_index(names="id", inplace=True)

print(">> Save data", flush=True)
id_map.to_csv(par["output_id_map"], index=False)

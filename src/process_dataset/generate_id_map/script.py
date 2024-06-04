import anndata as ad

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "id_map": "resources/neurips-2023-data/id_map.csv",
}
## VIASH END

print(">> Load dataset", flush=True)
de_test_h5ad = ad.read_h5ad(par["de_test_h5ad"])

print(">> Generate id_map file", flush=True)
id_map = de_test_h5ad.obs[["sm_name", "cell_type"]]
id_map.reset_index(drop=True, inplace=True)
id_map.reset_index(names="id", inplace=True)

print(">> Save data", flush=True)
id_map.to_csv(par["id_map"], index=False)

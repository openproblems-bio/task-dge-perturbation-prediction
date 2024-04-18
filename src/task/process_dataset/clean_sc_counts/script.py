import pandas as pd
import anndata as ad

## VIASH START
par = {
    "input": "resources/neurips-2023-raw/sc_counts.h5ad",
    "lincs_id_compound_mapping": "resources/neurips-2023-raw/lincs_id_compound_mapping.parquet",
    "output": "resources/neurips-2023-data/sc_counts_cleaned.h5ad",
}
## VIASH END

print(">> Load dataset", flush=True)
sc_counts = ad.read_h5ad(par["input"])
lincs_id_compound_mapping = pd.read_parquet(par["lincs_id_compound_mapping"])

print(">> Process dataset obs", flush=True)
sc_counts.obs = sc_counts.obs.reset_index().rename(columns={'index': 'obs_id'})
sc_counts.obs['control'] = sc_counts.obs['split'].eq("control")
sc_counts.obs = sc_counts.obs.drop(columns=["hashtag_id", "raw_cell_id", "container_format"])
sc_counts.obs["SMILES"] = sc_counts.obs["sm_name"].map(lincs_id_compound_mapping.set_index("sm_name")["smiles"])
sc_counts.obs["sm_lincs_id"] = sc_counts.obs["sm_name"].map(lincs_id_compound_mapping.set_index("sm_name")["sm_lincs_id"])
sc_counts.obs['library_id'] = (sc_counts.obs['plate_name'].astype(str) + '_' + sc_counts.obs['row'].astype(str)).astype('category')
sc_counts.obs = sc_counts.obs.set_index("obs_id")
sc_counts.obs['plate_well_cell_type'] = sc_counts.obs['plate_name'].astype('str') \
    + '_' + sc_counts.obs['well'].astype('str') \
    + '_' + sc_counts.obs['cell_type'].astype('str')
sc_counts.obs['plate_well_cell_type'] = sc_counts.obs['plate_well_cell_type'].astype('category')

print(">> Remove normalized counts, store only raw counts", flush=True)
sc_counts.X = sc_counts.layers["counts"].copy()
del sc_counts.layers["counts"]

print(">> Save dataset", flush=True)
sc_counts.write_h5ad(par["output"], compression="gzip")
import pandas as pd
import numpy as np

import anndata as ad
from tempfile import TemporaryDirectory
import os


## VIASH START
par = {
    "sc_counts": "resources/neurips-2023-raw/sc_counts.h5ad",
    "lincs_id_compound_mapping": "resources/neurips-2023-raw/lincs_id_compound_mapping.parquet",
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "de_test": "resources/neurips-2023-data/de_test.parquet",
    "id_map": "resources/neurips-2023-data/id_map.csv"
}
meta = {
    "resources_dir": "src/dge_perturbation_prediction/process_dataset",
}
## VIASH END

# import helper functions
import sys
sys.path.append(meta["resources_dir"])

from utils import sum_by, create_split_mapping, make_r_safe_names, _run_limma_for_cell_type, convert_de_df_to_anndata, anndata_to_dataframe
import limma_utils
import imp
imp.reload(limma_utils)

print(">> Load dataset", flush=True)
sc_counts = ad.read_h5ad(par["sc_counts"])
lincs_id_compound_mapping = pd.read_parquet(par["lincs_id_compound_mapping"])

print(">> Process dataset", flush=True)
# adapt column names and data fields
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

sc_counts.X = sc_counts.layers["counts"].copy()
del sc_counts.layers["counts"]

# compute pseudobulk
bulk_adata = sum_by(sc_counts, 'plate_well_cell_type')
bulk_adata.obs = bulk_adata.obs.drop(columns=['plate_well_cell_type'])
bulk_adata.X = np.array(bulk_adata.X.todense())

# remove samples with no counts
bulk_adata = bulk_adata[bulk_adata.X.sum(axis=1) > 0].copy()

factors = ['sm_name', 'donor_id', 'plate_name', 'row']

# mapping is created to recover the original names after transforming them for limma
mapping_sm_lincs_id_to_sm_name = bulk_adata.obs.set_index('sm_lincs_id')['sm_name'].to_dict()
# Apply the renaming function only to the entries of selected columns
for col in factors:
    if col in bulk_adata.obs.columns:
        bulk_adata.obs[col] = bulk_adata.obs[col].apply(make_r_safe_names)

mapping_split = create_split_mapping(bulk_adata.obs)
# Run limma
cell_types = bulk_adata.obs['cell_type'].unique()

# save limma output in temporary directory
temp_prefix = os.path.expanduser('~') + '/.tmp/limma/'
if not os.path.exists(temp_prefix):
    os.makedirs(temp_prefix)

de_dfs_train = []
de_dfs_test = []
# introduce train/test split so that the training output isn't biased
train_adata = bulk_adata[bulk_adata.obs['split'] != 'private_test'].copy()
test_cell_types = bulk_adata[bulk_adata.obs['split'] == 'private_test'].obs.cell_type.unique()
with TemporaryDirectory(prefix=temp_prefix) as tempdirname:
    data_dir = tempdirname + '/'
    for cell_type in cell_types:
        # calculate DE separately for test and train data to avoid bias
        if cell_type in test_cell_types:
            cell_type_selection = bulk_adata.obs['cell_type'].eq(cell_type)
            cell_type_bulk_adata = bulk_adata[cell_type_selection].copy()
            print(f">> Run limma for test data {cell_type}", flush=True)
            de_df = _run_limma_for_cell_type(cell_type_bulk_adata, data_dir, "Rscript", meta["resources_dir"])
            de_dfs_test.append(de_df)
        cell_type_selection = train_adata.obs['cell_type'].eq(cell_type)
        cell_type_bulk_adata = train_adata[cell_type_selection].copy()
        print(f">> Run limma for train data {cell_type}", flush=True)
        de_df = _run_limma_for_cell_type(cell_type_bulk_adata, data_dir, "Rscript", meta["resources_dir"])
        de_dfs_train.append(de_df)
print(f">> limma runs completed", flush=True)

de_df_train = pd.concat(de_dfs_train)
de_df_test = pd.concat(de_dfs_test)
de_adata_train = convert_de_df_to_anndata(de_df_train, 0.05)
de_adata_test = convert_de_df_to_anndata(de_df_test, 0.05)
# convert anndata back to dataframe with the format of kaggle data
de_df_train = anndata_to_dataframe(de_adata_train, mapping_split)
de_df_test = anndata_to_dataframe(de_adata_test, mapping_split)

# convert sm_names back to original names
de_df_test['sm_name'] = de_df_test['sm_lincs_id'].map(mapping_sm_lincs_id_to_sm_name)
de_df_train['sm_name'] = de_df_train['sm_lincs_id'].map(mapping_sm_lincs_id_to_sm_name)

de_test = de_df_test[de_df_test.split == "private_test"].copy()
de_test.reset_index(drop=True, inplace=True)
de_test.reset_index(names="id", inplace=True)
id_map = de_test[["id", "cell_type", "sm_name"]]

print(">> Write to disk", flush=True)
de_df_train.to_parquet(par["de_train"])
de_test.to_parquet(par["de_test"])
id_map.to_csv(par["id_map"], index=False)
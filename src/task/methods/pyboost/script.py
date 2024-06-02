# This script is based on an IPython notebook:
# https://github.com/Ambros-M/Single-Cell-Perturbations-2023/blob/main/notebooks/scp-26-py-boost-recommender-system-and-et.ipynb

import anndata as ad
import numpy as np
import pandas as pd
import warnings
import sys

warnings.simplefilter('ignore', FutureWarning)
np.set_printoptions(linewidth=195, edgeitems=3)
pd.set_option("min_rows", 6)

## VIASH START
par = dict(
    de_train_h5ad = "resources/neurips-2023-data/de_train.h5ad",
    layer = "clipped_sign_log10_pval",
    id_map = "resources/neurips-2023-data/id_map.csv",
    predictor_names = ["py_boost"],
    output = "output.h5ad",
)
meta = dict(
    resources_dir = "src/task/methods/pyboost"
)
## VIASH END

sys.path.append(meta["resources_dir"])
from anndata_to_dataframe import anndata_to_dataframe
from helper import predictors

print("Loading data\n", flush=True)
de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])
adata_obs = de_train_h5ad.uns["single_cell_obs"]

id_map = pd.read_csv(par['id_map'], index_col = 0)
# display(id_map)

# 18211 genes
genes = de_train_h5ad.var_names
de_train_indexed = de_train.set_index(['cell_type', 'sm_name'])[genes]

# All 146 sm_names
sm_names = sorted(de_train.sm_name.unique())
# Determine the 17 compounds (including the two control compounds) with data for almost all cell types
train_sm_names = de_train.query("cell_type == 'B cells'").sm_name.sort_values().values
# The other 129 sm_names
test_sm_names = [sm for sm in sm_names if sm not in train_sm_names]
# The three control sm_names
controls3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']

# All 6 cell types
cell_types = list(de_train_h5ad.obs.cell_type.cat.categories)
test_cell_types = list(id_map.cell_type.unique())
train_cell_types = [ct for ct in cell_types if not ct in test_cell_types]

# Cell counts
cell_count = adata_obs.groupby(['cell_type', 'sm_name']).size()
avg_cell_count = cell_count[~cell_count.index.get_level_values('sm_name').isin(controls3)].groupby('cell_type').mean()

# Cell type ratios (extrapolated from 17 train_sm_names)
temp = adata_obs.groupby(['cell_type', 'sm_name']).size().unstack().loc[cell_types]
cell_type_ratio = temp[list(train_sm_names) + ['Dimethyl Sulfoxide']].sum(axis=1)
cell_type_ratio /= cell_type_ratio.sum()

## Model fitting functions

# Define outliers which are excluded from training and validation
# removed_compounds = ['AT13387', 'Alvocidib', 'BAY 61-3606', 'BMS-387032', 
#                      'Belinostat', 'CEP-18770 (Delanzomib)', 'CGM-097', 'CGP 60474', 
#                      'Dabrafenib', 'Ganetespib (STA-9090)', 'I-BET151', 'IN1451', 
#                      'LY2090314', 'MLN 2238', 'Oprozomib (ONX 0912)', 
#                      'Proscillaridin A;Proscillaridin-A', 'Resminostat',
#                      'Scriptaid', 'UNII-BXU45ZH6LI', 'Vorinostat']
removed_compounds = []

# Drop outliers from training
de_tr = de_train_indexed.query("~sm_name.isin(@removed_compounds)")

# Fit all models and average their predictions
pred_list = [predictors[p](de_tr, id_map, train_sm_names, genes, cell_type_ratio)
             for p in par["predictor_names"]]
de_pred = sum(pred_list) / len(pred_list)

# Test for missing values
if de_pred.isna().any().any():
    print("Warning: This submission contains missing values. "
            "Don't submit it!")

# Write the files
print('Write output to file', flush=True)
output = ad.AnnData(
    layers={"prediction": de_pred.values},
    obs=pd.DataFrame(index=id_map.index),
    var=pd.DataFrame(index=genes),
    uns={
      "dataset_id": de_train_h5ad.uns["dataset_id"],
      "method_id": meta["functionality_name"]
    }
)

output.write_h5ad(par["output"], compression="gzip")

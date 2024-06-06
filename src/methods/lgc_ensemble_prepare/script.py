import os
import pandas as pd
import anndata as ad
import numpy as np
import sys
import torch
from sklearn.model_selection import KFold as KF
import json
if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print('using device: cpu', flush=True)

## VIASH START
par = {
    "de_train_h5ad": "resources/neurips-2023-data/de_train.h5ad",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "layer": "clipped_sign_log10_pval",
    "epochs": 10,
    "kf_n_splits": 3,
    "models": ["initial", "light", "heavy"],
    "train_data_aug_dir": "output/train_data_aug_dir",
}
meta = {
    "resources_dir": "src/methods/lgc_ensemble",
    "temp_dir": "/tmp"
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])


from helper_functions import seed_everything, one_hot_encode, save_ChemBERTa_features
from anndata_to_dataframe import anndata_to_dataframe
from helper_functions import combine_features


###################################################################
# interpreted from src/methods/lgc_ensemble/prepare_data.py
# prepare data
seed_everything()

if not os.path.exists(par["train_data_aug_dir"]):
    os.makedirs(par["train_data_aug_dir"], exist_ok=True)

## Read data
print("\nPreparing data...", flush=True)
de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])
de_train = de_train.drop(columns=['split'])
id_map = pd.read_csv(par["id_map"])

gene_names = list(de_train_h5ad.var_names)

print("Create data augmentation", flush=True)
de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()
std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()
std_sm_name = std_sm_name.fillna(0)
cell_types = de_cell_type.groupby('cell_type').quantile(0.1).reset_index()['cell_type'] # This is just to get cell types in the right order for the next line
quantiles_cell_type = pd.concat(
    [pd.DataFrame(cell_types)] +
    [
        de_cell_type.groupby('cell_type')[col].quantile([0.25, 0.50, 0.75], interpolation='linear').unstack().reset_index(drop=True)
        for col in list(de_train.columns)[5:]
    ],
    axis=1
)

print("Save data augmentation features", flush=True)
mean_cell_type.to_csv(f'{par["train_data_aug_dir"]}/mean_cell_type.csv', index=False)
std_cell_type.to_csv(f'{par["train_data_aug_dir"]}/std_cell_type.csv', index=False)
mean_sm_name.to_csv(f'{par["train_data_aug_dir"]}/mean_sm_name.csv', index=False)
std_sm_name.to_csv(f'{par["train_data_aug_dir"]}/std_sm_name.csv', index=False)
quantiles_cell_type.to_csv(f'{par["train_data_aug_dir"]}/quantiles_cell_type.csv', index=False)
with open(f'{par["train_data_aug_dir"]}/gene_names.json', 'w') as f:
    json.dump(gene_names, f)

print("Create one hot encoding features", flush=True)
one_hot_train, _ = one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], out_dir=par["train_data_aug_dir"])
one_hot_train = pd.DataFrame(one_hot_train)

print("Prepare ChemBERTa features", flush=True)
train_chem_feat, train_chem_feat_mean = save_ChemBERTa_features(de_train["SMILES"].tolist(), out_dir=par["train_data_aug_dir"], on_train_data=True)
sm_name2smiles = {smname:smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
test_smiles = list(map(sm_name2smiles.get, id_map['sm_name'].values))
_, _ = save_ChemBERTa_features(test_smiles, out_dir=par["train_data_aug_dir"], on_train_data=False)

###################################################################
# interpreted from src/methods/lgc_ensemble/train.py

## Prepare cross-validation
cell_types_sm_names = de_train[['cell_type', 'sm_name']]
cell_types_sm_names.to_csv(f'{par["train_data_aug_dir"]}/cell_types_sm_names.csv', index=False)

print("Store Xs and y", flush=True)
X_vec = combine_features(
    [mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],
    [train_chem_feat, train_chem_feat_mean],
    de_train,
    one_hot_train
)
np.save(f'{par["train_data_aug_dir"]}/X_vec_initial.npy', X_vec)
X_vec_light = combine_features(
    [mean_cell_type, mean_sm_name],
    [train_chem_feat, train_chem_feat_mean],
    de_train,
    one_hot_train
)
np.save(f'{par["train_data_aug_dir"]}/X_vec_light.npy', X_vec_light)
X_vec_heavy = combine_features(
    [quantiles_cell_type, mean_cell_type, mean_sm_name],
    [train_chem_feat,train_chem_feat_mean],
    de_train,
    one_hot_train,
    quantiles_cell_type
)
np.save(f'{par["train_data_aug_dir"]}/X_vec_heavy.npy', X_vec_heavy)

ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
y = de_train.drop(columns=ylist)
np.save(f'{par["train_data_aug_dir"]}/y.npy', y.values)

print("Store config and shapes", flush=True)
config = {
    "LEARNING_RATES": [0.001, 0.001, 0.0003],
    "CLIP_VALUES": [5.0, 1.0, 1.0],
    "EPOCHS": par["epochs"],
    "KF_N_SPLITS": par["kf_n_splits"],
    "SCHEMES": par["schemes"],
    "MODELS": par["models"],
    "DATASET_ID": de_train_h5ad.uns["dataset_id"],
}
with open(f'{par["train_data_aug_dir"]}/config.json', 'w') as file:
    json.dump(config, file)

shapes = {
    "xshapes": {
        'initial': X_vec.shape,
        'light': X_vec_light.shape,
        'heavy': X_vec_heavy.shape
    },
    "yshape": y.shape
}
with open(f'{par["train_data_aug_dir"]}/shapes.json', 'w') as file:
    json.dump(shapes, file)

print("Store cross-validation indices", flush=True)
kf_cv = KF(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)

def get_kv_index(X, kf):
    return [
        (
            tr.astype(int).tolist(),
            va.astype(int).tolist()
        )
        for tr, va in kf.split(X)
    ]

kf_cv_initial = get_kv_index(X_vec, kf_cv)
json.dump(kf_cv_initial, open(f'{par["train_data_aug_dir"]}/kf_cv_initial.json', 'w'))

kf_cv_light =   get_kv_index(X_vec_light, kf_cv)
json.dump(kf_cv_light, open(f'{par["train_data_aug_dir"]}/kf_cv_light.json', 'w'))

kf_cv_heavy = get_kv_index(X_vec_heavy, kf_cv)
json.dump(kf_cv_heavy, open(f'{par["train_data_aug_dir"]}/kf_cv_heavy.json', 'w'))

print("### Done.")

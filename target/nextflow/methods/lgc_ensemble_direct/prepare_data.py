import os
import pandas as pd
import anndata as ad
from helper_functions import seed_everything, one_hot_encode, save_ChemBERTa_features
from anndata_to_dataframe import anndata_to_dataframe

def prepare_data(par, paths):
    seed_everything()
    ## Read data
    print("\nPreparing data...")
    de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
    de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])
    de_train = de_train.drop(columns=['split'])
    id_map = pd.read_csv(par["id_map"])
    ## Create data augmentation
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()
    std_sm_name = std_sm_name.fillna(0)
    cell_types = de_cell_type.groupby('cell_type').quantile(0.1).reset_index()['cell_type'] # This is just to get cell types in the right order for the next line
    quantiles_cell_type = pd.concat([pd.DataFrame(cell_types)]+[de_cell_type.groupby('cell_type')[col]\
    .quantile([0.25, 0.50, 0.75], interpolation='linear').unstack().reset_index(drop=True) for col in list(de_train.columns)[5:]], axis=1)
    ## Save data augmentation features
    print(paths["train_data_aug_dir"])
    if not os.path.exists(paths["train_data_aug_dir"]):
        os.makedirs(paths["train_data_aug_dir"], exist_ok=True)

    mean_cell_type.to_csv(f'{paths["train_data_aug_dir"]}/mean_cell_type.csv', index=False)
    std_cell_type.to_csv(f'{paths["train_data_aug_dir"]}/std_cell_type.csv', index=False)
    mean_sm_name.to_csv(f'{paths["train_data_aug_dir"]}/mean_sm_name.csv', index=False)
    std_sm_name.to_csv(f'{paths["train_data_aug_dir"]}/std_sm_name.csv', index=False)
    quantiles_cell_type.to_csv(f'{paths["train_data_aug_dir"]}/quantiles_cell_type.csv', index=False)
    ## Create one hot encoding features
    one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], out_dir=paths["train_data_aug_dir"])
    ## Prepare ChemBERTa features
    save_ChemBERTa_features(de_train["SMILES"].tolist(), out_dir=paths["train_data_aug_dir"], on_train_data=True)
    sm_name2smiles = {smname:smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
    test_smiles = list(map(sm_name2smiles.get, id_map['sm_name'].values))
    save_ChemBERTa_features(test_smiles, out_dir=paths["train_data_aug_dir"], on_train_data=False)
    print("### Done.")

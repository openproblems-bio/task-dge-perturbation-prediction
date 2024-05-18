import pandas as pd
import numpy as np
from helper_functions import combine_features, train_validate

def train(par, paths):
    train_config = {
        "LEARNING_RATES": [0.001, 0.001, 0.0003],
        "CLIP_VALUES": [5.0, 1.0, 1.0],
        "EPOCHS": par["epochs"],
        "KF_N_SPLITS": par["kf_n_splits"],
    }
    print("\nRead data and build features...")
    de_train = pd.read_parquet(par["de_train"])
    de_train = de_train.drop(columns=['split'])
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(f'{paths["train_data_aug_dir"]}/one_hot_train.npy'))
    y = de_train.drop(columns=ylist)
    mean_cell_type = pd.read_csv(f'{paths["train_data_aug_dir"]}/mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{paths["train_data_aug_dir"]}/std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{paths["train_data_aug_dir"]}/mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{paths["train_data_aug_dir"]}/std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{paths["train_data_aug_dir"]}/quantiles_cell_type.csv')
    train_chem_feat = np.load(f'{paths["train_data_aug_dir"]}/chemberta_train.npy')
    train_chem_feat_mean = np.load(f'{paths["train_data_aug_dir"]}/chemberta_train_mean.npy')
    X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [train_chem_feat,train_chem_feat_mean], de_train, one_hot_train, quantiles_df)
    ## Start training
    print(f"Mean cell type:{mean_cell_type.shape}")
    print(f"Std cell type:{std_cell_type.shape}")
    print(f"Mean_sm_name:{mean_sm_name.shape}")
    print(f"Std_sm_name:{std_sm_name.shape}")
    print(f"Quantiles_df:{quantiles_df.shape}")
    print(f"Train_chem_feat:{train_chem_feat.shape}")
    print(f"Train_chem_feat_mean:{train_chem_feat_mean.shape}")
    print(f"X_vec:{X_vec.shape}")
    print(f"X_vec_light:{X_vec_light.shape}")
    print(f"X_vec_heavy:{X_vec_heavy.shape}")
    print(f"de_train:{de_train.shape}")
    print(f"Y:{y.shape}")
    cell_types_sm_names = de_train[['cell_type', 'sm_name']]
    print("\nTraining starting...")
    train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, train_config, par, paths)
    print("\nDone.")
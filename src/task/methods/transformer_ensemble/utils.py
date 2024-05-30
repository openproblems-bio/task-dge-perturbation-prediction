import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import torch.optim
from sklearn.preprocessing import StandardScaler
import pickle
from models import *


def reduce_labels(Y, n_components):
    if n_components == Y.shape[1]:
        return None, None, Y
    label_reducer = TruncatedSVD(n_components=n_components, n_iter=10)
    scaler = StandardScaler()

    Y_scaled = scaler.fit_transform(Y)
    Y_reduced = label_reducer.fit_transform(Y_scaled)

    return label_reducer, scaler, Y_reduced


def prepare_augmented_data(
        de_train,
        id_map,
        uncommon=False
    ):
    de_train = de_train.drop(columns = ['split'])
    xlist = ['cell_type', 'sm_name']
    _ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    y = de_train.drop(columns=_ylist)

    # Combine train and test data for one-hot encoding
    combined_data = pd.concat([de_train[xlist], id_map[xlist]])

    dum_data = pd.get_dummies(combined_data, columns=xlist)

    # Split the combined data back into train and test
    train = dum_data.iloc[:len(de_train)]
    test = dum_data.iloc[len(de_train):]
    if uncommon:
        uncommon = [f for f in train if f not in test]
        X = train.drop(columns=uncommon)
    X = train
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]

    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index().fillna(0)

    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index().fillna(0)

    # Append mean and std for 'cell_type'
    rows = []
    for name in de_cell_type['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_cell_type = pd.concat(rows)
    tr_cell_type = tr_cell_type.reset_index(drop=True)

    # Append mean and std for 'sm_name'
    rows = []
    for name in de_sm_name['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_sm_name = pd.concat(rows)
    tr_sm_name = tr_sm_name.reset_index(drop=True)

    # Similar process for test data
    rows = []
    for name in id_map['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_cell_type = pd.concat(rows)
    te_cell_type = te_cell_type.reset_index(drop=True)

    rows = []
    for name in id_map['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_sm_name = pd.concat(rows)
    te_sm_name = te_sm_name.reset_index(drop=True)

    # Join mean and std features to X0, test0
    X0 = X.join(tr_cell_type.iloc[:, 1:]).copy()
    X0 = X0.join(tr_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    X0 = X0.select_dtypes(exclude='object')

    y0 = y.iloc[:, :].copy()

    test0 = test.join(te_cell_type.iloc[:, 1:]).copy()
    test0 = test0.join(te_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    test0 = test0.select_dtypes(exclude='object')
    return X0.astype(np.float32).to_numpy(), y0, test0.astype(np.float32).to_numpy()


def prepare_augmented_data_mean_only(
        de_train,
        id_map
    ):
    de_train = de_train.drop(columns = ['split'])
    xlist = ['cell_type', 'sm_name']
    _ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    y = de_train.drop(columns=_ylist)
    # train = pd.get_dummies(de_train[xlist], columns=xlist)
    # test = pd.get_dummies(id_map[xlist], columns=xlist)
    # Combine train and test data for one-hot encoding
    combined_data = pd.concat([de_train[xlist], id_map[xlist]])

    dum_data = pd.get_dummies(combined_data, columns=xlist)

    # Split the combined data back into train and test
    train = dum_data.iloc[:len(de_train)]
    test = dum_data.iloc[len(de_train):]
    # uncommon = [f for f in train if f not in test]
    # X = train.drop(columns=uncommon)

    X = train
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    rows = []
    for name in de_cell_type['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        rows.append(mean_rows)
    tr_cell_type = pd.concat(rows)
    tr_cell_type = tr_cell_type.reset_index(drop=True)

    rows = []
    for name in de_sm_name['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        rows.append(mean_rows)
    tr_sm_name = pd.concat(rows)
    tr_sm_name = tr_sm_name.reset_index(drop=True)

    rows = []
    for name in id_map['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        rows.append(mean_rows)

    te_cell_type = pd.concat(rows)
    te_cell_type = te_cell_type.reset_index(drop=True)
    rows = []
    for name in id_map['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        rows.append(mean_rows)

    te_sm_name = pd.concat(rows)
    te_sm_name = te_sm_name.reset_index(drop=True)
    X0 = X.join(tr_cell_type.iloc[:, 1:]).copy()
    X0 = X0.join(tr_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    y0 = y.iloc[:, :].copy()
    test0 = test.join(te_cell_type.iloc[:, 1:]).copy()
    test0 = test0.join(te_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    return X0.astype(np.float32).to_numpy(), y0, test0.astype(np.float32).to_numpy()



def split_data(train_features, targets, test_size=0.3, shuffle=False, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(train_features, targets, test_size=test_size,
                                                      shuffle=shuffle, random_state=random_state)

    return X_train, X_val, y_train.to_numpy(), y_val.to_numpy()


def calculate_mrrmse_np(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = np.sqrt(np.mean((outputs - labels.reshape(-1, outputs.shape[1])) ** 2, axis=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = np.mean(rmse_per_row)
    return mrmse


# Function to calculate MRRMSE
def calculate_mrrmse(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = torch.sqrt(torch.mean((outputs - labels) ** 2, dim=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = torch.mean(rmse_per_row)
    return mrmse

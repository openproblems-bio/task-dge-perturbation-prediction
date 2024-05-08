import torch
import torch.nn as nn
import torch.optim
 

## VIASH START
par = {
  "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
  "de_test": "resources/neurips-2023-kaggle/de_test.parquet",
  "id_map": "resources/neurips-2023-kaggle/id_map.csv",
  "output": "output.parquet",
}
## VIASH END

import sys
sys.path.append(meta['resources_dir'])

import sys

from utils import *
from sklearn.cluster import KMeans
import copy
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pickle
import argparse
from models import CustomTransformer_mean_std, CustomTransformer_mean  # Can be changed to other models in models.py
import os


def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, val_dataloader, criterion, label_reducer=None, scaler=None, device='cpu'):
    model.eval()
    val_loss = 0.0
    val_predictions_list = []
    val_targets_list = []
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_targets_list.append(val_targets.clone().cpu())
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_predictions = model(val_inputs)
            if label_reducer:
                val_targets = torch.tensor(
                    label_reducer.transform(scaler.transform(val_targets.clone().cpu().detach().numpy())),
                    dtype=torch.float32).to(device)
            val_loss += criterion(val_predictions, val_targets).item()
            val_predictions_list.append(val_predictions.cpu())

    val_loss /= len(val_dataloader)

    val_predictions_stacked = torch.cat(val_predictions_list, dim=0)
    val_targets_stacked = torch.cat(val_targets_list, dim=0)

    return val_loss, val_targets_stacked, val_predictions_stacked


def validate_sampling_strategy(sampling_strategy):
    allowed_strategies = ['k-means', 'random']
    if sampling_strategy not in allowed_strategies:
        raise ValueError(f"Invalid sampling strategy. Choose from: {', '.join(allowed_strategies)}")


def train_func(X_train, Y_reduced, X_val, Y_val, n_components, num_epochs, batch_size, label_reducer, scaler,
               d_model=128, early_stopping=5000, device='cpu', mean_std=
               'mean_std'):
    best_mrrmse = float('inf')
    best_model = None
    best_val_loss = float('inf')
    best_epoch = 0
    if mean_std == 'mean_std':
        model = CustomTransformer_mean_std(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(
            device)
    elif mean_std == 'mean':
        model = CustomTransformer_mean(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(
            device)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                            torch.tensor(Y_reduced, dtype=torch.float32).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                              torch.tensor(
                                                  Y_val,
                                                  dtype=torch.float32).to(device)),
                                batch_size=batch_size, shuffle=False)
    if n_components < 18211:
        lr = 1e-3

    else:
        lr = 1e-5
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7, verbose=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9999, patience=500,
                                               verbose=True)
    criterion = nn.HuberLoss()
    # criterion = nn.L1Loss()
    # criterion = CustomLoss()
    # criterion = nn.MSELoss()
    model.train()
    counter = 0
    pbar = tqdm(range(num_epochs), position=0, leave=True)
    for epoch in range(num_epochs):
        _ = train_epoch(model, dataloader, optimizer, criterion)

        if counter >= early_stopping:
            break
        if scaler:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion,
                                                                              label_reducer, scaler)
            # Calculate MRRMSE for the entire validation set
            val_mrrmse = calculate_mrrmse_np(
                val_targets_stacked.cpu().detach().numpy(),
                scaler.inverse_transform((label_reducer.inverse_transform(
                    val_predictions_stacked.cpu().detach().numpy()))))
        else:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion)
            val_mrrmse = calculate_mrrmse_np(val_targets_stacked.cpu().detach().numpy(),

                                             val_predictions_stacked.cpu().detach().numpy())

        if val_mrrmse < best_mrrmse:
            best_mrrmse = val_mrrmse
            # best_model = copy.deepcopy(model)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
            best_epoch = epoch
        else:
            counter += 1

        pbar.set_description(
            f"Validation best MRRMSE: {best_mrrmse:.4f} Validation best loss:"
            f" {best_val_loss:.4f} Last epoch: {best_epoch}")
        pbar.update(1)
        # scheduler.step()  # for cosine anealing
        scheduler.step(val_loss)
    return label_reducer, scaler, best_model


def train_transformer_k_means_learning(X, Y, n_components, num_epochs, batch_size,
                                       d_model=128, early_stopping=5000, device='cpu', seed=18, mean_std='mean_std'):
    label_reducer, scaler, Y_reduced = reduce_labels(Y, n_components)
    Y_reduced = Y_reduced.to_numpy()
    Y = Y.to_numpy()
    num_clusters = 2
    validation_percentage = 0.1

    # Create a K-Means clustering model
    kmeans = KMeans(n_clusters=num_clusters, n_init=100)

    # Fit the model to your regression targets (Y)
    clusters = kmeans.fit_predict(Y)

    # Initialize lists to store the training and validation data
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    # Iterate through each cluster
    for cluster_id in range(num_clusters):
        # Find the indices of data points in the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        print(len(cluster_indices))
        if len(cluster_indices) >= 20:
            # Split the data points in the cluster into training and validation
            train_indices, val_indices = train_test_split(cluster_indices, test_size=validation_percentage,
                                                          random_state=seed)

            # Append the corresponding data points to the training and validation sets
            X_train.extend(X[train_indices])
            Y_train.extend(Y_reduced[train_indices])  # Y_reduced for train Y for validation
            X_val.extend(X[val_indices])
            Y_val.extend(Y[val_indices])
        else:
            X_train.extend(X[cluster_indices])
            Y_train.extend(Y_reduced[cluster_indices])  # Y_reduced for train Y for validation
    # Convert the lists to numpy arrays if needed
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    transfromer_model = train_func(X_train, Y_train, X_val, Y_val, n_components, num_epochs, batch_size,
                                   label_reducer, scaler, d_model, early_stopping, device, mean_std)

    return label_reducer, scaler, transfromer_model

 
def train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                           early_stopping, batch_size, device, output_folder, mean_std):
    # Training loop for k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = train_transformer_k_means_learning(
                one_hot_encode_features,
                targets,
                n_components,
                num_epochs=num_epochs,
                early_stopping=early_stopping,
                batch_size=batch_size,
                d_model=d_model, device=device, mean_std=mean_std)
            os.makedirs(f'{output_folder}', exist_ok=True)
            # Save the trained models
            with open(f'{output_folder}/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'{output_folder}/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)
                
            torch.save(transformer_model[2].state_dict(),
                       f'{output_folder}/transformer_model_{n_components}_{d_model}.pt')


def train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device, seed, validation_percentage, output_folder,mean_std):
    # Split the data for non-k_means sampling strategy
    X_train, X_val, y_train, y_val = split_data(one_hot_encode_features, targets, test_size=validation_percentage,
                                                shuffle=True, random_state=seed)

    # Training loop for non-k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, Y_reduced = reduce_labels(y_train, n_components)
            transformer_model = train_func(X_train, y_train, X_val, y_val,
                                           n_components,
                                           num_epochs=num_epochs,
                                           early_stopping=early_stopping,
                                           batch_size=batch_size,
                                           d_model=d_model,
                                           label_reducer=label_reducer,
                                           scaler=scaler,
                                           device=device,mean_std=mean_std)

            # Save the trained models
            os.makedirs(f'{output_folder}', exist_ok=True)
            with open(f'{output_folder}/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'{output_folder}/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)

            torch.save(transformer_model[2].state_dict(),
                       f'{output_folder}/transformer_model_{n_components}_{d_model}.pt')


def main1(sampling_strategy = "k-means", output_folder = 'trained_models_kmeans_mean_std'):
    # # Set up command-line argument parser
    # parser = argparse.ArgumentParser(description="Your script description here.")
    # parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    # args = parser.parse_args()
    # print(args)
    # # Check if the config file is provided
    # if not args.config:
    #     print("Please provide a config file using --config.")
    #     return

    # # Load and print configurations
    # config_file = args.config
    # config = load_and_print_config(config_file)

    # Access specific values from the config
    # n_components_list = config.get('n_components_list', [])
    # d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    # batch_size = config.get('batch_size', 32)
    # sampling_strategy = config.get('sampling_strategy', 'random')
    # data_file = par['de_train']
    # id_map_file = par['id_map']
    # validation_percentage = config.get('validation_percentage', 0.2)
    # device = config.get('device', 'cpu')
    # seed = config.get('seed', None)
    # num_epochs = config.get('num_epochs', 20000)
    # early_stopping = config.get('early_stopping', 5000)

    n_components_list = [18211]
    d_models_list = [128]  # embedding dimensions for the transformer models
    batch_size = 32
    data_file = par['de_train']
    id_map_file = par['id_map']
    validation_percentage = 0.1
    device = 'cpu'
    seed = None
    num_epochs = 20_000
    early_stopping = 5000
    print('start training')

    # Validate the sampling strategy
    validate_sampling_strategy(sampling_strategy)

    # Prepare augmented data
    if 'std' in output_folder:
        uncommon=False
        if 'trueuncommon' in output_folder:
            uncommon=True
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(data_file=data_file,
                                                                                id_map_file=id_map_file, uncommon=uncommon)
        mean_std = 'mean_std'
    if 'std' not in output_folder:
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(data_file=data_file,
                                                                                id_map_file=id_map_file)
        mean_std = 'mean'
    print('oht shape', one_hot_test.shape)
    print('oht feature shape', one_hot_encode_features.shape)
    if sampling_strategy == 'k-means':
        train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device, output_folder, mean_std)
    else:
        train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                                   early_stopping, batch_size, device, seed, validation_percentage, output_folder, mean_std)
    print("Finish running stage 1!")


import copy

import argparse

@torch.no_grad()
def predict_test(data, models, n_components_list, d_list, batch_size, device='cpu', outname='traineddata'):
    num_samples = len(data)
    de_train = pd.read_parquet(par["de_train"])
    id_map = pd.read_csv(par["id_map"])
    gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]

    for i, n_components in enumerate(n_components_list):
        for j, d_model in enumerate(d_list):
            combined_outputs = []
            label_reducer, scaler, transformer_model = models[f'{n_components},{d_model}']
            transformer_model.eval()
            for i in range(0, num_samples, batch_size):
                batch_unseen_data = data[i:i + batch_size]
                transformed_data = transformer_model(batch_unseen_data)
                if scaler:
                    transformed_data = torch.tensor(scaler.inverse_transform(
                        label_reducer.inverse_transform(transformed_data.cpu().detach().numpy()))).to(device)
                # print(transformed_data.shape)
                combined_outputs.append(transformed_data)

            # Stack the combined outputs
            combined_outputs = torch.vstack(combined_outputs)

            submission_df = pd.DataFrame(
                    combined_outputs.cpu().detach().numpy(),
                    index=id_map["id"],
                    columns=gene_names
                    ).reset_index()
            # output_path = "resources/neurips-2023-data/" + f"result_{n_components}_{d_model}"
            # par[f"result_{n_components}_{d_model}"] = output_path
            # submission_df.to_parquet(par[f"result_{n_components}_{d_model}"])
            submission_df.to_csv(f"{outname}_output.csv")
            # submission_df.to_csv(f"result_{n_components}_{d_model}.csv", index=False)
            print('finish one')

    return


def main2( models_dir = 'trained_models_nonkmeans'):
    # Set up command-line argument parser
    # parser = argparse.ArgumentParser(description="Your script description here.")
    # parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    # args = parser.parse_args()

    # # Check if the config file is provided
    # if not args.config:
    #     print("Please provide a config file using --config.")
    #     return

    # # Load and print configurations
    # config_file = args.config
    # config = load_and_print_config(config_file)
    # # Access specific values from the config
    # n_components_list = config.get('n_components_list', [])
    # d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    # batch_size = config.get('batch_size', 32)
    # data_file =  par['de_test']
    # id_map_file =  par['id_map']
    # device = config.get('device', 'cpu')
    # models_dir = config.get('dir', 'model_1_mean_std_only')

    n_components_list = [18211]
    d_models_list = [128]  # embedding dimensions for the transformer models
    batch_size = 32
    data_file =  par["de_train"]
    id_map_file =  par["id_map"]
    device = 'cpu'


    # Prepare augmented data
    if 'std' in models_dir:
        uncommon=False
        if 'trueuncommon' in models_dir:
            uncommon=True
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(data_file=data_file,
                                                                                id_map_file=id_map_file, uncommon=uncommon)
        mean_std = 'mean_std'
    else:
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(data_file=data_file,
                                                                                          id_map_file=id_map_file)
        mean_std = 'mean'
    print('oht shape', one_hot_test.shape)
    print('oht feature shape', one_hot_encode_features.shape)
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(device)  # Replace X_unseen with your new data
    transformer_models = {}
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = load_transformer_model(n_components,
                                                                              input_features=
                                                                              one_hot_encode_features.shape[
                                                                                  1],
                                                                              d_model=d_model,
                                                                              models_folder=f'{models_dir}',
                                                                              device=device,mean_std=mean_std)
            transformer_model.eval()
            transformer_models[f'{n_components},{d_model}'] = (
                copy.deepcopy(label_reducer), copy.deepcopy(scaler), copy.deepcopy(transformer_model))
    predict_test(unseen_data, transformer_models, n_components_list, d_models_list, batch_size, device=device, outname = models_dir)
    print("Finish running stage 2!")


import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def normalize_weights(weights):
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def calculate_weighted_sum(dataframes, weights):
    weighted_dfs = [df * weight for df, weight in zip(dataframes, weights)]
    return sum(weighted_dfs)


def convert_to_consistent_dtype(df):
    df = df.astype(float)
    df['id'] = df['id'].astype(int)
    return df


def set_id_as_index(df):
    df.set_index('id', inplace=True)
    return df


def create_submission_df(weighted_sum):
    sample_submission = pd.read_csv(r"sample_submission.csv")
    sample_columns = sample_submission.columns[1:]
    submission_df = pd.DataFrame(weighted_sum.iloc[:, :].to_numpy(), columns=sample_columns)
    submission_df.insert(0, 'id', range(255))
    return submission_df


def save_submission_df(submission_df, file_path='weighted_submission.csv'):
    submission_df.to_csv(file_path, index=False)


# def main3():
#     # Load CSV DataFrames
#     print(sys.path)
#     print(meta['resources_dir'])
#     df1 = load_data(r"./result (15).csv")
#     df2 = load_data(r"./result (9).csv")
#     df3 = load_data(r"./result (11).csv")
#     df6 = load_data(r"./result (8).csv")  # amplifier

#     # Define weights for each DataFrame
#     weights = [0.4, 0.1, 0.2, 0.3]

#     # Normalize weights for df1, df2, and df3 to ensure their sum is 1
#     normalized_weights = normalize_weights(weights[:-1]) + [weights[-1]]

#     # Apply normalized weights to each DataFrame
#     weighted_sum = calculate_weighted_sum([df1, df2, df3, df6], normalized_weights)

#     # Convert all columns to a consistent data type (e.g., float)
#     weighted_sum = convert_to_consistent_dtype(weighted_sum)

#     # Set 'id' column as the index
#     weighted_sum = set_id_as_index(weighted_sum)

#     # Create and save the resulting weighted sum DataFrame
#     submission_df = create_submission_df(weighted_sum)
#     # save_submission_df(submission_df)
#     submission_df.to_parquet(par['output'])

def main3():
    df1 = load_data(r"./trained_models_kmeans_mean_std_output.csv")
    df2 = load_data(r"./trained_models_kmeans_mean_std_trueuncommon_output.csv")
    df3 = load_data(r"./trained_models_kmeans_mean_output.csv")
    df4 = load_data(r"./trained_models_nonkmeans_mean_output.csv")
    weights = [0.5, 0.25, 0.25, 0.3]
    df = weights[0] * df1 + weights[1] * df2 + weights[2] * df3 + weights[3] * df4
    df.to_parquet(par['output'])



use_pre = False
if use_pre == False:
    main1("k-means", "trained_models_kmeans_mean_std") #df1
    main2("trained_models_kmeans_mean_std") 
    main1("k-means", "trained_models_kmeans_mean_std_trueuncommon") #df2
    main2("trained_models_kmeans_mean_std_trueuncommon") 
    main1("k-means", "trained_models_kmeans_mean") #df3
    main2("trained_models_kmeans_mean")
    main1("random", "trained_models_nonkmeans_mean") #df4
    main2("trained_models_nonkmeans_mean")
    main3()
else:
    main3()
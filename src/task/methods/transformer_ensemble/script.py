import pandas as pd
import sys
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## VIASH START
par = {
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "de_test": "resources/neurips-2023-data/de_test.parquet",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "output": "output.parquet",
    "num_train_epochs": 10,
}
meta = {
    "resources_dir": "src/task/methods/transformer_ensemble",
}
## VIASH END

sys.path.append(meta["resources_dir"])

# Fixed training params
d_model = 128
batch_size = 32
early_stopping = 5000

from utils import prepare_augmented_data, prepare_augmented_data_mean_only
from train import train_k_means_strategy, train_non_k_means_strategy

# determine n_components_list

de_train = pd.read_parquet(par["de_train"])
id_map = pd.read_csv(par["id_map"])

gene_names = list(de_train.columns[6:])
n_components = len(gene_names)

# train and predict models
argsets = [
    {
        "name": "trained_models_kmeans_mean_std",
        "mean_std": "mean_std",
        "uncommon": False,
        "sampling_strategy": "k-means",
        "weight": 0.4,
    },
    {
        "name": "trained_models_kmeans_mean_std_trueuncommon",
        "mean_std": "mean_std",
        "uncommon": True,
        "sampling_strategy": "k-means",
        "weight": 0.1,
    },
    {
        "name": "trained_models_kmeans_mean",
        "mean_std": "mean",
        "uncommon": False,
        "sampling_strategy": "k-means",
        "weight": 0.2,
    },
    {
        "name": "trained_models_nonkmeans_mean",
        "mean_std": "mean",
        "uncommon": False,
        "sampling_strategy": "random",
        "weight": 0.3,
    },
]


predictions = []

print(f"Train and predict models", flush=True)
for argset in argsets:
    print(f"Train and predict model {argset['name']}", flush=True)

    print(f"> Prepare augmented data", flush=True)
    if argset["mean_std"] == "mean_std":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(
            de_train=copy.deepcopy(de_train),
            id_map=copy.deepcopy(id_map),
            uncommon=argset["uncommon"],
        )
    elif argset["mean_std"] == "mean":
        one_hot_encode_features, targets, one_hot_test = (
            prepare_augmented_data_mean_only(de_train=de_train, id_map=id_map)
        )

    print(f"> Train model", flush=True)
    if argset["sampling_strategy"] == "k-means":
        label_reducer, scaler, transformer_model = train_k_means_strategy(
            n_components=n_components,
            d_model=d_model,
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=early_stopping,
            batch_size=batch_size,
            device=device,
            mean_std=argset["mean_std"],
        )
    else:
        label_reducer, scaler, transformer_model = train_non_k_means_strategy(
            n_components=n_components,
            d_model=d_model,
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=early_stopping,
            batch_size=batch_size,
            device=device,
            mean_std=argset["mean_std"],
        )

    print(f"> Predict model", flush=True)
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(
        device
    )  # Replace X_unseen with your new data

    num_features = one_hot_encode_features.shape[1]
    num_targets = targets.shape[1]

    if n_components == num_features:
        label_reducer = None
        scaler = None

    print(f"Predict on test data", flush=True)
    num_samples = len(unseen_data)
    transformed_data = []
    for i in range(0, num_samples, batch_size):
        batch_unseen_data = unseen_data[i : i + batch_size]
        batch_result = transformer_model(batch_unseen_data)
        transformed_data.append(batch_result)
    transformed_data = torch.vstack(transformed_data)
    if scaler:
        transformed_data = torch.tensor(
            scaler.inverse_transform(
                label_reducer.inverse_transform(transformed_data.cpu().detach().numpy())
            )
        ).to(device)

    pred = transformed_data.cpu().detach().numpy()
    predictions.append(pred)

print(f"Combine predictions", flush=True)
weighted_pred = sum(
    [argset["weight"] * pred for argset, pred in zip(argsets, predictions)]
) / sum([argset["weight"] for argset in argsets])

df = pd.DataFrame(weighted_pred, columns=gene_names)
df.reset_index(drop=True, inplace=True)
df.reset_index(names="id", inplace=True)

df.to_parquet(par["output"])

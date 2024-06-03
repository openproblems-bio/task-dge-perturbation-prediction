import pandas as pd
import anndata as ad
import sys
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## VIASH START
par = {
    "de_train_h5ad": "resources/neurips-2023-data/de_train.h5ad",
    "de_train": "resources/neurips-2023-data/de_train.h5ad",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "output": "output.h5ad",
    "num_train_epochs": 10,
    "layer": "clipped_sign_log10_pval"
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

from anndata_to_dataframe import anndata_to_dataframe
from utils import prepare_augmented_data, prepare_augmented_data_mean_only
from train import train_k_means_strategy, train_non_k_means_strategy

# read data
de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])
id_map = pd.read_csv(par["id_map"])

# determine other variables
gene_names = list(de_train_h5ad.var_names)
n_components = len(gene_names)

# train and predict models
# note, the weights intentionally don't add up to one
argsets = [
    # Note by author - weight_df1: 0.5 (utilizing std, mean, and clustering sampling, yielding 0.551)
    {
        "mean_std": "mean_std",
        "uncommon": False,
        "sampling_strategy": "random",
        "weight": 0.5,
    },
    # Note by author - weight_df2: 0.25 (excluding uncommon elements, resulting in 0.559)
    {
        "mean_std": "mean_std",
        "uncommon": True,
        "sampling_strategy": "random",
        "weight": 0.25,
    },
    # Note by author - weight_df3: 0.25 (leveraging clustering sampling, achieving 0.575)
    {
        "mean_std": "mean_std",
        "uncommon": False, # should this be set to False or True?
        "sampling_strategy": "k-means",
        "weight": 0.25,
    },
    # Note by author - weight_df4: 0.3 (incorporating mean, random sampling, and excluding std, attaining 0.554)
    {
        "mean_std": "mean",
        "uncommon": False,
        "sampling_strategy": "random",
        "weight": 0.3,
    }
]


predictions = []

print(f"Train and predict models", flush=True)
for i, argset in enumerate(argsets):
    print(f"Train and predict model {i}/{len(argsets)}", flush=True)

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
    else:
        raise ValueError("Invalid mean_std argument")

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
    elif argset["sampling_strategy"] == "random":
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
    else:
        raise ValueError("Invalid sampling_strategy argument")

    print(f"> Predict model", flush=True)
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(device)

    num_features = one_hot_encode_features.shape[1]
    num_targets = targets.shape[1]

    if n_components == num_features:
        label_reducer = None
        scaler = None

    print(f"Predict on test data", flush=True)
    num_samples = len(unseen_data)
    transformed_data = []
    for i in range(0, num_samples, batch_size):
        batch_result = transformer_model(unseen_data[i : i + batch_size])
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
# compute weighted sum
# note, the weights intentionally don't add up to one
weighted_pred = sum([
    pred * argset["weight"]
    for argset, pred in zip(argsets, predictions)
])


print('Write output to file', flush=True)
output = ad.AnnData(
    layers={"prediction": weighted_pred},
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
      "dataset_id": de_train_h5ad.uns["dataset_id"],
      "method_id": meta["functionality_name"]
    }
)

output.write_h5ad(par["output"], compression="gzip")

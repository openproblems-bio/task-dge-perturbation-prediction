import pandas as pd
import anndata as ad
import sys
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## VIASH START
par = {
    "de_train_h5ad": "resources/neurips-2023-kaggle/de_train.h5ad",
    "id_map": "resources/neurips-2023-kaggle/id_map.csv",
    "output": "output/prediction.h5ad",
    "output_model": "output/model/",
    "num_train_epochs": 10,
    "early_stopping": 5000,
    "batch_size": 64,
    "d_model": 128,
    "layer": "sign_log10_pval"
}
meta = {
    "resources_dir": "src/task/methods/transformer_ensemble",
}
## VIASH END

sys.path.append(meta["resources_dir"])

from utils import prepare_augmented_data, prepare_augmented_data_mean_only
from train import train_k_means_strategy, train_non_k_means_strategy

# create output model directory if need be
if par["output_model"]:
    os.makedirs(par["output_model"], exist_ok=True)

# read data
de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
id_map = pd.read_csv(par["id_map"])

# convert .obs categoricals to string for ease of use
for col in de_train_h5ad.obs.select_dtypes(include=["category"]).columns:
    de_train_h5ad.obs[col] = de_train_h5ad.obs[col].astype(str)
# reset index
de_train_h5ad.obs.reset_index(drop=True, inplace=True)

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
        "validation_percentage": 0.1,
    },
    # Note by author - weight_df2: 0.25 (excluding uncommon elements, resulting in 0.559)
    {
        "mean_std": "mean_std",
        "uncommon": True,
        "sampling_strategy": "random",
        "weight": 0.25,
        "validation_percentage": 0.1,

    },
    # Note by author - weight_df3: 0.25 (leveraging clustering sampling, achieving 0.575)
    {
        "mean_std": "mean_std",
        "uncommon": False,
        "sampling_strategy": "k-means",
        "weight": 0.25,
    },
    # Note by author - weight_df4: 0.3 (incorporating mean, random sampling, and excluding std, attaining 0.554)
    {
        "mean_std": "mean",
        "uncommon": False,
        "sampling_strategy": "random",
        "weight": 0.3,
        "validation_percentage": 0.1,
    }
]


predictions = []

print(f"Train and predict models", flush=True)
for i, argset in enumerate(argsets):
    print(f"Train and predict model {i+1}/{len(argsets)}", flush=True)

    print(f"> Prepare augmented data", flush=True)
    if argset["mean_std"] == "mean_std":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(
            de_train_h5ad=de_train_h5ad,
            id_map=id_map,
            layer=par["layer"],
            uncommon=argset["uncommon"],
        )
    elif argset["mean_std"] == "mean":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(
            de_train_h5ad=de_train_h5ad,
            id_map=id_map,
            layer=par["layer"],
        )
    else:
        raise ValueError("Invalid mean_std argument")

    print(f"> Train model", flush=True)
    if argset["sampling_strategy"] == "k-means":
        label_reducer, scaler, transformer_model = train_k_means_strategy(
            n_components=n_components,
            d_model=par["d_model"],
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=par["early_stopping"],
            batch_size=par["batch_size"],
            mean_std=argset["mean_std"],
            device=device,
        )
    elif argset["sampling_strategy"] == "random":
        label_reducer, scaler, transformer_model = train_non_k_means_strategy(
            n_components=n_components,
            d_model=par["d_model"],
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=par["early_stopping"],
            batch_size=par["batch_size"],
            mean_std=argset["mean_std"],
            validation_percentage=argset["validation_percentage"],
            device=device,
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
    for i in range(0, num_samples, par["batch_size"]):
        batch_result = transformer_model(unseen_data[i : i + par["batch_size"]])
        transformed_data.append(batch_result)
    transformed_data = torch.vstack(transformed_data)
    if scaler:
        transformed_data = torch.tensor(
            scaler.inverse_transform(
                label_reducer.inverse_transform(transformed_data.cpu().detach().numpy())
            )
        ).to(device)

    pred = transformed_data.cpu().detach().numpy()

    if par["output_model"]:
        model_path = f"{par['output_model']}/model_{i}.pt"
        torch.save(transformer_model.state_dict(), model_path)
        pred_path = f"{par['output_model']}/pred_{i}.csv"
        pd.DataFrame(pred).to_csv(pred_path, index=False)

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

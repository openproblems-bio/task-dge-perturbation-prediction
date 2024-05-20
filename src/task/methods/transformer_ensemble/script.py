import os
import sys
import tempfile
import shutil

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "output.parquet",
}
meta = {
    "resources_dir": "src/task/methods/lb2",
}
## VIASH END

sys.path.append(meta['resources_dir'])

from train import train_main
from predict import predict_main
from seq import seq_main

# determine n_components_list
import pandas as pd
de_train = pd.read_parquet(par["de_train"])
de_train.drop(columns=["cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control"], inplace=True)
n_components_list = [de_train.shape[1]]
del de_train

# determine model dirs
output_model = par.get("output_model") or tempfile.TemporaryDirectory(dir = meta["temp_dir"]).name
if not os.path.exists(output_model):
    os.makedirs(output_model, exist_ok=True)
if not par.get("output_model"):
  import atexit
  atexit.register(lambda: shutil.rmtree(output_model))

# train and predict models
argsets = [
  {
    "dir": f"{output_model}/trained_models_kmeans_mean_std",
    "mean_std": "mean_std",
    "uncommon": False,
    "sampling_strategy": "k-means",
    "weight": .4
  },
  {
    "dir": f"{output_model}/trained_models_kmeans_mean_std_trueuncommon",
    "mean_std": "mean_std",
    "uncommon": True,
    "sampling_strategy": "k-means",
    "weight": .1
  },
  {
    "dir": f"{output_model}/trained_models_kmeans_mean",
    "mean_std": "mean",
    "uncommon": False,
    "sampling_strategy": "k-means",
    "weight": .2
  },
  {
    "dir": f"{output_model}/trained_models_nonkmeans_mean",
    "mean_std": "mean",
    "uncommon": False,
    "sampling_strategy": "random",
    "weight": .3
  }
]

print(f"Train and predict models", flush=True)
for argset in argsets:
  print(f"Generate model {argset['dir']}", flush=True)
  train_main(par, n_components_list, argset['dir'], 
             mean_std=argset['mean_std'], uncommon=argset['uncommon'],
             sampling_strategy=argset['sampling_strategy'], device=device)

  print(f"Predict model {argset['dir']}", flush=True)
  predict_main(par, n_components_list, argset['dir'], mean_std=argset['mean_std'],
               uncommon=argset['uncommon'], device=device)

print(f"Combine predictions", flush=True)
seq_main(
  par,
  model_dirs=[argset['dir'] for argset in argsets],
  weights=[argset['weight'] for argset in argsets],
)
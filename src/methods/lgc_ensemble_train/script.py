import sys
import torch
import json
import numpy as np
import pandas as pd
if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print('using device: cpu', flush=True)

## VIASH START
par = {
    "train_data_aug_dir": "output/train_data_aug_dir",
    "scheme": "initial",
    "model": "LSTM",
    "fold": 0,
    "model_file": "output/model.pt",
    "log_file": "output/log.json",
}
meta = {
    "resources_dir": "src/methods/lgc_ensemble",
    "temp_dir": "/tmp"
}
## VIASH END

print(f"par: {par}", flush=True)

# import helper functions
sys.path.append(meta['resources_dir'])

from models import Conv, LSTM, GRU
from helper_functions import train_function

###################################################################
# Interpretation from src/methods/lgc_ensemble/helper_functions.py

print("Load data...", flush=True)
# read kf_cv_initial from json
kn_cv_path = f'{par["train_data_aug_dir"]}/kf_cv_{par["scheme"]}.json'
with open(kn_cv_path, 'r') as file:
    kf_cv = json.load(file)

train_idx, val_idx = kf_cv[par["fold"]]

X = np.load(f'{par["train_data_aug_dir"]}/X_vec_{par["scheme"]}.npy')
y = np.load(f'{par["train_data_aug_dir"]}/y.npy')

cell_types_sm_names = pd.read_csv(f'{par["train_data_aug_dir"]}/cell_types_sm_names.csv')

with open(f'{par["train_data_aug_dir"]}/config.json', 'r') as file:
    config = json.load(file)

print("Prepare data...", flush=True)
x_train, x_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
info_data = {
    'train_cell_type': cell_types_sm_names.iloc[train_idx]['cell_type'].tolist(),
    'val_cell_type': cell_types_sm_names.iloc[val_idx]['cell_type'].tolist(),
    'train_sm_name': cell_types_sm_names.iloc[train_idx]['sm_name'].tolist(),
    'val_sm_name': cell_types_sm_names.iloc[val_idx]['sm_name'].tolist()
}

models = {
    "LSTM": LSTM,
    "GRU": GRU,
    "Conv": Conv
}
schemes = ['initial', 'light', 'heavy']
scheme_idx = schemes.index(par["scheme"])
clip_norm = config["CLIP_VALUES"][scheme_idx]

ModelClass = models[par["model"]]
model = ModelClass(par["scheme"], X.shape, y.shape)

print("Start training...", flush=True)
model, results = train_function(
    model,
    model.name,
    x_train,
    y_train,
    x_val,
    y_val,
    info_data,
    config=config,
    clip_norm=clip_norm
)
model.to('cpu')

print("Save model...", flush=True)
torch.save(model.state_dict(), par["model_file"])
with open(par["log_file"], 'w') as file:
    json.dump(results, file)

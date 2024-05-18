import sys
import tempfile
import shutil

## VIASH START
par = {
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "de_test": "resources/neurips-2023-data/de_test.parquet",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "models": ["initial", "light"],
    "epochs": 1,
    "kf_n_splits": 2,
    "output": "output.parquet",
    "output_model": None
}
meta = {
    "resources_dir": "src/task/methods/lstm_gru_cnn_ensemble",
    "temp_dir": "/tmp"
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])

from prepare_data import prepare_data
from train import train
from predict import predict

# create a temporary directory for storing models
output_model = par["output_model"] or tempfile.TemporaryDirectory(dir = meta["temp_dir"]).name
paths = {
    "output": par["output"],
    "output_model": output_model,
    "train_data_aug_dir": f"{output_model}/train_data_aug_dir",
    "model_dir": f"{output_model}/model_dir",
    "logs_dir": f"{output_model}/logs"
}

# remove temp dir on exit
if not par["output_model"]:
	import atexit
	atexit.register(lambda: shutil.rmtree(output_model))

# prepare data
prepare_data(par, paths)

# train
train(par, paths)

# predict
predict(par, paths)

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
    "extra_output": None
}
meta = {
    "resources_dir": "src/task/methods/first_place",
    "temp_dir": "/tmp"
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])

from prepare_data import prepare_data
from train import train
from predict import predict

# create a temporary directory for storing models
extra_output = par["extra_output"] or tempfile.TemporaryDirectory(dir = meta["temp_dir"]).name
paths = {
    "output": par["output"],
    "extra_output": extra_output,
    "train_data_aug_dir": f"{extra_output}/train_data_aug_dir",
    "model_dir": f"{extra_output}/model_dir",
    "logs_dir": f"{extra_output}/logs"
}

# remove temp dir on exit
if not par["extra_output"]:
	import atexit
	atexit.register(lambda: shutil.rmtree(extra_output))

# prepare data
prepare_data(par, paths)

# train
train(par, paths)

# predict
predict(par, paths)

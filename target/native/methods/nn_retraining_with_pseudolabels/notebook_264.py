# Change history
# * Imported code from https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206
# * Derive gene_names from train_df to to avoid reading in the sample_submission.csv file
# * Make reps component arguments
# * Auto-reformatted the code
# * Restructured the code into a function `run_notebook_264()`

# -----------------------------------------------------------------------------
# Load dependencies
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Embedding,
    Flatten,
)
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers.legacy import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD


# -----------------------------------------------------------------------------
# Define helper functions and parameters
# -----------------------------------------------------------------------------
def custom_mean_rowwise_rmse(y_true, y_pred):
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def model_1(lr, emb_out, n_dim):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(256),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.2),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_2(lr, emb_out, dense_1, dense_2, dropout_1, dropout_2, n_dim):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),  # 256 - 2048
            Dense(dense_2, activation="relu"),
            Activation("relu"),
            BatchNormalization(),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_3(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 256
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(dense_4),  # 16 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_4),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_4(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_5(lr, emb_out, n_dim, dropout_1, dropout_2):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(256),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(1024),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_6(lr, emb_out, dense_1, dense_2, n_dim, dropout_1, dropout_2):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            BatchNormalization(),
            Dense(dense_1),  # 64 - 512
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_2),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_7(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 256
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(dense_4),  # 16 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_4),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_8(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def load_models():
    models = [
        model_1,
        model_2,
        model_3,
        model_5,
        model_6,
        model_7,
        model_8,
    ]
    return models


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------
def load_params():
    params_model_1 = {
        "params": {
            "epochs": 114,
            "bs": 128,
            "lr": 0.008457844054540857,
            "emb_out": 22,
            "n_dim": 50,
        },
        "value": 0.9060678655727635,
    }

    params_model_2 = {
        "params": {
            "epochs": 136,
            "bs": 64,
            "lr": 0.007787474024659863,
            "emb_out": 10,
            "dense_1": 384,
            "dense_2": 1280,
            "dropout_1": 0.4643149193312417,
            "dropout_2": 0.10101884612160547,
            "n_dim": 60,
        },
        "value": 0.9070240468092804,
    }

    params_model_3 = {
        "params": {
            "epochs": 157,
            "bs": 64,
            "lr": 0.004311857150745656,
            "emb_out": 62,
            "dense_1": 560,
            "dense_2": 480,
            "dense_3": 248,
            "dense_4": 224,
            "dropout_1": 0.4359908049836846,
            "dropout_2": 0.34432694543970555,
            "dropout_3": 0.01112409967333259,
            "dropout_4": 0.23133616975077548,
            "n_dim": 119,
        },
        "value": 0.9171315640806535,
    }

    params_model_4 = {
        "params": {
            "epochs": 147,
            "bs": 64,
            "lr": 0.005948541271442179,
            "emb_out": 46,
            "dense_1": 872,
            "dense_2": 264,
            "dense_3": 256,
            "dropout_1": 0.17543603718794346,
            "dropout_2": 0.3587657616370447,
            "dropout_3": 0.12077512068514727,
            "n_dim": 213,
        },
        "value": 0.9228638968500431,
    }

    params_model_5 = {
        "params": {
            "epochs": 122,
            "bs": 32,
            "lr": 0.004429076555977599,
            "emb_out": 32,
            "n_dim": 71,
            "dropout_1": 0.40604535344002984,
            "dropout_2": 0.178189970426619,
        },
        "value": 0.9083640103276015,
    }

    params_model_6 = {
        "params": {
            "epochs": 112,
            "bs": 128,
            "lr": 0.009773732221901085,
            "emb_out": 60,
            "dense_1": 436,
            "dense_2": 416,
            "n_dim": 126,
            "dropout_1": 0.4024659444883379,
            "dropout_2": 0.2573940194596736,
        },
        "value": 0.8909352668212382,
    }

    params_model_7 = {
        "params": {
            "epochs": 141,
            "bs": 128,
            "lr": 0.005530331519967936,
            "emb_out": 48,
            "dense_1": 712,
            "dense_2": 400,
            "dense_3": 232,
            "dense_4": 216,
            "dropout_1": 0.4903998136177629,
            "dropout_2": 0.032371643764537134,
            "dropout_3": 0.11138300987168903,
            "dropout_4": 0.019885384663655765,
            "n_dim": 100,
        },
        "value": 0.8978272722102707,
    }

    params_model_8 = {
        "params": {
            "epochs": 143,
            "bs": 192,
            "lr": 0.00971858172843266,
            "emb_out": 48,
            "dense_1": 312,
            "dense_2": 344,
            "dense_3": 248,
            "dropout_1": 0.10974777738609129,
            "dropout_2": 0.10106027333885811,
            "dropout_3": 0.09775833250663657,
            "n_dim": 100,
        },
        "value": 0.8885448573595669,
    }

    params = [
        params_model_1,
        params_model_2,
        params_model_3,
        params_model_5,
        params_model_6,
        params_model_7,
        params_model_8,
    ]
    return params


# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
def load_weights():
    w1 = [
        0.15224443321212433,
        0.7152220796128623,
        0.7547606691460997,
        0.05786285275052854,
        0.9602177109190158,
        0.4968056740470425,
        0.9881673272809887,
    ]
    return w1


# -----------------------------------------------------------------------------
# Predict functions
# -----------------------------------------------------------------------------
def split_params_to_training_model(model_params):
    model_params = model_params["params"]
    training_keys = ["epochs", "bs"]
    training_params = {k: model_params[k] for k in training_keys}
    model_params = {
        k: model_params[k] for k in model_params.keys() if k not in training_keys
    }
    return model_params, training_params


def fit_and_predict_embedding_nn(x, y, test_x, model_constructor, best_params):
    model_params, training_params = split_params_to_training_model(best_params)
    n_dim = model_params["n_dim"]
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    model.fit(
        x,
        y,
        epochs=training_params["epochs"],
        batch_size=training_params["bs"],
        verbose=0,
        shuffle=True,
    )
    return d.inverse_transform(model.predict(test_x, batch_size=1))


def predict(test_df, models, params, weights, le, new_names, original_y, reps):
    x_test = le.transform(test_df[["cell_type", "sm_name"]].values.flat).reshape(-1, 2)

    preds = []
    for model_i in range(len(models)):
        model = models[model_i]
        param = params[model_i]
        temp_pred = []
        for rep_i in range(reps):
            print(
                f"NB264, Training model {model_i + 1}/{len(models)}, Repeat {rep_i + 1}/{reps}",
                flush=True,
            )
            temp_pred.append(
                fit_and_predict_embedding_nn(
                    new_names, original_y, x_test, model, param
                )
            )
        temp_pred = np.median(temp_pred, axis=0)
        preds.append(temp_pred)

    pred = np.sum([w * p for w, p in zip(weights, preds)], axis=0) / sum(weights)
    return pred


def run_notebook_264(train_df, test_df, gene_names, reps):
    # determine mins and maxs for later clipping
    original_y = train_df.loc[:, gene_names].values
    mins = original_y.min(axis=0)
    maxs = original_y.max(axis=0)

    # determine label encoder
    original_x = train_df[["cell_type", "sm_name"]].values
    le = LabelEncoder()
    le.fit(original_x.flat)
    new_names = le.transform(original_x.flat).reshape(-1, 2)

    # load models, params, and weights
    models = load_models()
    params = load_params()
    weights = load_weights()

    # generate predictions
    pred = predict(test_df, models, params, weights, le, new_names, original_y, reps)

    # clip predictions
    clipped_pred = np.clip(pred, mins, maxs)

    # format outputs
    df = pd.DataFrame(clipped_pred, columns=gene_names)
    df["id"] = range(len(df))
    df = df.loc[:, ["id"] + gene_names]

    return df

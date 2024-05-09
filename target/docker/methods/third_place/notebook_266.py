# Change history
# * Imported code from https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153141755
# * Derive gene_names from train_df to to avoid reading in the sample_submission.csv file
# * Make reps component arguments
# * Auto-reformatted the code
# * Restructured the code into a function `run_notebook_266()`

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
# Helper functions
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
    models = (
        [model_1]
        + [model_2] * 3
        + [model_3] * 2
        + [model_4] * 2
        + [model_5] * 3
        + [model_6] * 3
        + [model_7] * 2
        + [model_8] * 4
    )
    return models


# -----------------------------------------------------------------------------
# Parameter sets
# -----------------------------------------------------------------------------
def load_params():
    params_model_1a = {
        "params": {
            "epochs": 200,
            "bs": 128,
            "lr": 0.008457844054540857,
            "emb_out": 22,
            "n_dim": 50,
        },
        "value": 0.9060678655727635,
    }

    ####

    params_model_2a = {
        "params": {
            "epochs": 200,
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

    params_model_2b = {
        "params": {
            "epochs": 200,
            "bs": 192,
            "lr": 0.00830447680398929,
            "emb_out": 32,
            "dense_1": 284,
            "dense_2": 1424,
            "dropout_1": 0.27860934847913565,
            "dropout_2": 0.04217965884576308,
            "n_dim": 96,
        },
        "value": 1.0158228546558994,
    }

    params_model_2c = {
        "params": {
            "epochs": 200,
            "bs": 64,
            "lr": 0.006661059864181284,
            "emb_out": 22,
            "dense_1": 232,
            "dense_2": 1184,
            "dropout_1": 0.46230673331531297,
            "dropout_2": 0.24430331733550426,
            "n_dim": 61,
        },
        "value": 0.5378740563213127,
    }

    #####

    params_model_3a = {
        "params": {
            "epochs": 200,
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

    params_model_3b = {
        "params": {
            "epochs": 200,
            "bs": 64,
            "lr": 0.007915642160705914,
            "emb_out": 22,
            "dense_1": 504,
            "dense_2": 136,
            "dense_3": 232,
            "dense_4": 512,
            "dropout_1": 0.0072011388198520605,
            "dropout_2": 0.07781770809801486,
            "dropout_3": 0.3482776196327668,
            "dropout_4": 0.4010684312497648,
            "n_dim": 55,
        },
        "value": 1.0557613871962215,
    }

    ####

    params_model_4a = {
        "params": {
            "epochs": 200,
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

    params_model_4b = {
        "params": {
            "epochs": 200,
            "bs": 128,
            "lr": 0.006444109866334638,
            "emb_out": 62,
            "dense_1": 552,
            "dense_2": 480,
            "dense_3": 216,
            "dropout_1": 0.323390730123547,
            "dropout_2": 0.15142047240687942,
            "dropout_3": 0.034625791669279364,
            "n_dim": 104,
        },
        "value": 0.8462075069648056,
    }

    #####

    params_model_5a = {
        "params": {
            "epochs": 200,
            "bs": 32,
            "lr": 0.005430251128204367,
            "emb_out": 56,
            "n_dim": 108,
            "dropout_1": 0.02868537022302934,
            "dropout_2": 0.35808251111776157,
        },
        "value": 0.835768615578779,
    }

    params_model_5b = {
        "params": {
            "epochs": 200,
            "bs": 192,
            "lr": 0.00528860262509972,
            "emb_out": 60,
            "n_dim": 78,
            "dropout_1": 0.2978319229273037,
            "dropout_2": 0.3236224036130246,
        },
        "value": 1.0125358317737336,
    }

    params_model_5c = {
        "params": {
            "epochs": 200,
            "bs": 32,
            "lr": 0.004429076555977599,
            "emb_out": 32,
            "n_dim": 71,
            "dropout_1": 0.40604535344002984,
            "dropout_2": 0.178189970426619,
        },
        "value": 0.9083640103276015,
    }

    ####

    params_model_6a = {
        "params": {
            "epochs": 200,
            "bs": 128,
            "lr": 0.0030468340279031702,
            "emb_out": 62,
            "dense_1": 396,
            "dense_2": 912,
            "n_dim": 144,
            "dropout_1": 0.2643057707162437,
            "dropout_2": 0.1738090239074675,
        },
        "value": 0.8388383786625531,
    }

    params_model_6b = {
        "params": {
            "epochs": 200,
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

    params_model_6c = {
        "params": {
            "epochs": 200,
            "bs": 160,
            "lr": 0.005742157072582258,
            "emb_out": 56,
            "dense_1": 504,
            "dense_2": 928,
            "n_dim": 134,
            "dropout_1": 0.26460638891781607,
            "dropout_2": 0.243272371789527,
        },
        "value": 0.9921304350469378,
    }

    ####

    params_model_7a = {
        "params": {
            "epochs": 200,
            "bs": 128,
            "lr": 0.0026256302897014814,
            "emb_out": 62,
            "dense_1": 824,
            "dense_2": 184,
            "dense_3": 208,
            "dense_4": 472,
            "dropout_1": 0.04406850232282358,
            "dropout_2": 0.051203939042409885,
            "dropout_3": 0.05926676325711479,
            "dropout_4": 0.08819762697219703,
            "n_dim": 167,
        },
        "value": 0.8347070421058967,
    }

    params_model_7b = {
        "params": {
            "epochs": 200,
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

    ####

    params_model_8a = {
        "params": {
            "epochs": 200,
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

    params_model_8b = {
        "params": {
            "epochs": 200,
            "bs": 192,
            "lr": 0.008078165473745607,
            "emb_out": 16,
            "dense_1": 1016,
            "dense_2": 392,
            "dense_3": 176,
            "dropout_1": 0.21410737149365255,
            "dropout_2": 0.40541433561062473,
            "dropout_3": 0.10476819447155189,
            "n_dim": 72,
        },
        "value": 0.767725144592772,
    }

    params_model_8c = {
        "params": {
            "epochs": 200,
            "bs": 160,
            "lr": 0.005427125417330768,
            "emb_out": 36,
            "dense_1": 760,
            "dense_2": 416,
            "dense_3": 240,
            "dropout_1": 0.16485069317527304,
            "dropout_2": 0.014216669745902685,
            "dropout_3": 0.05820818430142793,
            "n_dim": 128,
        },
        "value": 0.9778518605441292,
    }

    params_model_8d = {
        "params": {
            "epochs": 200,
            "bs": 224,
            "lr": 0.0077454113093514835,
            "emb_out": 38,
            "dense_1": 856,
            "dense_2": 352,
            "dense_3": 112,
            "dropout_1": 0.058963634508929205,
            "dropout_2": 0.10928657766717247,
            "dropout_3": 0.06218368685386452,
            "n_dim": 246,
        },
        "value": 0.837621399656469,
    }

    params = [
        params_model_1a,
        params_model_2a,
        params_model_2b,
        params_model_2c,
        params_model_3a,
        params_model_3b,
        params_model_4a,
        params_model_4b,
        params_model_5a,
        params_model_5b,
        params_model_5c,
        params_model_6a,
        params_model_6b,
        params_model_6c,
        params_model_7a,
        params_model_7b,
        params_model_8a,
        params_model_8b,
        params_model_8c,
        params_model_8d,
    ]

    return params


# -----------------------------------------------------------------------------
# Model weights
# -----------------------------------------------------------------------------


def load_weights():
    weights = {
        "w_0": 0.039503611057797205,
        "w_1": 0.22778329445024798,
        "w_2": 0.6470451053292054,
        "w_3": 0.14998376919294348,
        "w_4": 0.2417986187654036,
        "w_5": 0.018231625997453538,
        "w_6": 0.0670756518443389,
        "w_7": 0.45476025839087764,
        "w_8": 0.8953651930541969,
        "w_9": 0.2341721081339142,
        "w_10": 0.3780523499542964,
        "w_11": 0.3428728268932388,
        "w_12": 0.58025233963141,
        "w_13": 0.815817039069762,
        "w_14": 0.4865022144120589,
        "w_15": 0.01707600530474928,
        "w_16": 0.26623418232070073,
        "w_17": 0.20727099624448486,
        "w_18": 0.47400340366957744,
        "w_19": 0.6758531294442794,
    }

    weights = list(weights.values())

    return weights


# -----------------------------------------------------------------------------
# Prediction functions
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
    # for model, param n zip(models, params):
    for model_i in range(len(models)):
        model = models[model_i]
        param = params[model_i]
        temp_pred = []
        for rep_i in range(reps):
            print(
                f"NB266, Training model {model_i + 1}/{len(models)}, Repeat {rep_i + 1}/{reps}",
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


def run_notebook_266(train_df, test_df, pseudolabel, gene_names, reps):
    # determine mins and maxs for later clipping
    original_y = train_df.loc[:, gene_names].values
    mins = original_y.min(axis=0)
    maxs = original_y.max(axis=0)

    # combine pseudolabels into train_df
    train_df = pd.concat([train_df, pseudolabel]).reset_index(drop=True)
    original_y = train_df.loc[:, gene_names].values

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

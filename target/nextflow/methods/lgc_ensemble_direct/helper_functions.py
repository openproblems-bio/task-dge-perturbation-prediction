import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
from sklearn.model_selection import KFold as KF
from models import Conv, LSTM, GRU
from helper_classes import Dataset
from divisor_finder import find_balanced_divisors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('-----Seed Set!-----')
    
    
#### Data preprocessing utilities
def one_hot_encode(data_train, data_test, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    encoder = OneHotEncoder()
    encoder.fit(data_train)
    train_features = encoder.transform(data_train).toarray().astype(float)
    test_features = encoder.transform(data_test).toarray().astype(float)
    np.save(f"{out_dir}/one_hot_train.npy", train_features)
    np.save(f"{out_dir}/one_hot_test.npy", test_features)    
    return train_features, test_features

def pad_to_balanced_shape(x, target_shape):
    current_size = list(x.shape)
    target_size = current_size[:-1] + [target_shape[0] * target_shape[1]]
    padding_needed = target_size[-1] - current_size[-1]
    if padding_needed > 0:
        padding = np.zeros(current_size[:-1] + [padding_needed], dtype=x.dtype)
        padded = np.concatenate((x, padding), axis=-1)
    else:
        padded = x
    return padded    
        
def build_ChemBERTa_features(smiles_list):
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chemberta.eval()
    embeddings = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)
    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            model_output = chemberta(**encoded_input)
            embedding = model_output[0][::,0,::]
            embeddings[i] = embedding
            embedding = torch.mean(model_output[0], 1)
            embeddings_mean[i] = embedding
    return embeddings.numpy(), embeddings_mean.numpy()


def save_ChemBERTa_features(smiles_list, out_dir, on_train_data=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    emb, emb_mean = build_ChemBERTa_features(smiles_list)
    if on_train_data:
        np.save(f"{out_dir}/chemberta_train.npy", emb)
        np.save(f"{out_dir}/chemberta_train_mean.npy", emb_mean)
    else:
        np.save(f"{out_dir}/chemberta_test.npy", emb)
        np.save(f"{out_dir}/chemberta_test_mean.npy", emb_mean)                
    return emb, emb_mean
                
def combine_features(data_aug_dfs, chem_feats, main_df, one_hot_dfs=None, quantiles_df=None):
    """
    This function concatenates the provided vectors, matrices and data frames (i.e., one hot, std, mean, etc) into a single long vector. This is done for each input pair (cell_type, sm_name)
    """
    new_vecs = []
    chem_feat_dim = 600
    if len(data_aug_dfs) > 0:
        add_len = sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)
    else:
        add_len = chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else chem_feat_dim*len(chem_feats)
    if quantiles_df is not None:
        add_len += (quantiles_df.shape[1]-1)//3
    for i in range(len(main_df)):
        if one_hot_dfs is not None:
            vec_ = (one_hot_dfs.iloc[i,:].values).copy()
        else:
            vec_ = np.array([])
        for df in data_aug_dfs:
            if 'cell_type' in df.columns:
                values = df[df['cell_type']==main_df.iloc[i]['cell_type']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])
            else:
                assert 'sm_name' in df.columns
                values = df[df['sm_name']==main_df.iloc[i]['sm_name']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])
        for chem_feat in chem_feats:
            vec_ = np.concatenate([vec_, chem_feat[i]])
        final_vec = np.concatenate([vec_,np.zeros(add_len-vec_.shape[0],)])
        new_vecs.append(final_vec)

    new_final_vec = np.stack(new_vecs, axis=0).astype(float).reshape(len(main_df), 1, add_len)
    _, input_shape = find_balanced_divisors(new_final_vec.shape[-1])
    if input_shape[0]*input_shape[1] != new_final_vec.shape[-1]:
        new_final_vec = pad_to_balanced_shape(new_final_vec, (input_shape[0], input_shape[1]))
    return new_final_vec

def augment_data(x_, y_):
    copy_x = x_.copy()
    new_x = []
    new_y = y_.copy()
    dim = x_.shape[2]
    k = int(0.3*dim)
    for i in range(x_.shape[0]):
        idx = random.sample(range(dim), k=k)
        copy_x[i,:,idx] = 0
        new_x.append(copy_x[i])
    return np.stack(new_x, axis=0), new_y

#### Metrics
def mrrmse_np(y_pred, y_true):
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


#### Training utilities
def train_step(dataloader, model, opt, clip_norm):
    model.train()
    train_losses = []
    train_mrrmse = []
    for x, target in dataloader:
        model.to(device)
        x = x.to(device)
        target = target.to(device)
        loss = model(x, target)
        # aggregate loss if it's not a scalar
        if len(loss.size()) > 0:
            loss = loss.mean()
        train_losses.append(loss.item())
        pred = model(x).detach().cpu().numpy()
        train_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
    return np.mean(train_losses), np.mean(train_mrrmse)

def validation_step(dataloader, model):
    model.eval()
    val_losses = []
    val_mrrmse = []
    for x, target in dataloader:
        model.to(device)
        x = x.to(device)
        target = target.to(device)
        loss = model(x,target)
        # aggregate loss if it's not a scalar
        if len(loss.size()) > 0:
            loss = loss.mean()
        pred = model(x).detach().cpu().numpy()
        val_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        val_losses.append(loss.item())
    return np.mean(val_losses), np.mean(val_mrrmse)


def train_function(model, model_name, x_train, y_train, x_val, y_val, info_data, config, clip_norm=1.0):
    if model_name in ['GRU']:
        print('lr', config["LEARNING_RATES"][2])
        opt = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATES"][2])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATES"][0])
    model.to(device)
    results = {'train_loss': [], 'val_loss': [], 'train_mrrmse': [], 'val_mrrmse': [],\
               'train_cell_type': info_data['train_cell_type'], 'val_cell_type': info_data['val_cell_type'], 'train_sm_name': info_data['train_sm_name'], 'val_sm_name': info_data['val_sm_name'], 'runtime': None}
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    x_train_aug = np.concatenate([x_train, x_train_aug], axis=0)
    y_train_aug = np.concatenate([y_train, y_train_aug], axis=0)
    data_x_train = torch.FloatTensor(x_train_aug)
    data_y_train = torch.FloatTensor(y_train_aug)
    data_x_val = torch.FloatTensor(x_val)
    data_y_val = torch.FloatTensor(y_val)
    train_dataloader = DataLoader(Dataset(data_x_train, data_y_train), num_workers=1, batch_size=16, shuffle=True, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(Dataset(data_x_val, data_y_val), num_workers=1, batch_size=32, shuffle=False, pin_memory=True, persistent_workers=True)
    best_loss = np.inf
    best_weights = None
    t0 = time.time()
    for e in range(config["EPOCHS"]):
        loss, mrrmse = train_step(train_dataloader, model, opt, clip_norm)
        val_loss, val_mrrmse = validation_step(val_dataloader, model)
        results['train_loss'].append(float(loss))
        results['val_loss'].append(float(val_loss))
        results['train_mrrmse'].append(float(mrrmse))
        results['val_mrrmse'].append(float(val_mrrmse))
        if val_mrrmse < best_loss:
            best_loss = val_mrrmse
            best_weights = model.state_dict()
            # print('BEST ----> ')
        # print(f"{model.name} Epoch {e}, train_loss {round(loss,3)}, val_loss {round(val_loss, 3)}, val_mrrmse {val_mrrmse}")
    t1 = time.time()
    results['runtime'] = float(t1-t0)
    model.load_state_dict(best_weights)
    return model, results


def cross_validate_models(X, y, kf_cv, cell_types_sm_names, paths, config=None, scheme='initial', clip_norm=1.0):
    trained_models = []
    for i,(train_idx,val_idx) in enumerate(kf_cv.split(X)):
        print(f"\nSplit {i+1}/{kf_cv.n_splits}...")
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y.values[train_idx], y.values[val_idx]
        info_data = {'train_cell_type': cell_types_sm_names.iloc[train_idx]['cell_type'].tolist(),
                    'val_cell_type': cell_types_sm_names.iloc[val_idx]['cell_type'].tolist(),
                    'train_sm_name': cell_types_sm_names.iloc[train_idx]['sm_name'].tolist(),
                    'val_sm_name': cell_types_sm_names.iloc[val_idx]['sm_name'].tolist()}
        for Model in [LSTM, Conv, GRU]:
            model = Model(scheme, X.shape, y.shape)
            
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                model_name = model.module.name
            else:
                model_name = model.name

            model, results = train_function(model, model_name, x_train, y_train, x_val, y_val, info_data, config=config, clip_norm=clip_norm)
            model.to('cpu')
            trained_models.append(model)
            print(f'PATH OF THE MODEL EQUALS: {paths["model_dir"]}/pytorch_{model_name}_{scheme}_fold{i}.pt')
            torch.save(model.state_dict(), f'{paths["model_dir"]}/pytorch_{model_name}_{scheme}_fold{i}.pt')
            with open(f'{paths["logs_dir"]}/{model_name}_{scheme}_fold{i}.json', 'w') as file:
                json.dump(results, file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return trained_models

def train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, config, par, paths):
    kf_cv = KF(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)
    print(paths["model_dir"])
    if not os.path.exists(paths["model_dir"]):
        os.makedirs(paths["model_dir"], exist_ok=True)
    if not os.path.exists(paths["logs_dir"]):
        os.makedirs(paths["logs_dir"], exist_ok=True)
    shapes = {
        "xshapes": {
            'initial': X_vec.shape,
            'light': X_vec_light.shape,
            'heavy': X_vec_heavy.shape
        },
        "yshape": y.shape
    }
    with open(f'{paths["model_dir"]}/shapes.json', 'w') as file:
        json.dump(shapes, file)
    for scheme, clip_norm, input_features in zip(['initial', 'light', 'heavy'], config["CLIP_VALUES"], [X_vec, X_vec_light, X_vec_heavy]):
        if scheme in par["schemes"]:
            seed_everything()
            models = cross_validate_models(input_features, y, kf_cv, cell_types_sm_names, config=config, scheme=scheme, clip_norm=clip_norm, paths=paths)

#### Inference utilities
def inference_pytorch(model, dataloader):
    if isinstance(model, dict):
        model = load_model(model)
    model.eval()
    preds = []
    for x in dataloader:
        model.to(device)
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)

def average_prediction(X_test, trained_models):
    all_preds = []
    test_dataloader = DataLoader(Dataset(torch.FloatTensor(X_test)), num_workers=4, batch_size=64, shuffle=False)
    for i,model in enumerate(trained_models):
        current_pred = inference_pytorch(model, test_dataloader)
        all_preds.append(current_pred)
    return np.stack(all_preds, axis=1).mean(axis=1)


def weighted_average_prediction(X_test, trained_models, model_wise=[0.25, 0.35, 0.40], fold_wise=None):
    all_preds = []
    test_dataloader = DataLoader(Dataset(torch.FloatTensor(X_test)), num_workers=4, batch_size=64, shuffle=False)
    for i,model in enumerate(trained_models):
        current_pred = inference_pytorch(model, test_dataloader)
        current_pred = model_wise[i%3]*current_pred
        if fold_wise:
            current_pred = fold_wise[i//3]*current_pred
        all_preds.append(current_pred)
    return np.stack(all_preds, axis=1).sum(axis=1)

def load_trained_models(path, kf_n_splits=5):
    with open(f'{path}/shapes.json', 'r') as f:
        shapes = json.load(f)
    xshapes = shapes['xshapes']
    yshape = shapes['yshape']
    trained_models = {'initial': [], 'light': [], 'heavy': []}
    for scheme in ['initial', 'light', 'heavy']:
        for fold in range(kf_n_splits):
            for Model in [LSTM, Conv, GRU]:
                model = Model(scheme, xshapes[scheme], yshape)
                for weights_path in os.listdir(path):
                    if model.name in weights_path and scheme in weights_path and f'fold{fold}' in weights_path:
                        model.load_state_dict(torch.load(f'{path}/{weights_path}', map_location='cpu'))
                        trained_models[scheme].append(model)
    return trained_models

def lazy_load_trained_models(train_data_aug_dir, model_paths, kf_n_splits=5):
    with open(f'{train_data_aug_dir}/shapes.json', 'r') as f:
        shapes = json.load(f)
    xshapes = shapes['xshapes']
    yshape = shapes['yshape']
    trained_models = {'initial': [], 'light': [], 'heavy': []}
    for scheme in ['initial', 'light', 'heavy']:
        for fold in range(kf_n_splits):
            for model_name in ["LSTM", "Conv", "GRU"]:
                for weights_path in model_paths:
                    if model_name in weights_path and scheme in weights_path and f'fold{fold}' in weights_path:
                        # store settings in dict for later use
                        trained_models[scheme].append({
                            "model_name": model_name,
                            "model_path": weights_path,
                            "scheme": scheme,
                            "xshape": xshapes[scheme],
                            "yshape": yshape,
                            "fold": fold
                        })
    return trained_models

model_classes = {
    "LSTM": LSTM,
    "GRU": GRU,
    "Conv": Conv
}
def load_model(model_dict):
    ModelClass = model_classes[model_dict["model_name"]]
    model = ModelClass(model_dict["scheme"], model_dict["xshape"], model_dict["yshape"])
    model.load_state_dict(torch.load(model_dict["model_path"], map_location='cpu'))
    return model

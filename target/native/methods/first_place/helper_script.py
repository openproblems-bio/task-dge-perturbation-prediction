import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
from sklearn.model_selection import KFold as KF

class LogCoshLoss(nn.Module):
    """Loss function for regression tasks"""
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = (y_t - y_prime_t)/3 # divide by 3 to avoid numerical overflow in cosh
        clipped_ey_t = torch.clamp(ey_t, min=-50, max=50)
        return torch.mean(torch.log(torch.cosh(clipped_ey_t + 1e-12)))
    
    
class Dataset:
    """Python class to load the data for training and inference in Pytorch"""
    def __init__(self, data_x, data_y=None):
        super(Dataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        if self.data_y is not None:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dims_dict = {'conv': {'heavy': 13400, 'light': 4576, 'initial': 8992},
#                                     'rnn': {'linear': {'heavy': 99968, 'light': 24192, 'initial': 29568},
#                                            'input_shape': {'heavy': [779,142], 'light': [187,202], 'initial': [229,324]}
              
#                              }}

dims_dict = {'conv': {'heavy': 15624, 'light': 5312, 'initial': 10472},
                                    'rnn': {'linear': {'heavy': 3072, 'light': 9728, 'initial': 29440},
                                           'input_shape': {'heavy': [22, 5861], 'light': [74, 593], 'initial': [228, 379]}
                                           }}

class Conv(nn.Module):
    def __init__(self, scheme):
        super(Conv, self).__init__()
        self.name = 'Conv'
        self.conv_block = nn.Sequential(nn.Conv1d(1, 8, 5, stride=1, padding=0),
                                        nn.Dropout(0.3),
                                        nn.Conv1d(8, 8, 5, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv1d(8, 16, 5, stride=2, padding=0),
                                        nn.Dropout(0.3),
                                        nn.AvgPool1d(11),
                                        nn.Conv1d(16, 8, 3, stride=3, padding=0),
                                        nn.Flatten())
        self.scheme = scheme
        self.linear = nn.Sequential(
                nn.Linear(dims_dict['conv'][self.scheme], 1024),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.Dropout(0.3),
                nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        if y is None:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            return out
        else:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2
        

class LSTM(nn.Module):
    def __init__(self, scheme):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.scheme = scheme
        self.lstm = nn.LSTM(dims_dict['rnn']['input_shape'][self.scheme][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['rnn']['linear'][self.scheme], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = dims_dict['rnn']['input_shape'][self.scheme]
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2
        
        
class GRU(nn.Module):
    def __init__(self, scheme):
        super(GRU, self).__init__()
        self.name = 'GRU'
        self.scheme = scheme
        self.gru = nn.GRU(dims_dict['rnn']['input_shape'][self.scheme][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['rnn']['linear'][self.scheme], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = dims_dict['rnn']['input_shape'][self.scheme]
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2

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
    train_features = encoder.transform(data_train)
    test_features = encoder.transform(data_test)
    np.save(f"{out_dir}/one_hot_train.npy", train_features.toarray().astype(float))
    np.save(f"{out_dir}/one_hot_test.npy", test_features.toarray().astype(float))        
        
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
    return np.stack(new_vecs, axis=0).astype(float).reshape(len(main_df), 1, add_len)

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
        pred = model(x).detach().cpu().numpy()
        val_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        val_losses.append(loss.item())
    return np.mean(val_losses), np.mean(val_mrrmse)


def train_function(model, x_train, y_train, x_val, y_val, info_data, config, clip_norm=1.0):
    if model.name in ['GRU']:
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
            print('BEST ----> ')
        print(f"{model.name} Epoch {e}, train_loss {round(loss,3)}, val_loss {round(val_loss, 3)}, val_mrrmse {val_mrrmse}")
    t1 = time.time()
    results['runtime'] = float(t1-t0)
    model.load_state_dict(best_weights)
    return model, results


def cross_validate_models(X, y, kf_cv, cell_types_sm_names, model_dir, logs_dir, config=None, scheme='initial', clip_norm=1.0):
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
            model = Model(scheme)
            model, results = train_function(model, x_train, y_train, x_val, y_val, info_data, config=config, clip_norm=clip_norm)
            model.to('cpu')
            trained_models.append(model)
            print(f'PATH OF THE MODEL EQUALS: {model_dir}/pytorch_{model.name}_{scheme}_fold{i}.pt')
            torch.save(model.state_dict(), f'{model_dir}/pytorch_{model.name}_{scheme}_fold{i}.pt')
            with open(f'{logs_dir}/{model.name}_{scheme}_fold{i}.json', 'w') as file:
                json.dump(results, file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return trained_models

def train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, config, model_names, model_dir, logs_dir):
    kf_cv = KF(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)
    trained_models = {'initial': [], 'light': [], 'heavy': []}
    print(model_dir)
    if not os.path.exists(model_dir):
        print("MODEL DIR DID NOT EXIST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    for scheme, clip_norm, input_features in zip(['initial', 'light', 'heavy'], config["CLIP_VALUES"], [X_vec, X_vec_light, X_vec_heavy]):
        if scheme in model_names:
            seed_everything()
            models = cross_validate_models(input_features, y, kf_cv, cell_types_sm_names, config=config, scheme=scheme, clip_norm=clip_norm, model_dir=model_dir, logs_dir=logs_dir)
            trained_models[scheme].extend(models)
    return trained_models

#### Inference utilities
def inference_pytorch(model, dataloader):
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
    trained_models = {'initial': [], 'light': [], 'heavy': []}
    for scheme in ['initial', 'light', 'heavy']:
        for fold in range(kf_n_splits):
            for Model in [LSTM, Conv, GRU]:
                model = Model(scheme)
                for weights_path in os.listdir(path):
                    if model.name in weights_path and scheme in weights_path and f'fold{fold}' in weights_path:
                        model.load_state_dict(torch.load(f'{path}/{weights_path}', map_location='cpu'))
                        trained_models[scheme].append(model)
    return trained_models

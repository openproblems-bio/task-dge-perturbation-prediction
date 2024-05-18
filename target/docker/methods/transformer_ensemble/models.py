# Model Architecture
import torch
import torch.nn as nn
import torch.optim


class CustomTransformer(nn.Module):
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6):  # num_heads=8
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        # Embedding layer for sparse features
        # self.embedding = nn.Embedding(num_features, d_model)

        # self.norm = nn.BatchNorm1d(d_model, affine=True)
        self.norm = nn.LayerNorm(d_model)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
        #                                 dropout=0.1, device='cuda')
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, device='cuda', dropout=0.3,
                                       activation=nn.GELU(),
                                       batch_first=True), enable_nested_tensor=True, num_layers=num_layers
        )
        # Dropout layer for regularization
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        x = self.embedding(x)

        # x = (self.transformer(x,x))
        x = self.transformer(x)
        x = self.norm(x)
        # x = self.fc(self.dropout(x))
        x = self.fc(x)
        return x


class CustomTransformer_mean_std(nn.Module):  # mean + std
    def __init__(self, num_features, num_targets, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(CustomTransformer_mean_std, self).__init__()
        self.num_target_encodings = num_targets * 4
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x

class CustomTransformer_mean(nn.Module):  # mean + std
    def __init__(self, num_features, num_targets, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(CustomTransformer_mean, self).__init__()
        self.num_target_encodings = num_targets * 2
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x

class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.3, layer_norm=True):
        super(CustomMLP, self).__init__()
        layers = []

        for _ in range(num_layers):
            if layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
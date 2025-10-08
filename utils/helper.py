import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from vmdpy import VMD


def create_sequences(data, target, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len + horizon - 1])  # target at t + horizon
    return np.array(X), np.array(y)


def scaled_data(df):
    target_col = "Price"
    features = [col for col in df.columns if col != target_col and col != "Date"]
    train_size = int(len(df) * 0.7)

    train_df = df[:train_size]

    with open("../models/scaler/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)

    with open("../models/scaler/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    df[features] = feature_scaler.transform(df[features])
    df[target_col] = target_scaler.transform(df[[target_col]]).flatten()

    return df


def scaled_data_vmd(df, _K=5):
    target_col = "Price"
    features = [col for col in df.columns if col != target_col and col != "Date"]

    alpha = 2000
    tau = 0.0
    K = _K
    DC = 0
    init = 1
    tol = 1e-7

    vmd_results = {}

    for feature in features:
        signal = df[feature].values

        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

        vmd_results[feature] = {
            'modes': u,
            'freq_domain': u_hat,
            'frequencies': omega
        }

        mode_df = pd.DataFrame(u.T, columns=[f'{feature}_Mode_{i+1}' for i in range(K)])
        vmd_results[feature]['mode_df'] = mode_df

    for feature in features:
        modes = vmd_results[feature]['modes']

        for i in range(K):
            column_name = f"{feature}_mode_{i+1}"
            df[column_name] = modes[i]

    features = [col for col in df.columns if col != target_col and col != "Date"]

    train_size = int(len(df) * 0.7)
    train_df = df[:train_size]

    # Initialize scaler
    feature_vmd_scaler = StandardScaler()
    target_vmd_scaler = StandardScaler()

    feature_vmd_scaler.fit(train_df[features])
    target_vmd_scaler.fit(train_df[[target_col]])

    df[features] = feature_vmd_scaler.transform(df[features])
    df[target_col] = target_vmd_scaler.transform(df[[target_col]]).flatten()

    return df


def data_loaders(df, X, y, batch_size=256):
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.1)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # No shuffle for time series
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

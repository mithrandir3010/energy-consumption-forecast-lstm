import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LSTMForecast


# -----------------------------
# Utils
# -----------------------------
class MinMaxScaler1D:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        if self.max_ - self.min_ < 1e-12:
            self.max_ = self.min_ + 1e-12
        return self

    def transform(self, x: np.ndarray):
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x):
        return x * (self.max_ - self.min_) + self.min_


def create_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)  # (N, seq, 1)
        self.y = torch.tensor(y).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            pred = model(Xb)
            loss = criterion(pred, yb)
            if is_train:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    # -----------------------------
    # Config
    # -----------------------------
    CSV_PATH = os.environ.get("CSV_PATH", "full_data.csv")
    SEQ_LEN = int(os.environ.get("SEQ_LEN", "24"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
    EPOCHS = int(os.environ.get("EPOCHS", "50"))
    LR = float(os.environ.get("LR", "0.001"))
    PATIENCE = int(os.environ.get("PATIENCE", "7"))

    os.makedirs("artifacts", exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(CSV_PATH)

    # time parse: "01:01:2018:00:00" -> %d:%m:%Y:%H:%M
    df["time_parsed"] = pd.to_datetime(df["time"].astype(str).str.strip(), format="%d:%m:%Y:%H:%M", errors="coerce")
    df["target"] = pd.to_numeric(df["consumption_MWh"], errors="coerce")

    df = df.dropna(subset=["time_parsed", "target"]).sort_values("time_parsed").reset_index(drop=True)
    values = df["target"].values.astype(np.float32)

    n = len(values)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]

    scaler = MinMaxScaler1D().fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    X_train, y_train = create_sequences(train, SEQ_LEN)
    X_val, y_val = create_sequences(val, SEQ_LEN)
    X_test, y_test = create_sequences(test, SEQ_LEN)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Train
    # -----------------------------
    model = LSTMForecast(hidden_size=64, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    pat_cnt = 0
    best_path = os.path.join("artifacts", "lstm_model.pth")
    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        print(f"Epoch {epoch:02d} | train MSE: {tr_loss:.6f} | val MSE: {va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            pat_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # save history
    history_df = pd.DataFrame({
        "epoch": list(range(1, len(train_hist) + 1)),
        "train_mse": train_hist,
        "val_mse": val_hist
    })
    history_df.to_csv(os.path.join("artifacts", "training_history.csv"), index=False)

    # save scaler info
    with open(os.path.join("artifacts", "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"min": scaler.min_, "max": scaler.max_, "seq_len": SEQ_LEN}, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Evaluate (Test)
    # -----------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb).cpu().numpy().reshape(-1)
            true = yb.numpy().reshape(-1)
            preds.append(pred)
            trues.append(true)

    pred_scaled = np.concatenate(preds)
    true_scaled = np.concatenate(trues)

    pred = scaler.inverse_transform(pred_scaled)
    true = scaler.inverse_transform(true_scaled)

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)

    # Naive baseline
    naive_scaled = X_test[:, -1]
    naive = scaler.inverse_transform(naive_scaled)
    true2 = scaler.inverse_transform(y_test)

    mae_n = float(np.mean(np.abs(naive_

# train_microstructure.py
# Trains the Microstructure multi-head model
# 32 features → 6 heads | ResidualBlock × 2 | Stem(256) + Mid(128)
import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
DATA_PATH       = "synthetic_microstructure_dataset.csv"
SCALER_PATH     = "microstructure_scaler.pkl"
BEST_MODEL_PATH = "microstructure_best.pth"
FINAL_MODEL_PATH= "microstructure_final.pth"

BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 1e-3
WD          = 1e-4
VAL_RATIO   = 0.12
TRUNK_DIM   = 256
DROP        = 0.25
SEED        = 42

# head sizes: flow_dom(3) bar_intent(4) cluster(4) rej_struct(4) abs_state(3) micro_sig(3)
HEAD_SIZES    = [3, 4, 4, 4, 3, 3]
HEAD_WEIGHTS  = [1.0, 1.0, 0.9, 0.8, 0.8, 1.2]   # micro_sig weighted highest
LABEL_COLS    = ["label_flow_dom","label_bar_intent","label_cluster",
                 "label_rej_struct","label_abs_state","label_micro_sig"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class MicroNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7)
        )
        # 6 independent heads
        self.h0 = nn.Linear(trunk // 2, HEAD_SIZES[0])   # flow_dominance
        self.h1 = nn.Linear(trunk // 2, HEAD_SIZES[1])   # bar_intent
        self.h2 = nn.Linear(trunk // 2, HEAD_SIZES[2])   # cluster_state
        self.h3 = nn.Linear(trunk // 2, HEAD_SIZES[3])   # rejection_structure
        self.h4 = nn.Linear(trunk // 2, HEAD_SIZES[4])   # absorption_state
        self.h5 = nn.Linear(trunk // 2, HEAD_SIZES[5])   # micro_signal

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.h0(h), self.h1(h), self.h2(h), self.h3(h), self.h4(h), self.h5(h)


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class MicroDS(Dataset):
    def __init__(self, X, Y_list):
        self.X  = torch.tensor(X.astype(np.float32))
        self.Ys = [torch.tensor(y.astype(np.int64)) for y in Y_list]

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return (self.X[i],) + tuple(y[i] for y in self.Ys)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def class_weights(arr, n):
    cnt = Counter(arr.astype(int).tolist())
    tot = sum(cnt.values())
    return torch.tensor(
        [tot / (cnt.get(k,0)*n) if cnt.get(k,0)>0 else 1.0 for k in range(n)],
        dtype=torch.float32)


def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    correct = [0]*6; totals = [0]*6

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            outs = model(xb)

            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(6))

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
            for i in range(6):
                pred = outs[i].argmax(1)
                correct[i] += (pred == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    avg_loss = total_loss / total_n
    accs     = [correct[i]/totals[i] for i in range(6)]
    return avg_loss, accs


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    for lc in LABEL_COLS:
        if lc not in df.columns:
            raise RuntimeError(f"Missing label column: {lc}")

    Ys        = [df[lc].astype(int).values for lc in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X         = df[feat_cols].fillna(0.0)

    print(f"  Samples  : {len(df)}")
    print(f"  Features : {len(feat_cols)}")

    # Train / val split
    n       = len(df)
    n_val   = int(n * VAL_RATIO)
    idx     = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    Ytr  = [y[tr_idx] for y in Ys]
    Yva  = [y[va_idx] for y in Ys]

    # Scale
    scaler   = RobustScaler()
    Xtr_s    = scaler.fit_transform(Xtr.values)
    Xva_s    = scaler.transform(Xva.values)
    bundle   = {"scaler": scaler, "feature_names": list(feat_cols)}
    joblib.dump(bundle, SCALER_PATH)
    print(f"  Scaler saved → {SCALER_PATH}")

    # Loaders
    tr_loader = DataLoader(MicroDS(Xtr_s, Ytr), batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0)
    va_loader = DataLoader(MicroDS(Xva_s, Yva), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

    # Model + loss functions
    model    = MicroNet(input_dim=Xtr_s.shape[1]).to(DEVICE)
    loss_fns = []
    for i, (lc, nc) in enumerate(zip(LABEL_COLS, HEAD_SIZES)):
        w = class_weights(Ytr[i], nc).to(DEVICE)
        loss_fns.append(nn.CrossEntropyLoss(weight=w))
        print(f"  Head {i} ({lc}): classes={nc}  weights={w.cpu().numpy().round(3)}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.05)

    best_val = float("inf")
    print(f"\nTraining for {EPOCHS} epochs ...\n")

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, loss_fns, opt)
        va_loss, va_acc = run_epoch(model, va_loader, loss_fns)
        sched.step()

        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            saved = "  ← best"

        if ep % 5 == 0 or ep == 1:
            acc_str = " ".join(f"{a:.2f}" for a in va_acc)
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} | vaL={va_loss:.4f} | "
                  f"vaAcc=[{acc_str}] | lr={sched.get_last_lr()[0]:.2e}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅  Best  model → {BEST_MODEL_PATH}  (val_loss={best_val:.4f})")
    print(f"✅  Final model → {FINAL_MODEL_PATH}")
    print(f"\nHead names  : {LABEL_COLS}")
    print(f"Head sizes  : {HEAD_SIZES}")
    print(f"Feature cols: {list(feat_cols)}")


if __name__ == "__main__":
    main()

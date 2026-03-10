# microstructure_deploy.py
# Live inference wrapper for the Microstructure model
# Computes 32 features EXACTLY matching synthetic_microstructure.py
# Outputs 6 head predictions + confidence dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from utils import resource_path
except ImportError:
    def resource_path(p): return p

EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────
# Architecture  — must match train_microstructure.py exactly
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
    HEAD_SIZES = [3, 4, 4, 4, 3, 3]

    def __init__(self, input_dim, trunk=256, drop=0.25):
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
        hs = self.HEAD_SIZES
        h  = trunk // 2
        self.h0 = nn.Linear(h, hs[0])
        self.h1 = nn.Linear(h, hs[1])
        self.h2 = nn.Linear(h, hs[2])
        self.h3 = nn.Linear(h, hs[3])
        self.h4 = nn.Linear(h, hs[4])
        self.h5 = nn.Linear(h, hs[5])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.h0(h), self.h1(h), self.h2(h), self.h3(h), self.h4(h), self.h5(h)


# ─────────────────────────────────────────────────────────────────────
# Feature computation  — identical to synthetic_microstructure.py
# ─────────────────────────────────────────────────────────────────────

def _rolling_z(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : DataFrame with columns [open, high, low, close, volume]
            Needs at least 40 rows for z-scores to stabilise.
    Output: Same DataFrame with all 32 feature columns appended.
    """
    df = df.copy().reset_index(drop=True)

    # ── base anatomy ──────────────────────────────────────────────────
    df["range"]       = df["high"] - df["low"]
    df["body"]        = df["close"] - df["open"]
    df["abs_body"]    = df["body"].abs()
    df["dir"]         = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"]  = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"]  = df[["open","close"]].min(axis=1) - df["low"]

    # ── 23 kept features ──────────────────────────────────────────────
    df["body_ratio"]      = df["abs_body"]   / (df["range"] + EPS)
    df["upper_ratio"]     = df["upper_wick"] / (df["range"] + EPS)
    df["lower_ratio"]     = df["lower_wick"] / (df["range"] + EPS)
    df["body_ratio_s5"]   = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["upper_ratio_s5"]  = df["upper_ratio"].rolling(5, min_periods=1).mean()
    df["lower_ratio_s5"]  = df["lower_ratio"].rolling(5, min_periods=1).mean()

    df["body_z"]          = _rolling_z(df["abs_body"], 40).fillna(0.0)
    df["range_z"]         = _rolling_z(df["range"],   40).fillna(0.0)
    df["tail_asym"]       = (df["upper_wick"] - df["lower_wick"]) / (df["range"] + EPS)
    df["tail_to_body"]    = (df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)

    df["signed_vol"]      = df["volume"] * df["dir"]
    df["orderflow_5"]     = df["signed_vol"].rolling(5,  min_periods=1).sum()
    df["orderflow_10"]    = df["signed_vol"].rolling(10, min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"] < df["abs_body"].rolling(5, min_periods=1).mean()*0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"] >  0, "opp_wick"] = df["lower_wick"]
    df.loc[df["dir"] <  0, "opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"] * df["opp_wick"]).rolling(5, min_periods=1).sum()

    df["atr5"]            = df["range"].rolling(5,  min_periods=1).mean()
    df["atr10"]           = df["range"].rolling(10, min_periods=1).mean()
    df["pullback_depth"]  = df["range"] / (df["atr5"] + EPS)
    df["pullback_z"]      = _rolling_z(df["pullback_depth"], 20).fillna(0.0)
    df["exhaustion"]      = ((df["upper_wick"] + df["lower_wick"]) /
                              (df["abs_body"] + EPS)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"] * (
        1 + df["orderflow_5"].abs() / (df["volume"].rolling(5,min_periods=1).mean() + EPS))
    df["wick_body_inter"] = ((df["upper_wick"] - df["lower_wick"]) * df["abs_body"]).clip(-2.0, 2.0)
    df["body_vol_inter"]  = (df["abs_body"] * df["volume"]).clip(-1e6, 1e6)

    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    # ── 9 new physics features ────────────────────────────────────────
    df["close_gravity"]   = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["body_mid_disp"]   = (((df["open"]+df["close"])/2) -
                               ((df["high"]+df["low"])/2)) / (df["range"] + EPS)
    df["tick_delta"]      = df["close_gravity"] * 2 - 1

    delta_vol             = df["tick_delta"] * df["volume"]
    df["cum_delta_5z"]    = _rolling_z(delta_vol.rolling(5,min_periods=1).sum(), 20).fillna(0.0)

    vol_avg10             = df["volume"].rolling(10, min_periods=1).mean()
    dom_wick              = df[["upper_wick","lower_wick"]].max(axis=1)
    rej_dir               = np.where(df["lower_wick"] > df["upper_wick"], 1, -1)
    df["rejection_qual"]  = ((dom_wick / (df["range"]+EPS)) *
                              (df["volume"] / (vol_avg10+EPS)) * rej_dir).clip(-4, 4)

    comp_a                = (df["body_ratio"] < 0.25).astype(int)
    comp_b                = (df["range"] / (df["atr10"]+EPS) > 0.80).astype(int)
    comp_c                = (df["volume"] / (vol_avg10+EPS) > 1.20).astype(int)
    df["absorption_fp"]   = (comp_a + comp_b + comp_c).astype(float)

    bull                  = df["dir"] >= 0
    df["dir_efficiency"]  = np.where(bull,
                                     df["body_ratio"] * df["close_gravity"],
                                     df["body_ratio"] * (1 - df["close_gravity"]))

    vol_avg5              = df["volume"].rolling(5, min_periods=1).mean()
    df["bar_energy"]      = (df["range"] / (df["atr5"]+EPS) *
                              df["volume"] / (vol_avg5+EPS)).clip(0, 6)

    df["gap_pressure"]    = ((df["open"] - df["close"].shift(1).fillna(df["open"])) /
                               (df["atr5"]+EPS)).clip(-3, 3)

    return df


# ─────────────────────────────────────────────────────────────────────
# Predictor class
# ─────────────────────────────────────────────────────────────────────

# Human-readable labels for each head
HEAD_LABELS = {
    "flow_dominance":      {0:"seller_dominated", 1:"contested",     2:"buyer_dominated"},
    "bar_intent":          {0:"absorption",        1:"rejection",     2:"impulse",        3:"exhaustion"},
    "cluster_state":       {0:"no_cluster",        1:"building",      2:"mature",         3:"climax"},
    "rejection_structure": {0:"none",              1:"upper",         2:"lower",          3:"dual"},
    "absorption_state":    {0:"none",              1:"passive",       2:"active"},
    "micro_signal":        {0:"avoid",             1:"wait",          2:"act"},
}

FEATURE_COLS = sorted([
    "abs_body","upper_wick","lower_wick","range",
    "body_ratio","upper_ratio","lower_ratio",
    "body_ratio_s5","upper_ratio_s5","lower_ratio_s5",
    "body_z","range_z","tail_asym","tail_to_body",
    "orderflow_5","orderflow_10",
    "absorption_raw","pullback_z","exhaustion","expected_move_raw",
    "wick_body_inter","body_vol_inter","consec_len",
    "close_gravity","body_mid_disp","tick_delta","cum_delta_5z",
    "rejection_qual","absorption_fp","dir_efficiency",
    "bar_energy","gap_pressure",
])


class MicrostructurePredictor:
    def __init__(self,
                 model_path  = "microstructure_best.pth",
                 scaler_path = "microstructure_scaler.pkl",
                 min_bars    = 50):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_bars = min_bars

        # Load scaler bundle
        bundle = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]   # ordered list saved at training time

        # Load model
        self.model = MicroNet(input_dim=len(self.feature_cols)).to(self.device)
        sd = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # MT5
        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    # ── Live prediction from MT5 ──────────────────────────────────────

    def predict(self, symbol: str, timeframe=None):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, self.min_bars + 10)
        if rates is None or len(rates) < self.min_bars:
            return {"error": f"Insufficient data: got {0 if rates is None else len(rates)} bars"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    # ── Prediction from any OHLCV DataFrame ──────────────────────────

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        """
        df must have columns: open, high, low, close, volume
        Returns dict with all head predictions + confidence + raw probs.
        """
        df = compute_features(df).dropna(subset=self.feature_cols)
        if df.empty:
            return {"error": "No valid rows after feature computation"}

        row    = df.iloc[[-1]][self.feature_cols]
        scaled = self.scaler.transform(row.values)
        x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outs = self.model(x)

        head_names = list(HEAD_LABELS.keys())
        result     = {}
        for i, (name, out) in enumerate(zip(head_names, outs)):
            probs     = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx  = int(np.argmax(probs))
            label_map = HEAD_LABELS[name]
            result[name] = {
                "class":      pred_idx,
                "label":      label_map[pred_idx],
                "confidence": float(np.max(probs)),
                "probs":      {label_map[k]: round(float(probs[k]), 4)
                               for k in range(len(probs))},
            }

        # Convenience top-level signal
        result["signal"]     = result["micro_signal"]["label"]
        result["confidence"] = result["micro_signal"]["confidence"]
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result    = predictor.predict("EURUSD")

    print("\n── Microstructure Prediction ────────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<24} → {v['label']}  (conf={v['confidence']:.2%})")
    print(f"\n  SIGNAL: {result['signal'].upper()}  "
          f"(confidence={result['confidence']:.2%})")
    predictor.shutdown()

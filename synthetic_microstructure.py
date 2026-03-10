# synthetic_microstructure.py
# Generates training data for the Microstructure model
# 32 features | 6 heads | FX M15 auction physics
import pandas as pd
import numpy as np
import random
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw):
        return range(n)

N_SAMPLES   = 20000
SEQ_LEN     = 25          # bars fetched; last-bar features + 5-bar context
OUTPUT_PATH = "synthetic_microstructure_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────
# Candle generator
# ─────────────────────────────────────────────────────────────────────

def generate_candle(base_price, body_scale=0.00045, wick_mult=(0.1, 0.8),
                    vol_range=(200, 2000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body      = float(np.random.uniform(0.15, 1.0) * body_scale)
    wick      = float(body * np.random.uniform(*wick_mult))
    tail      = float(body * np.random.uniform(*wick_mult))
    o  = float(base_price)
    c  = float(o + direction * body)
    h  = max(o, c) + wick
    l  = min(o, c) - tail
    v  = int(np.random.uniform(*vol_range))
    return {"open": o, "high": float(h), "low": float(l),
            "close": float(c), "volume": v}, c


def rolling_z(x, w):
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — exactly matches deploy wrapper
# ─────────────────────────────────────────────────────────────────────

def compute_features(df):
    df  = df.copy().reset_index(drop=True)

    # ── base anatomy ──────────────────────────────────────────────────
    df["range"]       = df["high"] - df["low"]
    df["body"]        = df["close"] - df["open"]
    df["abs_body"]    = df["body"].abs()
    df["dir"]         = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"]  = df["high"]  - df[["open","close"]].max(axis=1)
    df["lower_wick"]  = df[["open","close"]].min(axis=1) - df["low"]

    # ── KEPT features (23) ────────────────────────────────────────────
    df["body_ratio"]     = df["abs_body"]   / (df["range"] + EPS)
    df["upper_ratio"]    = df["upper_wick"] / (df["range"] + EPS)
    df["lower_ratio"]    = df["lower_wick"] / (df["range"] + EPS)
    df["body_ratio_s5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["upper_ratio_s5"] = df["upper_ratio"].rolling(5, min_periods=1).mean()
    df["lower_ratio_s5"] = df["lower_ratio"].rolling(5, min_periods=1).mean()

    df["body_z"]         = rolling_z(df["abs_body"], 40).fillna(0.0)
    df["range_z"]        = rolling_z(df["range"],   40).fillna(0.0)
    df["tail_asym"]      = (df["upper_wick"] - df["lower_wick"]) / (df["range"] + EPS)
    df["tail_to_body"]   = (df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)

    df["signed_vol"]     = df["volume"] * df["dir"]
    df["orderflow_5"]    = df["signed_vol"].rolling(5,  min_periods=1).sum()
    df["orderflow_10"]   = df["signed_vol"].rolling(10, min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"] < df["abs_body"].rolling(5,min_periods=1).mean()*0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"] >  0, "opp_wick"] = df["lower_wick"]
    df.loc[df["dir"] <  0, "opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"] * df["opp_wick"]).rolling(5, min_periods=1).sum()

    df["atr5"]            = df["range"].rolling(5,  min_periods=1).mean()
    df["atr10"]           = df["range"].rolling(10, min_periods=1).mean()
    df["range_mean5"]     = df["atr5"]
    df["pullback_depth"]  = df["range"] / (df["range_mean5"] + EPS)
    df["pullback_z"]      = rolling_z(df["pullback_depth"], 20).fillna(0.0)
    df["exhaustion"]      = ((df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"] * (1 + df["orderflow_5"].abs() / (df["volume"].rolling(5,min_periods=1).mean() + EPS))

    df["wick_body_inter"] = ((df["upper_wick"] - df["lower_wick"]) * df["abs_body"]).clip(-2.0, 2.0)
    df["body_vol_inter"]  = (df["abs_body"] * df["volume"]).clip(-1e6, 1e6)

    # consecutive run length
    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    # ── NEW physics features (9) ──────────────────────────────────────

    # N1: close gravity — final intrabar vote
    df["close_gravity"]  = (df["close"] - df["low"]) / (df["range"] + EPS)

    # N2: body midpoint displacement — body above/below bar center?
    df["body_mid_disp"]  = (((df["open"]+df["close"])/2) - ((df["high"]+df["low"])/2)) / (df["range"] + EPS)

    # N3: tick delta proxy — directional volume estimate without L2
    df["tick_delta"]     = df["close_gravity"] * 2 - 1          # [-1, +1]

    # N4: cumulative 5-bar delta z-scored
    delta_vol            = df["tick_delta"] * df["volume"]
    df["cum_delta_5z"]   = rolling_z(delta_vol.rolling(5,min_periods=1).sum(), 20).fillna(0.0)

    # N5: rejection quality — dominant wick × volume weight, signed
    vol_avg10            = df["volume"].rolling(10, min_periods=1).mean()
    dom_wick             = df[["upper_wick","lower_wick"]].max(axis=1)
    rej_dir              = np.where(df["lower_wick"] > df["upper_wick"], 1, -1)
    df["rejection_qual"] = ((dom_wick / (df["range"]+EPS)) *
                            (df["volume"] / (vol_avg10+EPS)) *
                            rej_dir).clip(-4, 4)

    # N6: absorption fingerprint — 3-component iceberg test
    comp_a               = (df["body_ratio"] < 0.25).astype(int)
    comp_b               = (df["range"] / (df["atr10"]+EPS) > 0.80).astype(int)
    comp_c               = (df["volume"] / (vol_avg10+EPS) > 1.20).astype(int)
    df["absorption_fp"]  = (comp_a + comp_b + comp_c).astype(float)

    # N7: directional efficiency — quality of bar, not just size
    bull = df["dir"] >= 0
    df["dir_efficiency"] = np.where(bull,
                                    df["body_ratio"] * df["close_gravity"],
                                    df["body_ratio"] * (1 - df["close_gravity"]))

    # N8: bar energy signature — range × volume both must be elevated
    vol_avg5             = df["volume"].rolling(5, min_periods=1).mean()
    df["bar_energy"]     = (df["range"] / (df["atr5"]+EPS) *
                            df["volume"] / (vol_avg5+EPS)).clip(0, 6)

    # N9: open gap pressure — off-session shift relative to ATR
    df["gap_pressure"]   = ((df["open"] - df["close"].shift(1).fillna(df["open"])) /
                             (df["atr5"]+EPS)).clip(-3, 3)

    return df


# ─────────────────────────────────────────────────────────────────────
# Label generation — 6 heads
# ─────────────────────────────────────────────────────────────────────

def compute_labels(df):
    last = df.iloc[-1]

    # ── HEAD 1: flow_dominance  [0=seller 1=contested 2=buyer] ───────
    td   = float(last["tick_delta"])
    cg   = float(last["close_gravity"])
    cd5  = float(last["cum_delta_5z"])
    of5  = float(last["orderflow_5"])
    if   td > 0.25 and cg > 0.60 and cd5 > 0.3:    flow_dom = 2
    elif td < -0.25 and cg < 0.40 and cd5 < -0.3:   flow_dom = 0
    else:                                             flow_dom = 1

    # ── HEAD 2: bar_intent  [0=absorption 1=rejection 2=impulse 3=exhaustion] ─
    afp  = float(last["absorption_fp"])
    rq   = abs(float(last["rejection_qual"]))
    de   = float(last["dir_efficiency"])
    be   = float(last["bar_energy"])
    cl   = int(last["consec_len"])
    br   = float(last["body_ratio"])
    brs5 = float(last["body_ratio_s5"])
    if   afp >= 2:                                   bar_intent = 0   # absorption
    elif rq > 0.35 and br < 0.45:                    bar_intent = 1   # rejection
    elif de > 0.40 and be > 1.4:                     bar_intent = 2   # impulse
    elif cl >= 4 and br < brs5 * 0.75:               bar_intent = 3   # exhaustion
    else:                                             bar_intent = 2   # default impulse

    # ── HEAD 3: cluster_state  [0=none 1=building 2=mature 3=climax] ─
    vol_now  = float(last["volume"])
    vol_trend_flag = (df["volume"].iloc[-3:].mean() >
                      df["volume"].iloc[-6:-3].mean()) if len(df) >= 6 else False
    body_decay = (br < brs5 * 0.70)
    if   cl <= 1:                                     cluster = 0
    elif cl <= 3:                                     cluster = 1
    elif cl >= 5 and body_decay and vol_trend_flag:   cluster = 3   # climax
    else:                                             cluster = 2

    # ── HEAD 4: rejection_structure  [0=none 1=upper 2=lower 3=dual] ─
    ur = float(last["upper_ratio"])
    lr = float(last["lower_ratio"])
    rq_signed = float(last["rejection_qual"])
    if   ur > 0.32 and lr > 0.28:                    rej_struct = 3
    elif rq_signed < -0.15 and ur > 0.30:            rej_struct = 1
    elif rq_signed >  0.15 and lr > 0.30:            rej_struct = 2
    else:                                             rej_struct = 0

    # ── HEAD 5: absorption_state  [0=none 1=passive 2=active] ────────
    abs_raw = float(last["absorption_raw"])
    seq_mean_body = float(df["abs_body"].mean())
    if   afp >= 2:                                    abs_state = 2
    elif afp == 1 or abs_raw > seq_mean_body * 0.3:  abs_state = 1
    else:                                             abs_state = 0

    # ── HEAD 6: micro_signal  [0=avoid 1=wait 2=act] ──────────────────
    act_cond  = (flow_dom != 1 and
                 bar_intent in (1, 2) and
                 cluster in (1, 2) and
                 be > 1.1 and de > 0.35)
    avoid_cond= (bar_intent == 0 or    # absorption — don't fight iceberg
                 cluster == 3 or       # climax — don't join exhausted run
                 be < 0.4)             # thin bar — no energy
    if   act_cond:   micro_sig = 2
    elif avoid_cond: micro_sig = 0
    else:            micro_sig = 1

    return (int(flow_dom), int(bar_intent), int(cluster),
            int(rej_struct), int(abs_state), int(micro_sig))


# ─────────────────────────────────────────────────────────────────────
# Scenario generator
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "impulse_clean":     dict(bias=0.75, vol=(600,2200), body=0.0009, wicks=(0.05,0.35)),
    "impulse_messy":     dict(bias=0.70, vol=(500,2000), body=0.0008, wicks=(0.15,0.60)),
    "wick_rejection":    dict(bias=0.50, vol=(400,1600), body=0.0004, wicks=(0.40,1.20)),
    "absorption_slow":   dict(bias=0.55, vol=(300,1400), body=0.0003, wicks=(0.20,0.70)),
    "absorption_active": dict(bias=0.52, vol=(800,2800), body=0.0002, wicks=(0.30,0.80)),
    "chop":              dict(bias=0.50, vol=(150, 700), body=0.0003, wicks=(0.10,0.50)),
    "spike_news":        dict(bias=0.50, vol=(1200,6000),body=0.0012, wicks=(0.05,0.30)),
    "exhaustion_run":    dict(bias=0.72, vol=(400,2000), body=0.0007, wicks=(0.10,0.45)),
    "dual_rejection":    dict(bias=0.50, vol=(500,1800), body=0.0003, wicks=(0.50,1.50)),
}
WEIGHTS = [0.12, 0.10, 0.14, 0.10, 0.08, 0.16, 0.06, 0.14, 0.10]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(list(SCENARIOS.keys()), weights=WEIGHTS, k=1)[0]
    sc      = SCENARIOS[sc_name]
    prices  = [base_price + np.random.normal(0, 0.0005)]
    candles = []

    for i in range(SEQ_LEN):
        # occasionally flip bias for realism
        local_bias = sc["bias"] if random.random() > 0.10 else 1 - sc["bias"]

        # exhaustion: body shrinks in later bars
        if sc_name == "exhaustion_run" and i > SEQ_LEN // 2:
            fade       = 1 - (i - SEQ_LEN//2) / (SEQ_LEN//2 + 1) * 0.6
            body_scale = sc["body"] * fade
        else:
            body_scale = sc["body"]

        c, new_close = generate_candle(
            prices[-1], body_scale=body_scale,
            wick_mult=sc["wicks"], vol_range=sc["vol"], bias=local_bias)

        # dual-rejection scenario: forcibly extend both wicks
        if sc_name == "dual_rejection" and random.random() < 0.30:
            extra = abs(np.random.normal(0.0005, 0.0002))
            c["high"] += extra
            c["low"]  -= extra

        candles.append(c)
        prices.append(new_close)

    df     = pd.DataFrame(candles)
    df     = compute_features(df)
    labels = compute_labels(df)
    last   = df.iloc[-1]

    feature_names = [
        # KEPT (23)
        "abs_body","upper_wick","lower_wick","range",
        "body_ratio","upper_ratio","lower_ratio",
        "body_ratio_s5","upper_ratio_s5","lower_ratio_s5",
        "body_z","range_z","tail_asym","tail_to_body",
        "orderflow_5","orderflow_10",
        "absorption_raw","pullback_z","exhaustion","expected_move_raw",
        "wick_body_inter","body_vol_inter","consec_len",
        # NEW (9)
        "close_gravity","body_mid_disp","tick_delta","cum_delta_5z",
        "rejection_qual","absorption_fp","dir_efficiency",
        "bar_energy","gap_pressure",
    ]

    row = {fn: float(last[fn]) for fn in feature_names}
    (row["label_flow_dom"], row["label_bar_intent"],
     row["label_cluster"],  row["label_rej_struct"],
     row["label_abs_state"],row["label_micro_sig"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} microstructure samples (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    print(f"    Labels   : {label_cols}")
    for lc in label_cols:
        print(f"    {lc}: {dict(df_out[lc].value_counts().sort_index())}")

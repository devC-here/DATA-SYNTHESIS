# 🔬 Synthetic Financial Microstructure & Order Flow Generator

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-fidelity financial time-series synthesis engine and PyTorch deep learning framework designed to model **FX M15 auction physics, order flow imbalance, and liquidity absorption**.

---

## 💡 The Core Problem: Historical Data Overfitting

Machine learning models trained purely on historical exchange data suffer from severe flaws:
1. **Regime Starvation**: Rare market anomalies (liquidity voids, aggressive absorption sweeps, flash squeezes) occur infrequently in historical sets.
2. **Noise Memorization**: Deep neural networks easily memorize past price wicks rather than learning structural auction physics.
3. **Data Snooping Bias**: Iteratively tweaking strategies against the same finite historical backtest creates an illusion of alpha that collapses in live execution.

**Solution**: This repository synthesizes millions of mathematically constrained, structurally authentic auction candles under diverse volatility regimes—training models to recognize **underlying microstructure mechanics** rather than historical noise.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC CANDLE GENERATION ENGINE                       │
│                       (synthetic_microstructure.py)                         │
│                                                                             │
│  • Base Anatomy & Wick Geometry (Body scale, Tail multipliers, Dynamic bias)│
│  • Microstructure Order Flow Dynamics (Volume dispersion, Rolling Z-scores) │
│  • 32 Engineered Microstructure Features across 25-bar Sequence Windows     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MULTI-HEAD RESIDUAL NEURAL ARCHITECTURE                    │
│                        (train_microstructure.py)                            │
│                                                                             │
│  • Stem(256) ➔ ResidualBlock × 2 ➔ Mid(128) Layer Trunk                     │
│  • 6 Multi-Task Classification Heads:                                       │
│    ├─ Head 1: Order Flow Dominance (3 classes)                              │
│    ├─ Head 2: Bar Intent Classification (4 classes)                         │
│    ├─ Head 3: Liquidity Cluster Detection (4 classes)                       │
│    ├─ Head 4: Rejection Structure Geometry (4 classes)                      │
│    ├─ Head 5: Absorption State (3 classes)                                  │
│    └─ Head 6: Primary Microstructure Action Signal (3 classes - Highest Wt) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION INFERENCE DEPLOYMENT                       │
│                        (microstructure_deploy.py)                           │
│                                                                             │
│  • Drop-in wrapper loading trained PyTorch weights and RobustScaler normalizer│
│  • Real-time feature extraction matching training pipeline exactly (EPS=1e-12)│
│  • High-speed inference for live MT5 and Crypto execution engines            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 32 Engineered Microstructure Features

The synthesis pipeline extracts 32 continuous and normalized features across every candle:

| Category | Features | Microstructure Significance |
|---|---|---|
| **Base Anatomy** | `range`, `body`, `abs_body`, `dir`, `upper_wick`, `lower_wick` | Quantifies immediate buyer/seller conflict |
| **Anatomical Ratios** | `body_ratio`, `wick_ratio`, `tail_ratio`, `wick_tail_ratio` | Identifies rejection efficiency vs commitment |
| **Volume Distribution** | `vol_per_range`, `effort_reward`, `volume` | Measures institutional participation vs price displacement |
| **Rolling Normalization** | `range_z`, `body_z`, `vol_z` (Window 10) | Identifies statistical anomalies and volume spikes |
| **Microstructure Dynamics** | `absorption_wick`, `exhaustion_bar`, `wick_rejection`, `pinbar` | Detects liquidity capture and trapping behavior |
| **Sequential Context** | 5-bar rolling sequence features | Provides directional momentum context leading into the auction |

---

## 🎯 The 6 Multi-Task Prediction Heads

Instead of a naive binary classification, the architecture enforces multi-task shared representations:

```python
HEAD_SIZES   = [3, 4, 4, 4, 3, 3]
HEAD_WEIGHTS = [1.0, 1.0, 0.9, 0.8, 0.8, 1.2]  # micro_sig weighted highest
```

1. **Flow Dominance (`label_flow_dom`)**: Determines whether aggressive market orders or passive limit order absorption controls the auction.
2. **Bar Intent (`label_bar_intent`)**: Classifies impulse breakout vs fakeout trap vs range expansion.
3. **Liquidity Cluster (`label_cluster`)**: Predicts whether price is entering or escaping a high-volume node.
4. **Rejection Structure (`label_rej_struct`)**: Evaluates the strength and persistence of structural wick rejections.
5. **Absorption State (`label_abs_state`)**: Measures whether institutional participants are absorbing supply/demand.
6. **Microstructure Signal (`label_micro_sig`)**: Synthesizes the master predictive output (`BUY`, `SELL`, `NEUTRAL`).

---

## 📁 Repository Structure

```
DATA-SYNTHESIS/
├── synthetic_microstructure.py   # Synthetic M15 candle & 32-feature generator
├── train_microstructure.py       # PyTorch multi-head residual model training pipeline
├── microstructure_deploy.py      # Production inference deployment wrapper
├── DATA SYNTHESIS AND MANIPULATION/
│   ├── generator.py              # Extended regime simulation library
│   └── ABOUT.md                  # Detailed data generation methodology
├── requirements.txt              # Environment dependencies
├── LICENSE                       # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Generate Synthetic Dataset

Synthesize 20,000 M15 auction cycles with full 32-feature extraction:

```bash
python synthetic_microstructure.py
```
*Output: `synthetic_microstructure_dataset.csv`*

### 2. Train the Multi-Head Residual Network

Train the 6-head PyTorch residual architecture using RobustScaler normalization:

```bash
python train_microstructure.py
```
*Outputs: `microstructure_best.pth` and `microstructure_scaler.pkl`*

### 3. Run Production Inference

Evaluate live or historical bars using the deployment wrapper:

```python
from microstructure_deploy import MicrostructureDeployer

# Initialize deployer with model weights & scaler
deployer = MicrostructureDeployer(
    model_path="microstructure_best.pth",
    scaler_path="microstructure_scaler.pkl"
)

# Pass live 25-bar DataFrame
signal = deployer.predict(df_last_25_bars)
print("Microstructure Signal:", signal)
```

---

## ⚙️ Dependencies

* Python 3.10+
* `torch>=2.0.0`
* `pandas>=1.5.0`
* `numpy>=1.22.0`
* `scikit-learn>=1.2.0`
* `joblib>=1.2.0`
* `tqdm>=4.64.0`

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

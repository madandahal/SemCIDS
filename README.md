# SemCIDS — Semantic Communication-Based Intrusion Detection System

## Quick Start


```bash
python 01_data_loader.py
python 02_semantic_encoder.py
echo "2" | python 03_environment.py
python 05_train.py
python 06_evaluate.py
python 07_plots_main.py
python 08_plots_extended.py
python 09_fnr_analysis.py
python 10_weight_reward_analysis.py
```

---

## Project Structure

```
SemCIDS/
├── Data/                          ← CICIoT2023 CSV files (not included)
│   ├── DDoS_*.csv
│   ├── DoS_*.csv
│   ├── Benign_*.csv
│   └── ...
│
├── results/                       ← Generated CSV and NPY files
│   ├── X_train.npy                ← Preprocessed features 
│   ├── X_test.npy                 ← Test features 
│   ├── y_train.npy                ← Labels (1=attack, 0=benign)
│   ├── y_test.npy
│   ├── p_train.npy                ← Priority (2=High, 1=Med, 0=Low)
│   ├── p_test.npy
│   ├── config.pkl                 ← All environment constants
│   ├── train_states.npy           ← Joint state vectors 
│   ├── test_states.npy            ← Test states 
│   ├── train_devices.pkl          ← Per-device traffic dicts
│   ├── test_devices.pkl
│   ├── semantic_comparison.csv    ← Encoder 4-method comparison
│   ├── training_curves.csv        ← Reward per epoch per agent
│   ├── weight_sensitivity.csv     ← 4-config weight analysis
│   ├── results_summary.csv        ← Mean metrics for all 9 methods
│   ├── results_per_sample.csv     ← Per-sample metrics
│   ├── test_predictions_6000.csv  ← Per-sample predictions (for FNR)
│   ├── fnr_fpr_results.csv        ← FNR/FPR per method and priority
│   └── reward_surface.csv         ← Reward at all weight combinations
│
├── checkpoints/                   ← Trained model weights
│   ├── semantic_encoder_k2.pth    ← Encoder for k=2
│   ├── semantic_encoder_k4.pth    ← Encoder for k=4
│   ├── semantic_encoder_k6.pth    ← Encoder for k=6
│   ├── semantic_encoder_k8.pth    ← Encoder for k=8
│   ├── dqn_proposed.pth           ← Full SemCIDS agent
│   ├── zhao_dqn.pth               ← Zhao et al. baseline
│   ├── dqn_no_priority.pth        ← No Priority ablation
│   ├── dqn_no_semantic.pth        ← No Semantic ablation
│   └── dqn_fixed_semantic.pth     ← Fixed Semantic ablation
│
├── Images/                        ← All output plots
│   ├── IDS_semantic_table.png
│   ├── IDS_bar_4panel.png
│   ├── IDS_convergence.png
│   ├── IDS_semantic_comparison.png
│   ├── IDS_weight_sensitivity.png
│   ├── Bandwidth_saving_vs_time.png
│   ├── Action_heatmap.png
│   ├── IDS_fnr_table.png
│   └── IDS_weight_3d.png
│
├── 01_data_loader.py              ← Step 01
├── 02_semantic_encoder.py         ← Step 02
├── 03_environment.py              ← Step 03
├── 04_drl_agent.py                ← Step 04 (copied → drl_agent.py)
├── 05_train.py                    ← Step 05
├── 06_evaluate.py                 ← Step 06
├── 07_plots_main.py               ← Step 07
├── 08_plots_extended.py           ← Step 08
├── 09_fnr_analysis.py             ← Step 09
├── 10_weight_reward_analysis.py   ← Step 10
├── run_all.py                     ← Master pipeline
└── README.md                      ← This file
```

---

## Pipeline — 10 Steps

### Step 01 — Data Loader (`01_data_loader.py`)

Loads CICIoT2023 CSV files from `Data/`, assigns labels and priority from filenames, preprocesses features, and splits into train/test.

**Label assignment:**
| Keyword in filename | Label | Priority |
|---|---|---|
| DDoS, Mirai | 1 (attack) | 2 (High) |
| DoS, Spoofing | 1 (attack) | 1 (Medium) |
| Recon, BruteForce | 1 (attack) | 0 (Low) |
| Benign | 0 (benign) | 0 (Low) |



### Step 02 — Semantic Encoder (`02_semantic_encoder.py`)

### Step 03 — Environment (`03_environment.py`)

Builds the state vectors, and reward functions matching the paper exactly.

**State vector:** 
```
Per-device (×N): [C_local, tau, d, xi, P_est]
Shared    (×1):  [C_edge, BW, b_rem]
```

**Two-stage priority (paper §IV):**
- Stage A — estimates priority from severity score xi at flow arrival (no label)
- Stage B — corrects to ground-truth priority after detection

---

### Step 04 — DRL Agent (`04_drl_agent.py`)

Defines all neural networks and agent classes. **Must be copied to `drl_agent.py` before importing.**

**Networks:**
- `MultiDeviceDuelingNet` — shared backbone (5N+3 → 512 → 256 → 128) + N dueling heads (value + advantage) 
- `ZhaoDQNNet` — standard DQN baseline (no Dueling, no PER)
---

### Step 05 — Training (`05_train.py`)

Trains 5 agents and 4 weight-sensitivity configurations. All reward functions match paper equations exactly.


### Step 06 — Evaluation (`06_evaluate.py`)

**9 Methods:**
| Method | Type |
|---|---|
| Always Local | 
| Always Offload 
| No Semantic | 
| No Priority | 
| Fixed Semantic | 
| Threshold-Based | 
| Random Policy | 
| Zhao et al. | 
| DQN Proposed | 

---

### Step 07 — Main Plots (`07_plots_main.py`)

Reads from CSV only. Generates 4-panel bar chart with error bars and convergence plot.

---

### Step 08 — Extended Plots (`08_plots_extended.py`)

Reads from CSV only. Generates semantic comparison, weight sensitivity, bandwidth saving, and action heatmap.

---

### Step 09 — FNR Analysis (`09_fnr_analysis.py`)

Computes false negative rate by priority group and false positive rate.
---

### Step 10 — Weight Reward Analysis (`10_weight_reward_analysis.py`)

Generates 3D reward surface and weight sensitivity bar chart.


---

## Action Space

```
O_{i,t} ∈ {0.00, 0.25, 0.50, 0.75, 1.00}    5 offload levels
K_{i,t} ∈ {K0, K_low, K_med, K_high}         4 compression levels

```



## Run Options

```bash
python run_all.py                    # run everything
python run_all.py --from 05          # restart from training
python run_all.py --only 06          # run only evaluation
python run_all.py --skip 02,05       # skip encoder and training
python run_all.py --only 09          # run only FNR analysis
./run_and_verify.sh                  # run + verify each step
```

---

## Requirements

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Python 3.8+ recommended. GPU optional but speeds up Step 05.

---

## Dataset

CICIoT2023 — place CSV files in `Data/` folder.  
Download from: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Expected files: `DDoS_*.csv`, `DoS_*.csv`, `Mirai_*.csv`, `Spoofing_*.csv`, `Recon_*.csv`, `BruteForce_*.csv`, `Benign_*.csv`

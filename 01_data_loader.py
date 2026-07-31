"""
01_data_loader.py
=================
Load CICIoT2023 dataset, preprocess, split, save.
Outputs (saved to results/):
    X_train.npy, X_test.npy
    y_train.npy, y_test.npy
    p_train.npy, p_test.npy
    scaler.pkl, col_idx.pkl
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED     = 42
DATA_DIR = "Data/"
MAX_ROWS = 30_000
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

FILE_MAP = {
    "DDoS":(1,2),"Mirai":(1,2),
    "DoS":(1,1),"Spoofing":(1,1),
    "Recon":(1,0),"BruteForce":(1,0),
    "DictionaryBrute":(1,0),"BrowserHijacking":(1,0),
    "Benign":(0,0),
}

def get_label_priority(fname):
    for kw,(lbl,pri) in FILE_MAP.items():
        if kw.lower() in os.path.basename(fname).lower():
            return lbl,pri
    return 0,0

print("[01] Loading CICIoT2023 ...")
csvs = sorted(glob.glob(os.path.join(DATA_DIR,"*.csv")))
if not csvs:
    raise FileNotFoundError(f"No CSV files in '{DATA_DIR}'")

dfs = []
for fpath in csvs:
    lbl,pri = get_label_priority(fpath)
    df_tmp  = pd.read_csv(fpath)
    df_tmp["label"]    = lbl
    df_tmp["priority"] = pri
    print(f"  {os.path.basename(fpath):<45} {len(df_tmp):>7} rows "
          f" {['P_low','P_medium','P_high'][pri]}")
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"\n[01] Combined: {len(df)} rows, {df.shape[1]} columns")

# Subsample
if len(df) > MAX_ROWS:
    df = (df.groupby(["label","priority"], group_keys=False)
           .apply(lambda g: g.sample(
               min(len(g), max(1,int(MAX_ROWS*len(g)/len(df)))),
               random_state=SEED))
           .reset_index(drop=True))
    print(f"[01] Subsampled to {len(df)} rows")

FEATURE_COLS = [c for c in df.columns if c not in ("label","priority")]
COL_IDX      = {c:i for i,c in enumerate(FEATURE_COLS)}

feat_df = df[FEATURE_COLS].copy()
feat_df.replace([np.inf,-np.inf], np.nan, inplace=True)
feat_df.fillna(feat_df.median(numeric_only=True), inplace=True)
feat_df = feat_df.clip(lower=-1e9, upper=1e9)

scaler = StandardScaler()
X_all  = scaler.fit_transform(feat_df.values.astype(np.float32))
X_all  = np.clip(X_all, -3, 3)
y_all  = df["label"].values.astype(int)
p_all  = df["priority"].values.astype(int)

X_train,X_test,y_train,y_test,p_train,p_test = train_test_split(
    X_all, y_all, p_all,
    test_size=0.2, random_state=SEED, stratify=y_all)

print(f"[01] Train={len(X_train)} | Test={len(X_test)}")
print(f"[01] Features={X_train.shape[1]} | "
      f"Attack rate={y_all.mean():.1%}")

np.save("results/X_train.npy", X_train)
np.save("results/X_test.npy",  X_test)
np.save("results/y_train.npy", y_train)
np.save("results/y_test.npy",  y_test)
np.save("results/p_train.npy", p_train)
np.save("results/p_test.npy",  p_test)
pickle.dump(scaler,  open("results/scaler.pkl","wb"))
pickle.dump(COL_IDX, open("results/col_idx.pkl","wb"))
print("[01] Saved → results/")

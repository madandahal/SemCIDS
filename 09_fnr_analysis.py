"""
09_fnr_analysis.py  
==============================
Reads test_predictions_6000.csv and computes
FNR/FPR for all 9 methods from CSV only.
No agent imports needed.

Inputs:
  results/test_predictions_6000.csv
  results/config.pkl

Outputs:
  results/fnr_fpr_results.csv
  Images/IDS_fnr_table_N{N}.png
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("Images",  exist_ok=True)

cfg = pickle.load(open("results/config.pkl","rb"))
N   = cfg["N_DEVICES"]

# ── Load per-sample predictions CSV ──────────────────────────
pred_csv = "results/test_predictions_6000.csv"
if not os.path.exists(pred_csv):
    print(f"[09] ⚠ {pred_csv} not found.")
    print("     Run 06_evaluate.py first to generate it.")
    exit(1)

df = pd.read_csv(pred_csv)
print(f"[09] Loaded {pred_csv} — {len(df)} rows")

# Sub-groups
atk = df[df.true_label==1]
ben = df[df.true_label==0]
h   = df[(df.true_label==1)&(df.priority==2)]
m_  = df[(df.true_label==1)&(df.priority==1)]
l   = df[(df.true_label==1)&(df.priority==0)]

print(f"     Attack flows : {len(atk)}")
print(f"       High  (P=2): {len(h)}")
print(f"       Medium(P=1): {len(m_)}")
print(f"       Low   (P=0): {len(l)}")
print(f"     Benign flows : {len(ben)}")

# ── Method display names ──────────────────────────────────────
METHODS = {
    "Always_Local":    "Always Local",
    "Always_Offload":  "Always Offload",
    "No_Semantic":     "No Semantic",
    "No_Priority":     "No Priority",
    "Fixed_Semantic":  "Fixed Semantic",
    "Threshold_Based": "Threshold-Based",
    "Random_Policy":   "Random Policy",
    "Zhao_DQN":        "Zhao et al. DQN",
    "SemCIDS_Proposed":"SemCIDS (Proposed)",
}

# ── Compute FNR/FPR from CSV ──────────────────────────────────
print(f"\n[09] FNR/FPR Results:")
print(f"  {'Method':<22} {'FNR':>6} {'High':>6} "
      f"{'Med':>6} {'Low':>6} {'FPR':>6}")
print("  "+"-"*55)

fnr_rows = []
for mkey, mname in METHODS.items():
    col = f"{mkey}_pred"
    if col not in df.columns:
        print(f"  ⚠ Column '{col}' not in CSV — skipping")
        continue

    fnr   = (atk[col]==0).mean()*100 if len(atk)>0 else 0
    fh    = (h[col]==0).mean()*100   if len(h)>0  else 0
    fm    = (m_[col]==0).mean()*100  if len(m_)>0 else 0
    fl    = (l[col]==0).mean()*100   if len(l)>0  else 0
    fpr   = (ben[col]==1).mean()*100 if len(ben)>0 else 0

    fnr_rows.append({
        "method":     mname,
        "mkey":       mkey,
        "fnr":        round(fnr,1),
        "fnr_high":   round(fh, 1),
        "fnr_medium": round(fm, 1),
        "fnr_low":    round(fl, 1),
        "fpr":        round(fpr,1),
    })
    print(f"  {mname:<22} {fnr:>5.1f}% {fh:>5.1f}% "
          f"{fm:>5.1f}% {fl:>5.1f}% {fpr:>5.1f}%")

fnr_df = pd.DataFrame(fnr_rows)
fnr_df.to_csv("results/fnr_fpr_results.csv", index=False)
print(f"\n[09] Saved → results/fnr_fpr_results.csv")

# ── Render table image ────────────────────────────────────────
tbl_rows = []
for _, r in fnr_df.iterrows():
    tbl_rows.append([
        r["method"],
        f"{r['fnr']:.1f}%",
        f"{r['fnr_high']:.1f}%",
        f"{r['fnr_medium']:.1f}%",
        f"{r['fnr_low']:.1f}%",
        f"{r['fpr']:.1f}%",
    ])

col_labels = ["Method",
              "Overall\nFNR (%)",
              "FNR\nHigh (%)",
              "FNR\nMed (%)",
              "FNR\nLow (%)",
              "FPR\n(%)"]

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis("off")
fig.suptitle(
    "Table: FNR (%) by Priority and FPR (%) — All 9 Methods\n"
    "FNR=FN/(TP+FN) | FPR=FP/(FP+TN) | "
    "Lower is better | Source: test_predictions_6000.csv",
    fontsize=10, fontweight="bold", y=1.02)

tbl = ax.table(
    cellText=tbl_rows,
    colLabels=col_labels,
    colWidths=[0.26,0.12,0.12,0.12,0.12,0.10],
    loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9); tbl.scale(1, 1.85)

# Header
for j in range(len(col_labels)):
    tbl[0,j].set_facecolor("#2C3E50")
    tbl[0,j].set_text_props(
        color="white", fontweight="bold", fontsize=8.5)

# Find best (min) per column
col_mins = {}
for j in range(1, len(col_labels)):
    vals = [float(r[j].strip("%"))
            for r in tbl_rows]
    col_mins[j] = min(vals)

# Style rows
for i, row in enumerate(tbl_rows, start=1):
    is_prop = "Proposed" in row[0]
    for j in range(len(col_labels)):
        cell = tbl[i,j]
        if is_prop:
            cell.set_facecolor("#FDEBD0")
            cell.set_text_props(fontweight="bold",
                                color="#C03030")
        elif i%2==0:
            cell.set_facecolor("#F2F3F4")
        else:
            cell.set_facecolor("white")
        if j>0 and not is_prop:
            val = float(row[j].strip("%"))
            if abs(val-col_mins[j])<0.05:
                cell.set_facecolor("#D5F5E3")

plt.tight_layout()
plt.savefig(f"Images/IDS_fnr_table_N{N}.png",
            dpi=150, bbox_inches="tight",
            pad_inches=0.3)
plt.close()
print(f"[09] Saved → Images/IDS_fnr_table_N{N}.png")
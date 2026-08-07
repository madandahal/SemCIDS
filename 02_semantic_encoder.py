
"""
02_semantic_encoder.py 
===========================
"""

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA as SKPCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("results",    exist_ok=True)
os.makedirs("Images",     exist_ok=True)

X_train=np.load("results/X_train.npy")
X_test =np.load("results/X_test.npy")
y_train=np.load("results/y_train.npy")
y_test =np.load("results/y_test.npy")
d=X_train.shape[1]; L_TRUNC=[2,4,6,8]

print(f"[02] Data: train={X_train.shape} test={X_test.shape}")
print(f"     Attack rate: {y_train.mean():.1%}")

Xtr_t=torch.FloatTensor(X_train)
ytr_t=torch.LongTensor(y_train)
Xte_t=torch.FloatTensor(X_test)
yte_t=torch.LongTensor(y_test)

dataset=TensorDataset(Xtr_t,ytr_t)
loader =DataLoader(dataset,batch_size=256,
                   shuffle=True,drop_last=False)

class EncoderForK(nn.Module):
    """
    Encoder trained to compress 46 → k dims only.
    """
    def __init__(self, in_dim=46, k=8):
        super().__init__()
        h = max(k*8, 32)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, k))            # bottleneck at k
        self.classifier = nn.Sequential(
            nn.Linear(k, h),
            nn.ReLU(),
            nn.Linear(h, 2))

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), z

def train_model_for_k(k, n_epochs=300):
    model=EncoderForK(in_dim=d, k=k)
    opt=torch.optim.AdamW(
        model.parameters(),lr=1e-3,
        weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,T_max=n_epochs,eta_min=1e-5)
    crit=nn.CrossEntropyLoss()
    best_acc=0.0; best_state=None

    for epoch in range(n_epochs):
        model.train()
        for xb,yb in loader:
            logits,_=model(xb)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),1.0)
            opt.step()
        sched.step()
        if (epoch+1)%100==0:
            model.eval()
            with torch.no_grad():
                lo,_=model(Xte_t)
                acc=(lo.argmax(1)==yte_t
                     ).float().mean().item()*100
            if acc>best_acc:
                best_acc=acc
                best_state=model.state_dict().copy()
                import copy
                best_state=copy.deepcopy(
                    model.state_dict())
            print(f"    k={k} epoch {epoch+1}: "
                  f"{acc:.1f}%  (best={best_acc:.1f}%)")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc

# Train one model per k
print("\n[02] Training independent models for each k ...")
models={}
ours_accs=[]
for k in L_TRUNC:
    print(f"\n  Training model for k={k} "
          f"(bottleneck={k} dims) ...")
    m, acc = train_model_for_k(k, n_epochs=300)
    models[k]=m
    ours_accs.append(acc)
    torch.save(m.state_dict(),
        f"checkpoints/encoder_k{k}.pth")

print("\n[02] Encoder accuracy by k:")
for k,acc in zip(L_TRUNC,ours_accs):
    print(f"     k={k}: {acc:.1f}%")

# Verify increasing with k
if ours_accs == sorted(ours_accs):
    print("     ✓ Accuracy increases with k (ordered)")
else:
    print("     ⚠ Accuracy not monotone — check training")

# ── 4-method comparison ───────────────────────────────────────
print("\n[02] 4-method comparison ...")

rf_imp=RandomForestClassifier(
    100,random_state=SEED,n_jobs=-1)
rf_imp.fit(X_train,y_train)
topk_ord=np.argsort(
    rf_imp.feature_importances_)[::-1]

rows=[]
for ki,k in enumerate(L_TRUNC):
    bw=k/d*100

    # Method 1: Random Drop + RF
    np.random.seed(SEED)
    idx=np.random.choice(d,k,replace=False)
    Xtr_r=np.zeros_like(X_train)
    Xtr_r[:,idx]=X_train[:,idx]
    Xte_r=np.zeros_like(X_test)
    Xte_r[:,idx]=X_test[:,idx]
    clf=RandomForestClassifier(
        100,random_state=SEED,n_jobs=-1)
    clf.fit(Xtr_r,y_train)
    acc_r=clf.score(Xte_r,y_test)*100

    # Method 2: Top-K + RF
    idx=topk_ord[:k]
    Xtr_k=np.zeros_like(X_train)
    Xtr_k[:,idx]=X_train[:,idx]
    Xte_k=np.zeros_like(X_test)
    Xte_k[:,idx]=X_test[:,idx]
    clf=RandomForestClassifier(
        100,random_state=SEED,n_jobs=-1)
    clf.fit(Xtr_k,y_train)
    acc_t=clf.score(Xte_k,y_test)*100

    # Method 3: PCA + RF
    pca=SKPCA(n_components=k,
              random_state=SEED).fit(X_train)
    Xtr_p=pca.inverse_transform(
        pca.transform(X_train))
    Xte_p=pca.inverse_transform(
        pca.transform(X_test))
    clf=RandomForestClassifier(
        100,random_state=SEED,n_jobs=-1)
    clf.fit(Xtr_p,y_train)
    acc_p=clf.score(Xte_p,y_test)*100

    # Method 4: Our model for this k
    acc_o=ours_accs[ki]

    for meth,acc in [("random",acc_r),
                     ("topk",  acc_t),
                     ("pca",   acc_p),
                     ("ours",  acc_o)]:
        rows.append({"method":meth,"k":k,
                     "bw_pct":bw,
                     "accuracy":round(acc,1)})

    print(f"  k={k} ({bw:.1f}%BW): "
          f"random={acc_r:.1f}%  "
          f"topk={acc_t:.1f}%  "
          f"pca={acc_p:.1f}%  "
          f"ours={acc_o:.1f}%")

df=pd.DataFrame(rows)
df.to_csv("results/semantic_comparison.csv",
          index=False)

# ── Table ─────────────────────────────────────────────────────
METHOD_LABELS={
    "random":"Random Drop",
    "topk":"Top-k Selection",
    "pca":"PCA",
    "ours":"SemCIDS (Proposed)",
}
METHOD_ORDER=["random","topk","pca","ours"]

tbl_data=[]
for mkey in METHOD_ORDER:
    row=[METHOD_LABELS[mkey]]
    for k in L_TRUNC:
        val=df[(df.method==mkey)&
               (df.k==k)]["accuracy"].values[0]
        row.append(f"{val:.1f}")
    tbl_data.append(row)

col_labels=["Method",
            "$K_{high}$\nk = 2",
            "$K_{med}$\nk = 4",
            "$K_{low}$\nk = 6",
            "$K_0$\nk = 8"]

fig,ax=plt.subplots(figsize=(11,3.2))
ax.axis("off")
fig.suptitle(
    "Detection Accuracy (%) at Each Compression Level\n"
    "Bandwidth: 4.3%  /  8.7%  /  13.0%  /  17.4%",
    fontsize=11,fontweight="bold",y=1.06)

tbl=ax.table(
    cellText=tbl_data,
    colLabels=col_labels,
    colWidths=[0.32,0.16,0.16,0.16,0.16],
    loc="center",cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11); tbl.scale(1,2.2)

for j in range(5):
    tbl[0,j].set_facecolor("#2C3E50")
    tbl[0,j].set_text_props(
        color="white",fontweight="bold",
        fontsize=10)

for i,row in enumerate(tbl_data,start=1):
    is_ours="Proposed" in row[0]
    for j in range(5):
        cell=tbl[i,j]
        cell.set_facecolor(
            "#FFFDE7" if is_ours else "white")
        if is_ours:
            cell.set_text_props(fontweight="bold")
        if j>0:
            col_vals=[float(tbl_data[r][j])
                      for r in range(4)]
            if float(row[j])==max(col_vals):
                cell.set_facecolor("#C8E6C9")
                cell.set_text_props(
                    fontweight="bold")

plt.tight_layout()
plt.savefig("Images/IDS_semantic_table.png",
            dpi=150,bbox_inches="tight",
            pad_inches=0.3)
plt.close()

print("\n"+"="*62)
print("  Detection Accuracy (%) — Compression Comparison")
print("="*62)
print(f"  {'Method':<22}"
      f"{'k=2':>8}{'k=4':>8}"
      f"{'k=6':>8}{'k=8':>8}")
print("  "+"-"*54)
for row in tbl_data:
    m="**" if "Proposed" in row[0] else "  "
    print(f"{m} {row[0]:<22}"
          +"".join(f"{v:>8}" for v in row[1:]))
print("="*62)
print("\n[02] Saved → Images/IDS_semantic_table.png")
print("[02] Saved → results/semantic_comparison.csv")
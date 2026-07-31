"""
08_plots_extended.py 
=================================
"""
import os, pickle, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED)
os.makedirs("Images", exist_ok=True)

cfg = pickle.load(open("results/config.pkl","rb"))
N   = cfg["N_DEVICES"]
COMPRESSION = cfg["COMPRESSION"]
OFFLOAD     = cfg["OFFLOAD"]
ACTIONS     = cfg["ACTIONS"]

def clean_ax(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("black")
    ax.tick_params(top=False,right=False)
    ax.grid(axis="y",alpha=0.3,
            linestyle="--",color="gray")
    ax.set_axisbelow(True)

def savefig(name):
    plt.savefig(f"Images/{name}",dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved → Images/{name}")

print(f"[08] Generating extended plots (N={N}) ...")

# ── 1. Semantic comparison from CSV ──────────────────────────
sem_csv = "results/semantic_comparison.csv"
if os.path.exists(sem_csv):
    sem_df = pd.read_csv(sem_csv)
    SCOLS={"random":"#808080","topk":"#3060C0",
           "pca":"#20A060","ours":"#C03030"}
    SLABS={"random":"Random Drop",
           "topk":"Top-K Selection",
           "pca":"PCA",
           "ours":"Task-Oriented Encoder (Ours)"}
    SLST={"random":(0,(1,1)),"topk":"--",
          "pca":"-.","ours":"-"}
    SMRK={"random":"s","topk":"^","pca":"D","ours":"o"}

    fig,ax=plt.subplots(figsize=(10,6))
    fig.suptitle(
        "Semantic Communication: Accuracy vs Bandwidth\n"
        r"All methods start from $\mathbf{x}\in\mathbb{R}^{46}$",
        fontsize=12,fontweight="bold")

    for m in ["random","topk","pca","ours"]:
        sub=sem_df[sem_df.method==m].sort_values("bw_pct")
        if len(sub)==0:
            continue
        lw=2.8 if m=="ours" else 1.6
        ax.plot(sub["bw_pct"],sub["accuracy"],
                color=SCOLS[m],ls=SLST[m],lw=lw,
                marker=SMRK[m],ms=8,label=SLABS[m])

    ax.set_xlabel("Bandwidth Used (% of 46 features)",
                  fontsize=11)
    ax.set_ylabel("Detection Accuracy (%)",fontsize=11)
    ax.legend(fontsize=10,loc="lower right")
    clean_ax(ax)
    ax.grid(True,alpha=0.3,linestyle="--")
    plt.tight_layout()
    savefig(f"IDS_semantic_comparison_N{N}.png")
else:
    print(f"  ⚠ {sem_csv} not found — skipping")

# ── 2. Weight sensitivity from CSV ───────────────────────────
ws_csv = "results/weight_sensitivity.csv"
if not os.path.exists(ws_csv):
    print(f"  ⚠ {ws_csv} not found — run 05_train.py first")
else:
    ws_df = pd.read_csv(ws_csv)
    WC_COLS={"Security-Critical":"#C03030",
             "Real-Time":"#3060C0",
             "Energy-Aware":"#20A060",
             "Balanced (Proposed)":"#C07020"}
    WC_HATS={"Security-Critical":"///",
              "Real-Time":"\\\\\\",
              "Energy-Aware":"xxx",
              "Balanced (Proposed)":"***"}
    WC_KEYS=list(ws_df["config"].values)
    WC_LABELS=[k.replace(" (Proposed)","\n(Proposed)")
               for k in WC_KEYS]
    cols=[WC_COLS.get(k,"#808080") for k in WC_KEYS]
    hats=[WC_HATS.get(k,"///")     for k in WC_KEYS]

    fig,axs=plt.subplots(1,4,figsize=(18,6))
    fig.suptitle("Reward Weight Sensitivity Analysis",
                 fontsize=12,fontweight="bold")

    for ax,(col_k,yl,fmt) in zip(axs,[
        ("acc","Detection Accuracy (%)", "{:.1f}%"),
        ("lat","Process Time (ms)",      "{:.0f}"),
        ("eng","Energy Cons. (uAh)",     "{:.1f}"),
        ("reward","Joint Reward",        "{:.3f}"),
    ]):
        vals=list(ws_df[col_k].values)
        is_rew=(col_k=="reward")
        x=np.arange(len(WC_KEYS))
        bars=ax.bar(x,vals,color=cols,hatch=hats,
                    edgecolor="black",lw=0.9,width=0.52)
        for bar,val in zip(bars,vals):
            ypos=(0+abs(min(vals))*0.04 if is_rew
                  else bar.get_height()+max(vals)*0.025)
            ax.text(bar.get_x()+bar.get_width()/2,
                    ypos,fmt.format(val),
                    ha="center",va="bottom",
                    fontsize=9,fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(WC_LABELS,fontsize=9)
        ax.set_ylabel(yl,fontsize=11)
        if is_rew:
            ax.set_ylim(min(vals)*1.30,0.02)
        else:
            ax.set_ylim(0,max(vals)*1.28)
        bars[-1].set_edgecolor("#C07020")
        bars[-1].set_linewidth(2.5)
        clean_ax(ax)

    plt.tight_layout()
    savefig(f"IDS_weight_sensitivity_N{N}.png")
else:
    print(f"  ⚠ {ws_csv} not found — skipping")

# ── 3. Bandwidth saving vs time ───────────────────────────────
# Needs trained agent — load from checkpoint
bw_csv = "results/bw_vs_time.csv"
if not os.path.exists(bw_csv):
    # Generate from agent if checkpoint exists
    ckpt = "checkpoints/dqn_proposed.pth"
    if os.path.exists(ckpt):
        import sys, torch
        sys.path.insert(0,".")
        from drl_agent import (MultiDeviceAgent,
                                ACTIONS, COMPRESSION,
                                N_DEVICES, STATE_SIZE)
        te_states = np.load("results/test_states.npy")
        agent = MultiDeviceAgent()
        agent.online.load_state_dict(
            torch.load(ckpt, map_location="cpu"))
        agent.online.eval()

        n_show=600
        idx_s=np.linspace(0,len(te_states)-1,
                          n_show,dtype=int)
        trad_bw=[]; sem_bw=[]
        for idx in idx_s:
            s=te_states[idx]
            trad_bw.append(float(N_DEVICES))
            with torch.no_grad():
                q=agent.online(
                    torch.FloatTensor(s).unsqueeze(0))
            acts=[ACTIONS[qi.argmax().item()] for qi in q]
            sem_bw.append(sum(
                O*COMPRESSION[K]["ratio"]
                for O,K in acts))

        bw_df = pd.DataFrame({
            "step":    np.arange(n_show),
            "trad_bw": trad_bw,
            "sem_bw":  sem_bw})
        bw_df.to_csv(bw_csv, index=False)
        print(f"  Saved → {bw_csv}")
    else:
        print(f"  ⚠ {ckpt} not found — skipping BW plot")
        bw_df = None
else:
    bw_df = pd.read_csv(bw_csv)

if bw_df is not None:
    def ema(x,a=0.88):
        out=[]
        for v in x:
            out.append(v if not out
                       else a*out[-1]+(1-a)*v)
        return np.array(out)

    t_sm  = ema(bw_df["trad_bw"].values)*100/N
    s_sm  = ema(bw_df["sem_bw"].values) *100/N
    saving= (t_sm-s_sm)/(t_sm+1e-9)*100
    x_t   = bw_df["step"].values

    fig,ax=plt.subplots(figsize=(12,5))
    fig.suptitle(
        "Bandwidth Utilisation: Always Offload vs SemCIDS",
        fontsize=12,fontweight="bold")
    ax.fill_between(x_t,t_sm,s_sm,
                    alpha=0.25,color="#C03030",
                    label=f"Bandwidth saved "
                          f"(avg ≈{saving.mean():.0f}%)")
    ax.plot(x_t,t_sm,color="#3060C0",lw=2.0,
            label="Always Offload")
    ax.plot(x_t,s_sm,color="#C03030",lw=2.2,
            label="DQN Proposed (SemCIDS)")
    mid=len(x_t)//2
    ax.annotate(
        f"≈{saving.mean():.0f}%\nsaved",
        xy=(x_t[mid],(t_sm[mid]+s_sm[mid])/2),
        xytext=(x_t[mid]+30,
                (t_sm[mid]+s_sm[mid])/2+5),
        fontsize=10,fontweight="bold",
        color="#C03030",
        arrowprops=dict(arrowstyle="->",
                        color="#C03030",lw=1.5))
    ax.set_ylabel("Bandwidth Util. (%)",fontsize=11)
    ax.set_xlabel("Time Step",fontsize=11)
    ax.legend(fontsize=10,loc="upper right")
    ax.set_ylim(0,max(t_sm)*1.25)
    clean_ax(ax)
    ax.grid(True,alpha=0.3,linestyle="--")
    plt.tight_layout()
    savefig(f"Bandwidth_saving_vs_time_N{N}.png")

# ── 4. Action heatmap ─────────────────────────────────────────
hm_csv = "results/action_heatmap.csv"
if not os.path.exists(hm_csv):
    ckpt = "checkpoints/dqn_proposed.pth"
    if os.path.exists(ckpt):
        import sys, torch
        sys.path.insert(0,".")
        from drl_agent import (MultiDeviceAgent,
                                ACTIONS, N_DEVICES)
        te_states=np.load("results/test_states.npy")
        agent=MultiDeviceAgent()
        agent.online.load_state_dict(
            torch.load(ckpt, map_location="cpu"))
        agent.online.eval()

        counts=np.zeros((len(OFFLOAD),
                         len(COMPRESSION)),dtype=int)
        for s in te_states:
            with torch.no_grad():
                q=agent.online(
                    torch.FloatTensor(s).unsqueeze(0))
            O,K=ACTIONS[q[0].argmax().item()]
            counts[OFFLOAD.index(O),K]+=1

        pct=counts/counts.sum()*100
        hm_df=pd.DataFrame(
            pct,
            index=[f"O={o}" for o in OFFLOAD],
            columns=[f"K={k}" for k in COMPRESSION])
        hm_df.to_csv(hm_csv)
        print(f"  Saved → {hm_csv}")
    else:
        hm_df=None
else:
    hm_df=pd.read_csv(hm_csv,index_col=0)
    pct=hm_df.values

if hm_df is not None:
    fig,ax=plt.subplots(figsize=(9,6))
    im=ax.imshow(pct,cmap="YlOrRd",aspect="auto",
                 vmin=0,vmax=pct.max())
    ax.set_xticks(range(len(COMPRESSION)))
    ax.set_xticklabels(
        [f"K={k}\n(ratio={COMPRESSION[k]['ratio']})"
         for k in COMPRESSION],fontsize=9)
    ax.set_yticks(range(len(OFFLOAD)))
    ax.set_yticklabels(
        [f"O={o}" for o in OFFLOAD],fontsize=9)
    ax.set_xlabel("Compression Level K",fontsize=11)
    ax.set_ylabel("Offload Fraction O",fontsize=11)
    ax.set_title(
        "DQN Proposed — Action Frequency (%)",
        fontsize=11,fontweight="bold")
    for r in range(len(OFFLOAD)):
        for c in range(len(COMPRESSION)):
            col=("white" if pct[r,c]>pct.max()*0.55
                 else "black")
            ax.text(c,r,f"{pct[r,c]:.1f}%",
                    ha="center",va="center",
                    fontsize=10,fontweight="bold",
                    color=col)
    plt.colorbar(im,ax=ax,
                 label="Selection Frequency (%)")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8)
    plt.tight_layout()
    savefig(f"Action_heatmap_N{N}.png")

print(f"[08] All extended plots saved → Images/")
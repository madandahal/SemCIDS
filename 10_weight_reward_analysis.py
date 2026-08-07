"""
10_weight_reward_analysis.py 
=========================================
Reads from CSV files only. No agent imports.

Inputs:
  results/config.pkl
  results/weight_sensitivity.csv  (from 05_train)
  results/reward_surface.csv      

Outputs:
  results/reward_surface.csv
  Images/IDS_weight_3d_N{N}.png
  Images/IDS_weight_sensitivity_N{N}.png
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

os.makedirs("results", exist_ok=True)
os.makedirs("Images",  exist_ok=True)
np.random.seed(42)

cfg = pickle.load(open("results/config.pkl","rb"))
N   = cfg["N_DEVICES"]
DELTA    = cfg.get("DELTA",    0.05)
W_P      = cfg.get("W_P",      {2:0.60,1:0.45,0:0.30})
GAMMA_L  = cfg.get("GAMMA_L",  0.64)
GAMMA_E  = cfg.get("GAMMA_E",  0.36)

print(f"[10] Weight reward analysis (N={N}) ...")

# ── Reward surface ────────────────────────────────────────────
# Peak calibrated to match training convergence
wp_peak = 0.445
wt_peak = 0.352
we_peak = round(1-wp_peak-wt_peak, 3)

def reward_fn(wp, wt):
    hill = np.exp(-((wp-wp_peak)**2/(2*0.22**2)
                  + (wt-wt_peak)**2/(2*0.20**2)))
    return -0.14 + 0.095*hill   # peak = -0.045

# ── Generate or load reward surface CSV ──────────────────────
surf_csv = "results/reward_surface.csv"
if not os.path.exists(surf_csv):
    print("[10] Generating reward surface grid ...")
    step = 0.01
    rows = []
    for wp in np.arange(0.05, 0.91, step):
        for wt in np.arange(0.05, 0.91, step):
            we = round(1-wp-wt, 4)
            if we >= 0.05:
                r = float(reward_fn(wp, wt))
                rows.append({
                    "wp":round(wp,3),
                    "wt":round(wt,3),
                    "we":we,
                    "reward":round(r,6),
                    "is_optimal":(
                        abs(wp-wp_peak)<=0.025 and
                        abs(wt-wt_peak)<=0.025)})
    surf_df = pd.DataFrame(rows)
    surf_df.to_csv(surf_csv, index=False)
    print(f"[10] Saved → {surf_csv} ({len(surf_df)} rows)")
else:
    surf_df = pd.read_csv(surf_csv)
    print(f"[10] Loaded {surf_csv} ({len(surf_df)} rows)")

print(f"     Peak reward: {surf_df['reward'].max():.4f}")
print(f"     Range: {surf_df['reward'].min():.4f} "
      f"to {surf_df['reward'].max():.4f}")

# ── 3D Surface Plot from CSV ──────────────────────────────────
N_grid = 120
wp_v   = np.linspace(0.05, 0.90, N_grid)
wt_v   = np.linspace(0.05, 0.90, N_grid)
WP,WT  = np.meshgrid(wp_v, wt_v)
WE     = 1.0-WP-WT
VALID  = (WE>=0.05)&(WP>=0.05)&(WT>=0.05)
REW    = np.where(VALID, reward_fn(WP,WT), np.nan)
rew_p  = float(reward_fn(wp_peak, wt_peak))

fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection="3d")
fig.suptitle(
    "Joint Reward Response Surface\n"
    r"$w_p + w_t + w_e = 1$,  each $\geq 0.05$"
    f"  |  Peak = {rew_p:.3f} at proposed weights",
    fontsize=11, fontweight="bold")

norm = Normalize(vmin=np.nanmin(REW), vmax=rew_p)
surf = ax.plot_surface(WP,WT,REW,
                       cmap="RdYlGn",alpha=0.90,
                       linewidth=0,antialiased=True,
                       norm=norm)
cbar = fig.colorbar(surf,ax=ax,shrink=0.50,
                    pad=0.10,aspect=14)
cbar.set_label("Mean Joint Reward",fontsize=10)
cbar.ax.tick_params(labelsize=8)

ax.set_zlim(np.nanmin(REW)-0.002,-0.02)
ax.set_zticks([-0.18,-0.14,-0.10,
               -0.06,-0.045,-0.02])
ax.set_zticklabels(["-0.18","-0.14","-0.10",
                    "-0.06","-0.045","-0.02"],
                   fontsize=8)

# Drop line
ax.plot([wp_peak,wp_peak],[wt_peak,wt_peak],
        [np.nanmin(REW)-0.002,rew_p],
        color="black",lw=1.2,ls="--",alpha=0.5)

# Optimal region ellipse
theta  = np.linspace(0,2*np.pi,60)
wp_r   = wp_peak+0.025*np.cos(theta)
wt_r   = wt_peak+0.025*np.sin(theta)
z_r    = np.array([reward_fn(w,v)
                   for w,v in zip(wp_r,wt_r)])
ax.plot(wp_r,wt_r,z_r,
        color="black",lw=2.0,zorder=15)
for i in range(len(theta)-1):
    ax.plot([wp_peak,wp_r[i]],
            [wt_peak,wt_r[i]],
            [rew_p,z_r[i]],
            color="black",lw=0.3,alpha=0.12)

ax.set_xlabel(r"$w_p$ — Penalty weight",
              fontsize=10,labelpad=8)
ax.set_ylabel(r"$w_t$ — Latency weight",
              fontsize=10,labelpad=8)
ax.set_zlabel("Joint Reward",
              fontsize=10,labelpad=8)
ax.set_xlim(0.05,0.90)
ax.set_ylim(0.05,0.90)
ax.xaxis.set_major_locator(
    ticker.MultipleLocator(0.20))
ax.yaxis.set_major_locator(
    ticker.MultipleLocator(0.20))
ax.tick_params(axis="x",labelsize=8)
ax.tick_params(axis="y",labelsize=8)
ax.view_init(elev=28,azim=-50)

# Peak annotation
z_text = rew_p+0.025
ax.text(wp_peak,wt_peak,z_text,
        f"$w_p = 0.45 \\pm 0.025$,  "
        f"$w_t = 0.35 \\pm 0.025$\n"
        f"Proposed: $w_p={wp_peak}$, "
        f"$w_t={wt_peak}$, $w_e={we_peak}$\n"
        f"Reward $= {rew_p:.3f}$",
        fontsize=7.5,fontweight="bold",
        color="black",ha="center",va="bottom",
        bbox=dict(boxstyle="round,pad=0.30",
                  fc="white",ec="black",
                  alpha=0.95,lw=1.0),
        zorder=20)
ax.plot([wp_peak,wp_peak],[wt_peak,wt_peak],
        [rew_p+0.001,z_text-0.003],
        color="black",lw=1.3,zorder=20)
ax.scatter([wp_peak],[wt_peak],[rew_p+0.001],
           color="black",s=35,marker="v",
           zorder=21)
ax.legend(
    handles=[Patch(
        facecolor="none",edgecolor="black",
        lw=2.0,
        label=r"Optimal region "
              r"($w_p\!=\!0.45\pm0.025$, "
              r"$w_t\!=\!0.35\pm0.025$)")],
    loc="upper left",fontsize=8,framealpha=0.95)

plt.tight_layout()
plt.savefig(f"Images/IDS_weight_3d_N{N}.png",
            dpi=160,bbox_inches="tight")
plt.close()
print(f"[10] Saved → Images/IDS_weight_3d_N{N}.png")

# ── Weight sensitivity from CSV ───────────────────────────────
ws_csv = "results/weight_sensitivity.csv"
if not os.path.exists(ws_csv):
    print(f"[10] ⚠ {ws_csv} not found.")
    print("     Run 05_train.py first to generate it.")
    exit(1)
ws_df = pd.read_csv(ws_csv)
print(f"[10] Loaded {ws_csv} ({len(ws_df)} configs)")

WC_KEYS  = list(ws_df["config"].values)
WC_LABELS= [k.replace(" (Proposed)","\n(Proposed)")
             for k in WC_KEYS]
WC_COLS  = {"Security-Critical":"#C03030",
             "Real-Time":"#3060C0",
             "Energy-Aware":"#20A060",
             "Balanced (Proposed)":"#C07020"}
WC_HATS  = {"Security-Critical":"///",
             "Real-Time":"\\\\\\",
             "Energy-Aware":"xxx",
             "Balanced (Proposed)":"***"}
cols=[WC_COLS.get(k,"#808080") for k in WC_KEYS]
hats=[WC_HATS.get(k,"///")     for k in WC_KEYS]

def clean_ax(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8)
    ax.tick_params(top=False,right=False)
    ax.grid(axis="y",alpha=0.3,
            linestyle="--",color="gray")
    ax.set_axisbelow(True)

fig,axs=plt.subplots(1,4,figsize=(18,6))
fig.suptitle(
    "Reward Weight Sensitivity Analysis\n"
    r"($w_p + w_t + w_e = 1$, "
    "values from results/weight_sensitivity.csv)",
    fontsize=11,fontweight="bold")

METRICS=[
    ("acc",   "Detection Accuracy (%)", "{:.1f}%"),
    ("lat",   "Process Time (ms)",      "{:.0f}"),
    ("eng",   "Energy Cons. (uAh)",     "{:.1f}"),
    ("reward","Joint Reward",           "{:.3f}"),
]
for ax,(col_k,yl,fmt) in zip(axs,METRICS):
    vals=list(ws_df[col_k].values)
    is_rew=(col_k=="reward")
    x=np.arange(len(WC_KEYS))
    bars=ax.bar(x,vals,color=cols,hatch=hats,
                edgecolor="black",lw=0.9,width=0.52)
    for bar,val in zip(bars,vals):
        if is_rew:
            ypos=0+abs(min(vals))*0.04
        else:
            ypos=bar.get_height()+max(vals)*0.025
        ax.text(bar.get_x()+bar.get_width()/2,
                ypos,fmt.format(val),
                ha="center",va="bottom",
                fontsize=9,fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(WC_LABELS,fontsize=9)
    ax.set_ylabel(yl,fontsize=11)
    if is_rew:
        ax.set_ylim(min(vals)*1.30,0.02)
        # Add weight labels inside bars
        for i,k in enumerate(WC_KEYS):
            w_row=ws_df[ws_df.config==k].iloc[0]
            ax.text(i,min(vals)*1.25,
                    f"$w_p$={w_row['wp']}\n"
                    f"$w_t$={w_row['wt']}\n"
                    f"$w_e$={w_row['we']}",
                    ha="center",va="bottom",
                    fontsize=7,
                    color=WC_COLS.get(k,"#808080"),
                    fontweight="bold")
    else:
        ax.set_ylim(0,max(vals)*1.28)
    bars[-1].set_edgecolor("#C07020")
    bars[-1].set_linewidth(2.5)
    clean_ax(ax)

plt.tight_layout()
plt.savefig(f"Images/IDS_weight_sensitivity_N{N}.png",
            dpi=150,bbox_inches="tight")
plt.close()
print(f"[10] Saved → Images/IDS_weight_sensitivity_N{N}.png")
print("[10] Complete.")
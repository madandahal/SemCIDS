"""
05_train.py 
=====================================================
"""
import os, sys, random, pickle
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
from drl_agent import (MultiDeviceAgent, ZhaoAgent,
                        ACTIONS, COMPRESSION, OFFLOAD,
                        N_DEVICES, STATE_SIZE, N_ACT_PER)

SEED=42; random.seed(SEED)
np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

# ── Load config ───────────────────────────────────────────────
cfg      = pickle.load(open("results/config.pkl","rb"))
T_MAX    = cfg["T_MAX"];   E_MAX    = cfg["E_MAX"]
BW_MAX   = cfg["BW_MAX"];  C_EDGE_MAX = cfg["C_EDGE_MAX"]
ALPHA_TX = cfg.get("ALPHA_TX",  0.50)
ALPHA_ENC= cfg.get("ALPHA_ENC", 0.40)
ALPHA_LOC= cfg.get("ALPHA_LOC", 0.15)
BETA_K   = cfg.get("BETA_K",    1.20)
BETA_CAP = cfg.get("BETA_CAP",  1.00)
BETA_ACC = cfg.get("BETA_ACC",  1.00)
BETA_LAT = cfg.get("BETA_LAT",  1.00)
ALPHA_MIN= cfg.get("ALPHA_MIN", 0.60)
DELTA    = cfg.get("DELTA",     0.05)
W_P      = cfg.get("W_P", {2:0.60,1:0.45,0:0.30})
GAMMA_L  = cfg.get("GAMMA_L",   0.64)
GAMMA_E  = cfg.get("GAMMA_E",   0.36)
ALPHA_C  = cfg.get("ALPHA_C",   45.0)
ALPHA_E  = cfg.get("ALPHA_E",   80.0)
ALPHA_B  = cfg.get("ALPHA_B",   30.0)

tr_states = np.load("results/train_states.npy")
tr_devs   = pickle.load(open("results/train_devices.pkl","rb"))

N_EPOCHS=30; WARMUP=2; CONV_WINDOW=6; CONV_TOL=0.005

# ── Functions  ───────────────────

def detection_accuracy(t, O, K):
    """alpha_{i,t} = (1-O)*alpha_local + O*alpha_edge"""
    af  = COMPRESSION[K]["acc_f"]
    xi  = t["intensity"]
    pri = t["priority"]
    pb  = {2:1.0,1:0.95,0:0.90}.get(int(pri),0.95)
    return float(np.clip(
        (1-O)*(0.88-0.50*xi)
        + O*min(1.0,(0.76+0.23*xi)*af*pb), 0, 1))

def get_C_local(xi):
    """C_local = C_local_max - alpha_c * xi"""
    return float(np.clip(
        75.0 - ALPHA_C*xi, 30.0, 75.0))

def get_C_edge(devs):
    """C_edge shared — computed from avg intensity"""
    avg_xi = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(
        C_EDGE_MAX - ALPHA_E*avg_xi,
        C_EDGE_MAX*0.3, C_EDGE_MAX))

def get_BW(devs):
    """BW_t shared — computed from avg intensity"""
    avg_xi = float(np.mean([d["intensity"] for d in devs]))
    return float(np.clip(
        BW_MAX - ALPHA_B*avg_xi, 15.0, BW_MAX))

def compute_energy(t, O, K, C_local):
    """
    E = O*(E_trans + E_comp) + (1-O)*E_local
    E_trans = alpha_tx  * tau * R_K
    E_comp  = alpha_enc * tau * C_K
    E_local = alpha_loc * d / C_local
    """
    R_K = COMPRESSION[K]["ratio"]
    C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]
    d   = t["demand"]
    E_trans = ALPHA_TX  * tau * R_K
    E_comp  = ALPHA_ENC * tau * C_K
    E_local = ALPHA_LOC * d / (C_local+1e-6)
    E = O*(E_trans+E_comp) + (1-O)*E_local
    return float(np.clip(E, 0, E_MAX))

def compute_latency(t, O, K, C_local,
                    BW_t, C_edge,
                    all_devs, all_actions):
    """
    T = O*(T_trans+T_comp) + T_edge + (1-O)*T_local
    T_trans = tau*R_K/BW_t
    T_edge  = sum_i(O_i*d_i)/C_edge  (shared)
    T_comp  = beta_K * C_K
    T_local = d / C_local
    """
    R_K = COMPRESSION[K]["ratio"]
    C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]
    d   = t["demand"]
    T_trans = tau * R_K / (BW_t+1e-6)
    T_comp  = BETA_K * C_K
    T_local = d / (C_local+1e-6)
    T_edge  = sum(
        O_j * dv["demand"]
        for dv,(O_j,_) in zip(all_devs, all_actions)
    ) / (C_edge+1e-6)
    T = O*(T_trans+T_comp) + T_edge + (1-O)*T_local
    return float(np.clip(T, 0, T_MAX))

def compute_penalty(t, O, acc, T_t, C_local):
    """
    P = P_cap + P_acc + P_lat
    P_cap = beta_cap * max(0, (1-O)*tau - C_local)
    P_acc = beta_acc * max(0, alpha_min - acc)
    P_lat = beta_lat * max(0, T - T_max)
    """
    tau   = t["tau"]
    P_cap = BETA_CAP * max(0.0, (1-O)*tau - C_local)
    P_acc = BETA_ACC * max(0.0, ALPHA_MIN - acc)
    P_lat = BETA_LAT * max(0.0, T_t - T_MAX)
    return float(P_cap + P_acc + P_lat)

def compute_reward_device(penalty, T_t, E_t, O, priority):
    """
    r_{i,t} = -(w_p*P + w_t*T_norm + w_e*E_norm)
              + delta*O*(1-O)
    """
    w_p = W_P.get(int(priority), 0.45)
    w_t = (1-w_p)*GAMMA_L
    w_e = (1-w_p)*GAMMA_E
    T_n = T_t/T_MAX
    E_n = E_t/E_MAX
    cost  = w_p*penalty + w_t*T_n + w_e*E_n
    bonus = DELTA * O * (1-O)
    return float(np.clip(-cost+bonus, -1.0, 0.1))

def compute_joint_reward(device_rewards):
    """r_t = (1/N) * sum_i r_{i,t}"""
    return float(np.mean(device_rewards))

def allocate_bandwidth(actions):
    demands=[O*COMPRESSION[K]["ratio"] for O,K in actions]
    total=sum(demands)+1e-9
    return [d/total for d in demands] if total>1.0 else demands

# ── Training loop ─────────────────────────────────────────────
# ── Convergence targets for DQN Proposed ─────────────────────
# These are the values from Fig.3 that the agent must reach.
# Training is considered successful if final epoch values
# are within tolerance of these targets.
TARGETS = {
    "DQN Proposed": dict(
        reward  = (-0.055, None),   # must be > -0.055
        latency = (None,   0.005),  # must be < 0.005 (5ms)
        energy  = (None,   1.50),   # must be < 1.50
    ),
    "No Semantic": dict(
        reward  = (-0.080, None),
        latency = (None,   0.006),
        energy  = (None,   2.00),
    ),
    "No Priority": dict(
        reward  = (-0.090, None),
        latency = (None,   0.007),
        energy  = (None,   2.00),
    ),
    "Fixed Semantic": dict(
        reward  = (-0.110, None),
        latency = (None,   0.010),
        energy  = (None,   2.00),
    ),
    "Zhao et al.": dict(
        reward  = (-0.100, None),
        latency = (None,   0.008),
        energy  = (None,   2.00),
    ),
}

def train_dqn(agent, label,
              use_priority=True,
              use_semantic=True,
              force_k=None):
    rew_hist=[]; lat_hist=[]; eng_hist=[]
    print(f"\n[05] Training {label} ...")

    for epoch in range(N_EPOCHS):
        agent.decay_eps(epoch, N_EPOCHS)
        if epoch < WARMUP: agent.eps = 1.0

        idxs = np.random.permutation(len(tr_states))
        ep_r=[]; ep_l=[]; ep_e=[]

        for i in idxs:
            s    = tr_states[i]
            devs = tr_devs[i]

            # Agent selects actions
            a_idx = agent.act(s)
            if not use_semantic:
                # Zhao: K=0, only offload fraction
                actions=[(ACTIONS[a][0],0) for a in a_idx]
            elif hasattr(agent,'_blind'):
                actions=[(OFFLOAD[a],0) for a in a_idx]
            elif force_k is not None:
                # Force a specific K level
                actions=[(ACTIONS[a][0], force_k)
                         for a in a_idx]
            else:
                actions=[ACTIONS[a] for a in a_idx]

            # Shared physics values
            C_edge = get_C_edge(devs)
            BW_t   = get_BW(devs)

            step_rews=[]; step_lats=[]; step_engs=[]
            for di,(O,K) in enumerate(actions):
                t     = devs[di]
                pri   = t["priority"] if use_priority else 1
                C_local = get_C_local(t["intensity"])

                # Detection accuracy
                acc = detection_accuracy(t, O, K)

                # Latency (uses shared C_edge, BW_t)
                T_t = compute_latency(
                    t, O, K, C_local,
                    BW_t, C_edge,
                    devs, actions)

                # Energy
                E_t = compute_energy(t, O, K, C_local)

                # Penalty
                pen = compute_penalty(
                    t, O, acc, T_t, C_local)

                # Per-device reward — uses Stage B priority
                # (ground truth, known after detection)
                r_i = compute_reward_device(
                    pen, T_t, E_t, O,
                    t["priority"] if use_priority
                    else 1)

                step_rews.append(r_i)
                step_lats.append(T_t)
                step_engs.append(E_t)

            # Joint reward = mean across devices
            joint_r = compute_joint_reward(step_rews)

            # Update b_rem for next state
            bw_used = sum(
                O*COMPRESSION[K]["ratio"]
                for O,K in actions)
            bw_rem = max(0.0, 1.0-bw_used)
            ni  = (i+1) % len(tr_states)
            s2  = tr_states[ni].copy()
            s2[-1] = bw_rem   # update b_rem in state

            # Push to buffer and update
            if hasattr(agent,'per'):
                agent.per.push(s,a_idx,joint_r,s2,0.0)
                agent.update()
            else:
                agent.push(s,a_idx,joint_r,s2)
                agent.update()

            ep_r.append(joint_r)
            ep_l.append(float(np.mean(step_lats)))
            ep_e.append(float(np.mean(step_engs)))

        rew_hist.append(np.mean(ep_r))
        lat_hist.append(np.mean(ep_l))
        eng_hist.append(np.mean(ep_e))
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Rew {rew_hist[-1]:+.4f} | "
              f"Lat {lat_hist[-1]:.4f} | "
              f"Eng {eng_hist[-1]:.4f} | "
              f"Eps {agent.eps:.3f}")

        # Early stopping
        if epoch+1 >= CONV_WINDOW:
            w = CONV_WINDOW
            if (np.std(rew_hist[-w:]) < CONV_TOL and
                np.std(lat_hist[-w:]) < CONV_TOL and
                np.std(eng_hist[-w:]) < CONV_TOL):
                print(f"  ✓ Converged at epoch {epoch+1}")
                break

    # ── Check convergence targets ─────────────────────────────
    final_rew = float(np.mean(rew_hist[-3:]))
    final_lat = float(np.mean(lat_hist[-3:]))
    final_eng = float(np.mean(eng_hist[-3:]))

    print(f"\n  Final metrics for {label}:")
    print(f"    Reward  : {final_rew:+.4f}")
    print(f"    Latency : {final_lat:.4f}")
    print(f"    Energy  : {final_eng:.4f}")

    if label in TARGETS:
        tgt = TARGETS[label]
        ok_r = (tgt["reward"][0] is None or
                final_rew > tgt["reward"][0])
        ok_l = (tgt["latency"][1] is None or
                final_lat < tgt["latency"][1])
        ok_e = (tgt["energy"][1] is None or
                final_eng < tgt["energy"][1])
        print(f"    Reward  target > {tgt['reward'][0]}: "
              f"{'✓' if ok_r else '✗ WARN'}")
        print(f"    Latency target < {tgt['latency'][1]}: "
              f"{'✓' if ok_l else '✗ WARN'}")
        print(f"    Energy  target < {tgt['energy'][1]}: "
              f"{'✓' if ok_e else '✗ WARN'}")
        if not all([ok_r, ok_l, ok_e]):
            print(f"  ⚠ {label} did not reach targets — "
                  f"consider more epochs or tuning lr")

    return rew_hist, lat_hist, eng_hist

# ── Train all 5 agents that need training ─────────────────────
# Methods 1,2,6,7 are rule-based — no training needed
# Random Policy: random action each step — no training needed

agent_dqn     = MultiDeviceAgent()   # DQN Proposed
agent_zhao    = ZhaoAgent()           # Zhao et al.
agent_nopri   = MultiDeviceAgent()   # No Priority ablation
agent_nosem   = MultiDeviceAgent()   # No Semantic ablation
agent_fixsem  = MultiDeviceAgent()   # Fixed Semantic ablation

print("\n[05] Training 5 agents ...")
print("     Rule-based (no training needed):")
print("       Always Local, Always Offload,")
print("       Threshold-Based, Random Policy")

# 1. DQN Proposed — full SemCIDS
rew_dqn, lat_dqn, eng_dqn = train_dqn(
    agent_dqn, "DQN Proposed",
    use_priority=True,
    use_semantic=True,
    force_k=None)

# 2. Zhao et al. — standard DQN, no priority, no semantic
rew_zhao, lat_zhao, eng_zhao = train_dqn(
    agent_zhao, "Zhao et al.",
    use_priority=False,
    use_semantic=False,
    force_k=None)

# 3. No Priority — same as Proposed but w_p uniform
rew_nopri, lat_nopri, eng_nopri = train_dqn(
    agent_nopri, "No Priority",
    use_priority=False,
    use_semantic=True,
    force_k=None)

# 4. No Semantic — DRL but K=0 forced (no compression)
rew_nosem, lat_nosem, eng_nosem = train_dqn(
    agent_nosem, "No Semantic",
    use_priority=True,
    use_semantic=True,
    force_k=0)             # K=0 always

# 5. Fixed Semantic — DRL but K=2 forced always
rew_fixsem, lat_fixsem, eng_fixsem = train_dqn(
    agent_fixsem, "Fixed Semantic",
    use_priority=True,
    use_semantic=True,
    force_k=2)             # K=2 always

# Save all checkpoints
torch.save(agent_dqn.online.state_dict(),
           "checkpoints/dqn_proposed.pth")
torch.save(agent_zhao.online.state_dict(),
           "checkpoints/zhao_dqn.pth")
torch.save(agent_nopri.online.state_dict(),
           "checkpoints/dqn_no_priority.pth")
torch.save(agent_nosem.online.state_dict(),
           "checkpoints/dqn_no_semantic.pth")
torch.save(agent_fixsem.online.state_dict(),
           "checkpoints/dqn_fixed_semantic.pth")
print("\n[05] Saved 5 checkpoints → checkpoints/")

rows=[]
for name, rews, lats, engs in [
    ("DQN Proposed",   rew_dqn,    lat_dqn,    eng_dqn),
    ("No Semantic",    rew_nosem,  lat_nosem,  eng_nosem),
    ("No Priority",    rew_nopri,  lat_nopri,  eng_nopri),
    ("Fixed Semantic", rew_fixsem, lat_fixsem, eng_fixsem),
    ("Zhao et al.",    rew_zhao,   lat_zhao,   eng_zhao),
]:
    for ep,(r,l,e) in enumerate(zip(rews,lats,engs),1):
        rows.append({"method":name,"epoch":ep,
                     "reward":r,"latency":l,"energy":e})
pd.DataFrame(rows).to_csv(
    "results/training_curves.csv",index=False)
print("[05] Saved → results/training_curves.csv")

# ── Weight sensitivity — train 4 agents with ─────────────────
# different weight configs and record final reward
# Security-Critical: w_p=0.70
# Real-Time:         w_p=0.20, w_t=0.64
# Energy-Aware:      w_p=0.20, w_e=0.64
# Balanced(Proposed):w_p=0.45 (default)
print("\n[05] Training weight sensitivity agents ...")

WS_CONFIGS = [
    {"config":"Security-Critical",
     "wp":0.70,"wt":0.19,"we":0.11},
    {"config":"Real-Time",
     "wp":0.20,"wt":0.64,"we":0.16},
    {"config":"Energy-Aware",
     "wp":0.20,"wt":0.16,"we":0.64},
    {"config":"Balanced (Proposed)",
     "wp":0.45,"wt":0.35,"we":0.20},
]

def train_with_weights(wp, wt, we, label):
    """Train agent with custom reward weights."""
    # Override W_P with uniform scaled by wp
    # and w_t, w_e computed from wt/we directly
    gamma_l_local = wt/(wt+we) if (wt+we)>0 else 0.64
    gamma_e_local = we/(wt+we) if (wt+we)>0 else 0.36

    agent = MultiDeviceAgent()
    opt   = torch.optim.AdamW(
        agent.online.parameters(),
        lr=5e-4, weight_decay=1e-4)

    print(f"  Training {label} "
          f"(wp={wp},wt={wt:.2f},we={we:.2f}) ...")

    ep_rewards=[]
    for epoch in range(N_EPOCHS):
        agent.decay_eps(epoch, N_EPOCHS)
        if epoch<WARMUP: agent.eps=1.0
        idxs=np.random.permutation(len(tr_states))
        ep_r=[]
        for i in idxs:
            s=tr_states[i]; devs=tr_devs[i]
            a_idx=agent.act(s)
            actions=[ACTIONS[a] for a in a_idx]
            C_edge=get_C_edge(devs)
            BW_t  =get_BW(devs)
            step_rews=[]
            for di,(O,K) in enumerate(actions):
                t      =devs[di]
                C_local=get_C_local(t["intensity"])
                acc    =detection_accuracy(t,O,K)
                T_t    =compute_latency(
                    t,O,K,C_local,BW_t,C_edge,
                    devs,actions)
                E_t    =compute_energy(t,O,K,C_local)
                pen    =compute_penalty(
                    t,O,acc,T_t,C_local)
                # Use custom weights
                T_n=T_t/T_MAX; E_n=E_t/E_MAX
                cost=(wp*pen
                      +wt*T_n
                      +we*E_n)
                bonus=DELTA*O*(1-O)
                r_i=float(np.clip(
                    -cost+bonus,-1.0,0.1))
                step_rews.append(r_i)
            joint_r=float(np.mean(step_rews))
            bw_used=sum(
                O*COMPRESSION[K]["ratio"]
                for O,K in actions)
            bw_rem=max(0.0,1.0-bw_used)
            ni=(i+1)%len(tr_states)
            s2=tr_states[ni].copy(); s2[-1]=bw_rem
            agent.per.push(s,a_idx,joint_r,s2,0.0)
            agent.update()
            ep_r.append(joint_r)
        ep_rewards.append(float(np.mean(ep_r)))

    final_reward=float(np.mean(ep_rewards[-3:]))
    print(f"    Final reward: {final_reward:.4f}")
    return agent, final_reward

ws_rows=[]
for wc in WS_CONFIGS:
    ws_agent, final_r = train_with_weights(
        wc["wp"], wc["wt"], wc["we"],
        wc["config"])

    # Quick eval on test set
    te_states_ws=np.load("results/test_states.npy")
    te_devs_ws=pickle.load(
        open("results/test_devices.pkl","rb"))
    accs=[]; lats=[]; engs=[]
    ws_agent.online.eval()
    for i in range(len(te_states_ws)):
        s=te_states_ws[i]; devs=te_devs_ws[i]
        with torch.no_grad():
            q=ws_agent.online(
                torch.FloatTensor(s).unsqueeze(0))
        actions=[ACTIONS[qi.argmax().item()] for qi in q]
        C_edge=get_C_edge(devs); BW_t=get_BW(devs)
        for di,(O,K) in enumerate(actions):
            t=devs[di]
            C_local=get_C_local(t["intensity"])
            accs.append(detection_accuracy(t,O,K))
            lats.append(compute_latency(
                t,O,K,C_local,BW_t,C_edge,
                devs,actions))
            engs.append(compute_energy(t,O,K,C_local))

    ws_rows.append({
        "config":  wc["config"],
        "wp":      wc["wp"],
        "wt":      wc["wt"],
        "we":      wc["we"],
        "acc":     round(float(np.mean(accs))*100, 1),
        "lat":     round(float(np.mean(lats))*1000, 1),
        "eng":     round(float(np.mean(engs))*10,  2),
        "reward":  round(final_r, 4),
    })

ws_df=pd.DataFrame(ws_rows)
ws_df.to_csv("results/weight_sensitivity.csv",index=False)
print("\n[05] Saved → results/weight_sensitivity.csv")
print(ws_df[["config","wp","wt","we",
             "acc","reward"]].to_string(index=False))
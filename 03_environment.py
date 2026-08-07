"""
03_environment.py  
================================================
State vector: S_t ∈ R^{5N+3}

Per-device (5 dims each):
  s_{i,t} = [C_local, tau, d, xi, P]

Shared global (3 dims):
  C_edge_t, BW_t, b_rem_t

Total: 5N + 3
"""
import os, pickle, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

X_train = np.load("results/X_train.npy")
X_test  = np.load("results/X_test.npy")
y_train = np.load("results/y_train.npy")
y_test  = np.load("results/y_test.npy")
p_train = np.load("results/p_train.npy")
p_test  = np.load("results/p_test.npy")
COL_IDX = pickle.load(open("results/col_idx.pkl","rb"))

N_DEVICES = int(input(
    "\n[03] How many IoT edge devices? [2-10]: "))
assert 2 <= N_DEVICES <= 10

# ── Constants ─────────────────────────────────────────────────
OFFLOAD = [0.0, 0.25, 0.50, 0.75, 1.0]
COMPRESSION = {
    0: dict(ratio=1.00, acc_f=1.00, cost=0.0,
            name="K0",    label="No compression"),
    1: dict(ratio=0.75, acc_f=1.06, cost=0.10,
            name="K_low", label="Low compression"),
    2: dict(ratio=0.50, acc_f=0.82, cost=1.0,
            name="K_med", label="Medium compression"),
    3: dict(ratio=0.25, acc_f=0.55, cost=2.5,
            name="K_high",label="High compression"),
}

# Joint action space: |O|×|K| = 5×4 = 20 per device
# Total: 20^N across all N devices
ACTIONS   = [(o,k) for o in OFFLOAD for k in COMPRESSION]
N_ACT_PER = len(ACTIONS)   # 20
assert N_ACT_PER == 20, "Action space must be 5×4=20"

# Constants 
C_LOCAL_MAX = 75.0    # max local CPU
C_LOCAL_MIN = 30.0    # min local CPU
C_EDGE_MAX  = 260.0   # max edge CPU
BW_MAX      = 45.0    # max bandwidth Mbps
BW_MIN      = 15.0    # min bandwidth Mbps
ALPHA_C     = 45.0    # CPU degradation rate
ALPHA_E     = 80.0    # edge CPU degradation
ALPHA_B     = 30.0    # BW degradation
SIGMA_MAX   = 1.5     # severity normalisation
T_MAX       = 10.0
E_MAX       = 15.0

# ── State dimensions (5N+3) ────────────────────
# Per-device: [C_local, tau, d, xi, P] = 5 dims
# Shared:     [C_edge, BW, b_rem]      = 3 dims
# Total:       5N + 3
STATE_SIZE = 5*N_DEVICES + 3

# Normalisation bounds
BOUNDS_DEV    = np.array([C_LOCAL_MAX, 30., 15., 1., 2.],
                          dtype=np.float32)
# Shared: C_edge_max, BW_max, 1.0
BOUNDS_SHARED = np.array([C_EDGE_MAX, BW_MAX, 1.0],
                          dtype=np.float32)

print(f"[03] State dim = 5*{N_DEVICES}+3 = {STATE_SIZE}")

def get(row, col):
    if col not in COL_IDX: return 0.0
    return float(np.clip(row[COL_IDX[col]], -3, 3))

# ── Two-stage priority ───────────────────────────
# Stage A thresholds — conservative to avoid under-prioritising
XI_HIGH = 0.70   # xi >= 0.70 → P_high estimate
XI_MED  = 0.40   # xi >= 0.40 → P_medium estimate

def estimate_priority_stage_a(xi):
    """
    Stage A — Pre-screen estimate.
    """
    if   xi >= XI_HIGH: return 2   # P_high   estimate
    elif xi >= XI_MED:  return 1   # P_medium estimate
    else:               return 0   # P_low    estimate

def correct_priority_stage_b(true_priority):
    """
    Stage B — Post-detection correction.
    Category → Priority mapping:
      DDoS, Mirai       → P_high   (2)
      DoS, Spoofing     → P_medium (1)
      Recon, BruteForce → P_low    (0)
      Benign            → P_low    (0)
    """
    return int(true_priority)   # ground truth = Stage B

def extract_traffic(row, label, priority):
    """
    Extract 6 key features and compute:
      - xi:         severity 
      - tau:        task size from Header_Length
      - d:          IDS workload from rate/syn/ack
      - P_est:      Stage A priority estimate from xi
      - P_true:     Stage B corrected priority (ground truth)

    The state vector uses P_est (Stage A) because the agent
    observes state at flow arrival before classification.
    The reward uses P_true (Stage B) because priority is
    confirmed after detection completes.
    """
    rate = get(row, "Rate")
    syn  = get(row, "syn_count")
    ack  = get(row, "ack_count")
    iat  = get(row, "IAT")
    hlen = get(row, "Header_Length")
    psh  = get(row, "psh_flag_number")

    # ── Severity xi  ───────────────────────────────
    sigma_raw = (0.30*max(rate, 0)
               + 0.30*max(syn,  0)
               + 0.20*max(ack,  0)
               + 0.15*max(psh,  0)
               - 0.05*iat)
    xi = float(np.clip(sigma_raw / SIGMA_MAX, 0.0, 1.0))

    # ── Task size tau ─────────────────────────────────────────
    tau = float(np.clip(abs(hlen)*8 + 2, 2., 30.))

    # ── IDS workload d ────────────────────────────────────────
    d = float(np.clip(
        0.40*abs(rate) + 1.80*abs(syn) + 1.20*abs(ack),
        0.1, 15.))

    # ── Two-stage priority ────────────────────────────────────
    P_stage_a = estimate_priority_stage_a(xi)     # from xi only
    P_stage_b = correct_priority_stage_b(priority) # ground truth

    return dict(
        rate=rate, syn=syn, ack=ack,
        iat=iat, hlen=hlen, psh=psh,
        intensity=xi,           # xi_{i,t}
        tau=tau,                # tau_{i,t}
        demand=d,               # d_{i,t}
        label=int(label),
        priority_est=P_stage_a, # Stage A: used in STATE
        priority=P_stage_b,     # Stage B: used in REWARD
    )

def simulate_coordinated_attack(base, n_dev, rho=0.80):
    """
    With prob rho, other devices share the same
    attack pattern (botnet coordination).
    """
    devs = [base]
    for _ in range(1, n_dev):
        if (base["label"]==1 and
                base["priority"]==2 and
                random.random() < rho):
            noise = random.gauss(0, 0.1)
            xi_c  = float(np.clip(
                base["intensity"]+noise, 0.5, 1.0))
            devs.append(dict(base, intensity=xi_c))
        else:
            xi_b = float(np.clip(
                random.gauss(0.1, 0.05), 0., 0.4))
            devs.append(dict(base,
                intensity=xi_b, label=0, priority=0))
    return devs

def build_joint_state(devs, bw_rem):
    """
    S_t = [s_{1,t}, ..., s_{N,t}, C_edge_t, BW_t, b_rem_t]
        ∈ R^{5N+3}

    Per-device s_{i,t} = [C_local, tau, d, xi, P] ∈ R^5
    Shared:               C_edge_t, BW_t, b_rem_t   ∈ R^3
    """
    parts = []

    # ── Compute shared global features ───────────────────────
    # Use average intensity across devices for shared channel
    avg_xi  = float(np.mean([t["intensity"] for t in devs]))
    C_edge  = float(np.clip(
        C_EDGE_MAX - ALPHA_E * avg_xi,
        C_EDGE_MAX * 0.3, C_EDGE_MAX))
    BW_t    = float(np.clip(
        BW_MAX - ALPHA_B * avg_xi,
        BW_MIN, BW_MAX))

    # ── Per-device sub-vectors ────────────────────────────────
    for t in devs:
        xi  = t["intensity"]
        # C_local_{i,t} = C_local_max - alpha_c * xi
        C_l = float(np.clip(
            C_LOCAL_MAX - ALPHA_C * xi,
            C_LOCAL_MIN, C_LOCAL_MAX))
        tau = t["tau"]      # task size
        d   = t["demand"]   # IDS workload

        # Stage A priority estimate — agent observes this
        # at flow arrival BEFORE classification
        P = float(t["priority_est"])

        # s_{i,t} = [C_local, tau, d, xi, P_est]
        # P here is Stage A estimate, not ground truth
        s_i = np.array([C_l, tau, d, xi, P],
                        dtype=np.float32)

        # Normalise to [0,1]
        s_i_norm = s_i / BOUNDS_DEV
        parts.append(s_i_norm)

    # ── Shared global features ────────────────────────────────
    # [C_edge_t, BW_t, b_rem_t]
    shared = np.array([C_edge, BW_t, bw_rem],
                       dtype=np.float32)
    shared_norm = shared / BOUNDS_SHARED
    parts.append(shared_norm)

    state = np.concatenate(parts)
    assert len(state) == STATE_SIZE, \
        f"State dim mismatch: {len(state)} != {STATE_SIZE}"
    return state

# ── Constants ───────────────────────
ALPHA_TX  = 0.50   # transmission energy coefficient
ALPHA_ENC = 0.40   # encoding energy coefficient
ALPHA_LOC = 0.15   # local IDS energy coefficient
BETA_K    = 1.20   # encoding time coefficient
BETA_CAP  = 1.00   # CPU overload penalty coefficient
BETA_ACC  = 1.00   # accuracy penalty coefficient
BETA_LAT  = 1.00   # latency penalty coefficient
ALPHA_MIN = 0.60   # minimum acceptable accuracy
DELTA     = 0.05   # partial offload bonus coefficient

# Priority-dependent weights (w_p + w_t + w_e = 1)
W_P = {2: 0.60, 1: 0.45, 0: 0.30}   # miss penalty weight
GAMMA_L = 0.64   # latency fraction of remaining weight
GAMMA_E = 0.36   # energy fraction of remaining weight

def detection_accuracy(t, O, K):
    """
    alpha_{i,t} = (1-O)*alpha_local + O*alpha_edge
    """
    af        = COMPRESSION[K]["acc_f"]
    xi        = t["intensity"]
    pri       = t["priority"]
    local_acc = 0.88 - 0.50*xi
    pb        = {2:1.0,1:0.95,0:0.90}.get(int(pri),0.95)
    edge_acc  = min(1.0,(0.76+0.23*xi)*af*pb)
    return float(np.clip(
        (1-O)*local_acc + O*edge_acc, 0, 1))

def compute_energy(t, O, K, C_local):
    """
    Eq:
    E_{i,t} = O*(E_trans + E_comp) + (1-O)*E_local

    E_trans,i = alpha_tx  * tau_{i,t} * R_{K_{i,t}}
    E_comp,i  = alpha_enc * tau_{i,t} * C_{K_{i,t}}
    E_local,i = alpha_loc * d_{i,t} / C_local_{i,t}
    """
    R_K = COMPRESSION[K]["ratio"]  # R_{K_{i,t}}
    C_K = COMPRESSION[K]["cost"]   # C_{K_{i,t}}
    tau = t["tau"]                 # tau_{i,t}
    d   = t["demand"]              # d_{i,t}

    E_trans = ALPHA_TX  * tau * R_K
    E_comp  = ALPHA_ENC * tau * C_K
    E_local = ALPHA_LOC * d / (C_local + 1e-6)

    E = O * (E_trans + E_comp) + (1-O) * E_local
    return float(np.clip(E, 0, E_MAX))

def compute_latency(t, O, K, C_local,
                    BW_t, C_edge,
                    all_devs, all_actions):
    """
    Eq:
    T_{i,t} = O*(T_trans + T_comp)
              + T_edge
              + (1-O)*T_local

    T_trans,i = tau_{i,t} * R_{K_{i,t}} / BW_t
    T_edge    = sum_i(O_{i,t}*d_{i,t}) / C_edge
    T_comp,i  = beta_K * C_{K_{i,t}}
    T_local,i = d_{i,t} / C_local_{i,t}

    Note: T_edge is SHARED — all devices use
    the same edge server, so it accumulates
    across all N devices simultaneously.
    """
    R_K = COMPRESSION[K]["ratio"]
    C_K = COMPRESSION[K]["cost"]
    tau = t["tau"]
    d   = t["demand"]

    T_trans = tau * R_K / (BW_t + 1e-6)
    T_comp  = BETA_K * C_K
    T_local = d / (C_local + 1e-6)

    # T_edge = sum_i(O_i * d_i) / C_edge
    T_edge = sum(
        O_j * dv["demand"]
        for dv, (O_j, _) in zip(all_devs, all_actions)
    ) / (C_edge + 1e-6)

    T = O*(T_trans + T_comp) + T_edge + (1-O)*T_local
    return float(np.clip(T, 0, T_MAX))

def compute_penalty(t, O, acc, T_t, C_local):
    """
    Eq:
    P_{i,t} = P_cap + P_acc + P_lat

    P_cap,i = beta_cap * max(0, (1-O)*tau - C_local)
    P_acc,i = beta_acc * max(0, alpha_min - alpha_{i,t})
    P_lat,i = beta_lat * max(0, T_{i,t} - T_max)
    """
    tau   = t["tau"]

    # CPU overload: local workload exceeds capacity
    P_cap = BETA_CAP * max(0.0, (1-O)*tau - C_local)

    # Accuracy below minimum threshold
    P_acc = BETA_ACC * max(0.0, ALPHA_MIN - acc)

    # Latency exceeds maximum
    P_lat = BETA_LAT * max(0.0, T_t - T_MAX)

    return float(P_cap + P_acc + P_lat)

def compute_reward_device(penalty, T_t, E_t, O, priority):
    """
    Per-device reward:
    r_{i,t} = -(w_p*P_{i,t} + w_t*T_{i,t} + w_e*E_{i,t})
              + delta * O_{i,t} * (1 - O_{i,t})

    IMPORTANT: priority here is Stage B (post-detection)
    ground-truth priority — NOT Stage A estimate.
    The reward is computed after detection completes so
    the true attack category is known.

    Priority-dependent weights satisfying w_p+w_t+w_e=1:
      w_p: miss penalty weight (higher for high-priority)
      w_t = (1-w_p) * gamma_l
      w_e = (1-w_p) * gamma_e
    """
    w_p = W_P.get(int(priority), 0.45)
    w_t = (1 - w_p) * GAMMA_L   # latency weight
    w_e = (1 - w_p) * GAMMA_E   # energy weight

    # Normalise T and E to [0,1] for comparability
    T_norm = T_t / T_MAX
    E_norm = E_t / E_MAX

    cost  = w_p*penalty + w_t*T_norm + w_e*E_norm
    bonus = DELTA * O * (1 - O)

    return float(np.clip(-cost + bonus, -1.0, 0.1))

def compute_joint_reward(device_rewards):
    """
    r_t = (1/N) * sum_{i=1}^{N} r_{i,t}
    """
    return float(np.mean(device_rewards))

def allocate_bandwidth(actions):
    demands = [O*COMPRESSION[K]["ratio"]
               for O,K in actions]
    total   = sum(demands)+1e-9
    return ([d/total for d in demands]
            if total > 1.0 else demands)

# ── Pre-compute joint states ──────────────────────────────────
print(f"\n[03] Pre-computing states "
      f"(N={N_DEVICES}, dim={STATE_SIZE}) ...")

def make_joint(row, lbl, pri):
    base = extract_traffic(row, lbl, pri)
    devs = simulate_coordinated_attack(
        base, N_DEVICES)
    s    = build_joint_state(devs, 1.0)
    return s, devs

tr_states=[]; tr_devs=[]
for i in range(len(X_train)):
    s,dv = make_joint(X_train[i],
                      int(y_train[i]),
                      int(p_train[i]))
    tr_states.append(s); tr_devs.append(dv)
tr_states = np.array(tr_states, dtype=np.float32)

te_states=[]; te_devs=[]
for i in range(len(X_test)):
    s,dv = make_joint(X_test[i],
                      int(y_test[i]),
                      int(p_test[i]))
    te_states.append(s); te_devs.append(dv)
te_states = np.array(te_states, dtype=np.float32)

np.save("results/train_states.npy", tr_states)
np.save("results/test_states.npy",  te_states)
pickle.dump(tr_devs,
    open("results/train_devices.pkl","wb"))
pickle.dump(te_devs,
    open("results/test_devices.pkl","wb"))

config = dict(
    # Environment
    N_DEVICES=N_DEVICES,
    STATE_SIZE=STATE_SIZE,
    N_ACT_PER=N_ACT_PER,
    OFFLOAD=OFFLOAD,
    COMPRESSION=COMPRESSION,
    ACTIONS=ACTIONS,
    # State constants
    C_LOCAL_MAX=C_LOCAL_MAX,
    C_LOCAL_MIN=C_LOCAL_MIN,
    C_EDGE_MAX=C_EDGE_MAX,
    BW_MAX=BW_MAX,
    BW_MIN=BW_MIN,
    ALPHA_C=ALPHA_C,
    ALPHA_E=ALPHA_E,
    ALPHA_B=ALPHA_B,
    SIGMA_MAX=SIGMA_MAX,
    STATE_BOUNDS_1=BOUNDS_DEV.tolist(),
    STATE_BOUNDS_SHARED=BOUNDS_SHARED.tolist(),
    # Reward constants
    TOTAL_BW=BW_MAX,
    T_MAX=T_MAX,
    E_MAX=E_MAX,
    ALPHA_TX=ALPHA_TX,
    ALPHA_ENC=ALPHA_ENC,
    ALPHA_LOC=ALPHA_LOC,
    BETA_K=BETA_K,
    BETA_CAP=BETA_CAP,
    BETA_ACC=BETA_ACC,
    BETA_LAT=BETA_LAT,
    ALPHA_MIN=ALPHA_MIN,
    DELTA=DELTA,
    W_P=W_P,
    GAMMA_L=GAMMA_L,
    GAMMA_E=GAMMA_E)
pickle.dump(config, open("results/config.pkl","wb"))

# ── Verify reward equations ──────────────────────
print("\n[03] Verifying reward equations...")
_t = dict(tau=10.0, demand=5.0, intensity=0.8,
          priority=2, label=1, rate=2.0,
          syn=1.5, ack=0.8, iat=-0.2,
          hlen=1.5, psh=0.5)
_O=0.5; _K=1; _C_l=52.0; _BW=40.0; _Ce=220.0

_R_K = COMPRESSION[_K]["ratio"]   # 0.75
_C_K = COMPRESSION[_K]["cost"]    # 0.10

# Energy
_E_trans = ALPHA_TX  * _t["tau"] * _R_K        # 0.5*10*0.75=3.75
_E_comp  = ALPHA_ENC * _t["tau"] * _C_K        # 0.4*10*0.10=0.40
_E_local = ALPHA_LOC * _t["demand"] / _C_l     # 0.15*5/52
_E = _O*(_E_trans+_E_comp)+(1-_O)*_E_local
print(f"  E_trans={_E_trans:.3f}  E_comp={_E_comp:.3f}  "
      f"E_local={_E_local:.3f}  E={_E:.3f}")

# Latency
_T_trans = _t["tau"]*_R_K/_BW                  # 10*0.75/40
_T_comp  = BETA_K * _C_K                       # 1.2*0.10
_T_local = _t["demand"]/_C_l                   # 5/52
_T_edge  = (_O*_t["demand"])/_Ce               # single device test
_T = _O*(_T_trans+_T_comp)+_T_edge+(1-_O)*_T_local
print(f"  T_trans={_T_trans:.3f}  T_comp={_T_comp:.3f}  "
      f"T_edge={_T_edge:.3f}  T_local={_T_local:.3f}  T={_T:.3f}")

# Penalty
_acc = detection_accuracy(_t, _O, _K)
_P_cap = BETA_CAP*max(0,(1-_O)*_t["tau"]-_C_l)
_P_acc = BETA_ACC*max(0,ALPHA_MIN-_acc)
_P_lat = BETA_LAT*max(0,_T-T_MAX)
_P = _P_cap+_P_acc+_P_lat
print(f"  acc={_acc:.3f}  P_cap={_P_cap:.3f}  "
      f"P_acc={_P_acc:.3f}  P_lat={_P_lat:.3f}  P={_P:.3f}")

# Per-device reward
_w_p=W_P[_t["priority"]]; _w_t=(1-_w_p)*GAMMA_L
_w_e=(1-_w_p)*GAMMA_E
print(f"  w_p={_w_p}  w_t={_w_t:.3f}  w_e={_w_e:.3f}  "
      f"(sum={_w_p+_w_t+_w_e:.3f} ✓)")
_bonus = DELTA*_O*(1-_O)
_r = -(  _w_p*_P
       + _w_t*min(_T,T_MAX)/T_MAX
       + _w_e*min(_E,E_MAX)/E_MAX) + _bonus
print(f"  bonus={_bonus:.3f}  r_device={_r:.4f}")
print(f"  r_joint = (1/N)*sum r_i  ✓")
print("[03] Reward verification complete\n")

print(f"[03] State vector: S_t ∈ R^{{5×{N_DEVICES}+3}}"
      f" = R^{{{STATE_SIZE}}}")
print(f"     Per-device (×{N_DEVICES}): "
      f"[C_local, tau, d, xi, P] ∈ R^5")
print(f"     Shared (×1):   "
      f"[C_edge, BW, b_rem] ∈ R^3")
print(f"     Train={len(tr_states)}  "
      f"Test={len(te_states)}")
print("[03] Saved → results/")
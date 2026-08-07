"""
04_drl_agent.py
===============
Neural network definitions:
  - MultiDeviceDuelingNet
  - ZhaoDQNNet
Imports this module — no direct execution needed.
"""
import os, random, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

config     = pickle.load(open("results/config.pkl","rb"))
N_DEVICES  = config["N_DEVICES"]
STATE_SIZE = config["STATE_SIZE"]
N_ACT_PER  = config["N_ACT_PER"]
OFFLOAD    = config["OFFLOAD"]
COMPRESSION= config["COMPRESSION"]
ACTIONS    = config["ACTIONS"]

# ── Dueling DDQN Network ──────────────────────────────────────
class MultiDeviceDuelingNet(nn.Module):
    """
    Shared backbone + N independent dueling heads.
    Backbone: R^{5N+3} → 512 → 256 → 128 (LayerNorm + SiLU)
    Per-device value head:    128 → 64 → 1
    Per-device advantage head:128 → 64 → 20
    Q^(i) = V^(i) + A^(i) - mean(A^(i))
    """
    def __init__(self, state_size, n_devices, n_act_per):
        super().__init__()
        self.n_devices = n_devices
        self.shared = nn.Sequential(
            nn.Linear(state_size,512),nn.LayerNorm(512),nn.SiLU(),
            nn.Linear(512,256),      nn.LayerNorm(256),nn.SiLU(),
            nn.Linear(256,128),      nn.LayerNorm(128),nn.SiLU())
        self.value_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
            for _ in range(n_devices)])
        self.adv_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128,64),nn.SiLU(),
                          nn.Linear(64,n_act_per))
            for _ in range(n_devices)])
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.shared(x); q_list=[]
        for i in range(self.n_devices):
            v=self.value_heads[i](f); a=self.adv_heads[i](f)
            q_list.append(v+a-a.mean(dim=1,keepdim=True))
        return q_list

# ── Zhao et al. DQN Network ───────────────────────────────────
class ZhaoDQNNet(nn.Module):
    """
    Zhao et al. (2022) — standard DQN, no Dueling, no semantic.
    Action: offload fraction only (5 levels, K=0 always).
    """
    def __init__(self, state_size, n_devices, n_offload=5):
        super().__init__()
        self.n_devices = n_devices
        self.shared = nn.Sequential(
            nn.Linear(state_size,256),nn.ReLU(),
            nn.Linear(256,128),       nn.ReLU(),
            nn.Linear(128,64),        nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Linear(64,n_offload) for _ in range(n_devices)])
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.shared(x)
        return [head(f) for head in self.heads]

# ── Prioritised Experience Replay ─────────────────────────────
class PER:
    def __init__(self, cap=60000, alpha=0.6,
                 beta0=0.4, beta_steps=5000):
        self.cap=cap; self.alpha=alpha; self.beta=beta0
        self.b_inc=(1.0-beta0)/beta_steps
        self.buf=[]; self.prios=np.zeros(cap,dtype=np.float32)
        self.pos=0; self.max_p=1.0

    def push(self, s, a, r, s2, d):
        tr=(s,a,r,s2,d)
        if len(self.buf)<self.cap: self.buf.append(tr)
        else: self.buf[self.pos]=tr
        self.prios[self.pos]=self.max_p
        self.pos=(self.pos+1)%self.cap

    def sample(self, bs):
        n=len(self.buf); probs=self.prios[:n]**self.alpha
        probs/=probs.sum()
        idxs=np.random.choice(n,bs,replace=False,p=probs)
        batch=[self.buf[i] for i in idxs]
        w=(n*probs[idxs])**(-self.beta); w/=w.max()
        self.beta=min(1.0,self.beta+self.b_inc)
        return batch,idxs,torch.FloatTensor(w)

    def update(self, idxs, errors):
        for i,e in zip(idxs,errors):
            p=float(abs(e)+1e-6)
            self.prios[i]=p; self.max_p=max(self.max_p,p)

    def __len__(self): return len(self.buf)

# ── DQN Proposed Agent ────────────────────────────────────────
class MultiDeviceAgent:
    def __init__(self):
        self.online = MultiDeviceDuelingNet(STATE_SIZE,N_DEVICES,N_ACT_PER)
        self.target = MultiDeviceDuelingNet(STATE_SIZE,N_DEVICES,N_ACT_PER)
        self.target.load_state_dict(self.online.state_dict())
        self.opt   = optim.AdamW(self.online.parameters(),
                                  lr=5e-4, weight_decay=1e-4)
        self.per   = PER()
        self.gamma = 0.99; self.eps=1.0
        self.eps_min=0.03; self.bs=256; self.n_upd=0

    def act(self, s, greedy=False):
        if not greedy and random.random()<self.eps:
            return [random.randint(0,N_ACT_PER-1)
                    for _ in range(N_DEVICES)]
        with torch.no_grad():
            q_list=self.online(torch.FloatTensor(s).unsqueeze(0))
            return [q.argmax().item() for q in q_list]

    def update(self):
        if len(self.per)<self.bs: return None
        batch,idxs,w=self.per.sample(self.bs)
        s,a_list,r,s2,d=zip(*batch)
        S =torch.FloatTensor(np.array(s))
        S2=torch.FloatTensor(np.array(s2))
        R =torch.FloatTensor(r).unsqueeze(1)
        D =torch.FloatTensor(d).unsqueeze(1)
        W =w.unsqueeze(1)
        total_loss=torch.tensor(0.0,requires_grad=True)
        td_errors =np.zeros(self.bs)
        q_on =self.online(S)
        with torch.no_grad():
            q_on2 =self.online(S2)
            q_tgt2=self.target(S2)
        for di in range(N_DEVICES):
            A_i=torch.LongTensor([a[di] for a in a_list]).unsqueeze(1)
            Q_i=q_on[di].gather(1,A_i)
            a2 =q_on2[di].argmax(1,keepdim=True)
            Q2 =q_tgt2[di].gather(1,a2)
            tgt=R+self.gamma*Q2*(1-D)
            li =(W*nn.SmoothL1Loss(reduction='none')(Q_i,tgt)).mean()
            total_loss=total_loss+li
            td_errors+=(tgt-Q_i).detach().squeeze().numpy()
        self.opt.zero_grad(); total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1.0)
        self.opt.step()
        self.per.update(idxs,td_errors/N_DEVICES)
        self.n_upd+=1
        if self.n_upd%200==0:
            for p_o,p_t in zip(self.online.parameters(),
                                self.target.parameters()):
                p_t.data.copy_(0.005*p_o.data+0.995*p_t.data)
        return total_loss.item()

    def decay_eps(self, ep, total):
        self.eps=self.eps_min+0.5*(1-self.eps_min)*(
            1+math.cos(math.pi*ep/total))

# ── Zhao et al. Agent ─────────────────────────────────────────
class ZhaoAgent:
    def __init__(self):
        self.online=ZhaoDQNNet(STATE_SIZE,N_DEVICES,n_offload=5)
        self.target=ZhaoDQNNet(STATE_SIZE,N_DEVICES,n_offload=5)
        self.target.load_state_dict(self.online.state_dict())
        self.opt=optim.Adam(self.online.parameters(),lr=1e-3)
        self.buf=[]; self.buf_cap=10000
        self.gamma=0.99; self.eps=1.0; self.eps_min=0.05
        self.bs=128; self.n_upd=0

    def _blind(self, s):
        s2=s.copy()
        for d in range(N_DEVICES): s2[d*5+4]=0.5
        return s2

    def act(self, s, greedy=False):
        sb=self._blind(s)
        if not greedy and random.random()<self.eps:
            return [random.randint(0,4) for _ in range(N_DEVICES)]
        with torch.no_grad():
            q=self.online(torch.FloatTensor(sb).unsqueeze(0))
            return [qi.argmax().item() for qi in q]

    def push(self, s, a, r, s2):
        if len(self.buf)>=self.buf_cap: self.buf.pop(0)
        self.buf.append((self._blind(s),a,r,self._blind(s2)))

    def update(self):
        if len(self.buf)<self.bs: return None
        batch=random.sample(self.buf,self.bs)
        s,a_list,r,s2=zip(*batch)
        S =torch.FloatTensor(np.array(s))
        S2=torch.FloatTensor(np.array(s2))
        R =torch.FloatTensor(r).unsqueeze(1)
        total_loss=torch.tensor(0.0,requires_grad=True)
        q_on=self.online(S)
        with torch.no_grad(): q_tgt=self.target(S2)
        for di in range(N_DEVICES):
            A_i=torch.LongTensor([a[di] for a in a_list]).unsqueeze(1)
            Q_i=q_on[di].gather(1,A_i)
            Q2 =q_tgt[di].max(1,keepdim=True)[0]
            total_loss=total_loss+nn.MSELoss()(Q_i,R+self.gamma*Q2)
        self.opt.zero_grad(); total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1.0)
        self.opt.step(); self.n_upd+=1
        if self.n_upd%500==0:
            self.target.load_state_dict(self.online.state_dict())
        return total_loss.item()

    def decay_eps(self, ep, total):
        self.eps=max(self.eps_min,1.0-ep/total)

if __name__=="__main__":
    agent=MultiDeviceAgent()
    s=np.zeros(STATE_SIZE,dtype=np.float32)
    q=agent.online(torch.FloatTensor(s).unsqueeze(0))
    print(f"[04] Network OK — {N_DEVICES} heads, "
          f"each Q∈R^{N_ACT_PER}")
    print(f"[04] Total params: "
          f"{sum(p.numel() for p in agent.online.parameters()):,}")
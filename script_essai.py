"""
OPF triphasé (4 fils) multi‑période — capacité d’accueil PV & contrôle coordonné PV+ESS
Version : v2 — profils horaires réels + Ybus raffiné (mutuelles, neutre) + pertes I²R + export CSV

⚙️ Points clés
- Entrées optionnelles (CSV, même répertoire que ce script) :
  • branches.csv        → topologie & paramètres électriques par tronçon
  • profiles_load.csv   → P/Q par BUS‑PHASE‑TEMPS (colonnes : bus, phase, t, P_kW, Q_kVAr)
  • profiles_pv_pu.csv  → profil PV p.u. par pas de temps (colonnes : t, pv_pu)
  Si absents, des valeurs par défaut sont utilisées.
- Unités : modèle en **pu**. Choisir Sbase et Vbase ci‑dessous ; R/X sont converties via Zbase.
- Deux passes :
  • mode='capacity'  : maximise PPVmax (capacité d’accueil)
  • mode='quality'   : minimise les écarts de tension (et calcule pertes, VUF, SOC) avec PPVmax fixé
- Exports (CSV) :
  • export_voltages.csv  (bus, phase, t, U_mag_pu, U_re, U_im)
  • export_vuf.csv       (bus, t, VUF)
  • export_soc.csv       (bus_ESS, t, SOC)
  • export_losses.csv    (t, from, to, phase, I_mag_pu, R_pu, P_loss_kW)

NB : Ce squelette est compact pour clarté ; adaptez les paramètres à vos données réelles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math, os
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# =============================
# 0) Bases et constantes réseau
# =============================
@dataclass
class Bases:
    Sbase_kVA: float = 100.0      # ⚠️ adapter (ex: 630 au poste si vous voulez)
    Vbase_LL_V: float = 400.0     # tension composée BT (V)

    @property
    def Vbase_phase_V(self) -> float:
        return self.Vbase_LL_V / math.sqrt(3)

    @property
    def Zbase_ohm(self) -> float:
        # Zbase = Vbase_phase^2 / Sbase (kVA → kW ~ kVA à pf≈1 ; ici on garde kVA)
        return (self.Vbase_phase_V ** 2) / (self.Sbase_kVA * 1000.0)

BASE = Bases()
PHASES = ['a','b','c','n']
PHASES_ABC = ['a','b','c']
PHASE_IDX = {p:i for i,p in enumerate(PHASES)}

# Limites réseau
U_MIN = 0.95
U_MAX = 1.05
VUF_MAX = 0.02  # 2 %
PV_INVERTER_S_OVER_P = 1.10
ESS_SOC_MIN, ESS_SOC_MAX = 0.20, 0.80
ETA_CH, ETA_DIS = 0.94, 0.94
DT_H = 1.0

# =============================
# 1) Topologie & profils (CSV)
# =============================
# Fichiers attendus (optionnels)
F_BRANCH = 'branches.csv'
F_LOAD   = 'profiles_load.csv'
F_PVPU   = 'profiles_pv_pu.csv'

# Topologie par défaut (figure 1→28)
EDGES_DEFAULT = [
    (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),(11,12),
    (1,13),(13,14),
    (1,15),(15,16),(16,17),(17,18),
    (1,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28)
]
ALL_BUSES = sorted({b for e in EDGES_DEFAULT for b in e})
T_DEFAULT = list(range(24))

# Profils par défaut (remplacer via CSV pour les données réelles)
pv_pu_default = np.array([0,0,0,0,0,0.05,0.20,0.45,0.70,0.85,0.95,1.0,1.0,0.95,0.80,0.50,0.25,0.10,0.02,0,0,0,0,0], dtype=float)
LOAD_P0_default = {2:(30.673,32.670,31.741), 8:(0.0043,25.887,23.504), 13:(1.8312,2.8787,0.025), 15:(23.732,30.932,11.825), 19:(27.459,32.394,0.026)}
LOAD_Q0_default = {b:(0.0,0.0,0.0) for b in LOAD_P0_default}
ESS_NODES_DEFAULT = [4,13]
ESS_P_MAX_kW = 4.0
CAP_kWh_default = 20.0

# Lecture des CSV si présents

def load_profiles():
    # PV p.u.
    if os.path.exists(F_PVPU):
        dfpv = pd.read_csv(F_PVPU)
        pv_vec = dfpv.sort_values('t')['pv_pu'].to_numpy(dtype=float)
    else:
        pv_vec = pv_pu_default

    # Charges P/Q
    LoadP: Dict[Tuple[int,str,int], float] = {}
    LoadQ: Dict[Tuple[int,str,int], float] = {}
    if os.path.exists(F_LOAD):
        dfl = pd.read_csv(F_LOAD)
        for _,r in dfl.iterrows():
            b = int(r['bus']); p = str(r['phase']).strip().lower(); t = int(r['t'])
            LoadP[(b,p,t)] = float(r['P_kW']); LoadQ[(b,p,t)] = float(r['Q_kVAr'])
        Tset = sorted({k[2] for k in LoadP.keys()})
    else:
        # construit 24h constants à partir des quelques points par défaut
        Tset = T_DEFAULT
        for t in Tset:
            for b,(Pa,Pb,Pc) in LOAD_P0_default.items():
                LoadP[(b,'a',t)]=Pa; LoadP[(b,'b',t)]=Pb; LoadP[(b,'c',t)]=Pc
            for b,(Qa,Qb,Qc) in LOAD_Q0_default.items():
                LoadQ[(b,'a',t)]=Qa; LoadQ[(b,'b',t)]=Qb; LoadQ[(b,'c',t)]=Qc

    return pv_vec, LoadP, LoadQ, Tset

@dataclass
class BranchParam:
    i:int; j:int; length_m:float
    # R/X par km et par phase (ohm/km)
    Rkm_a:float; Rkm_b:float; Rkm_c:float; Rkm_n:float
    Xkm_a:float; Xkm_b:float; Xkm_c:float; Xkm_n:float
    k_mutual:float=0.01     # couplages mutuels phase‑phase
    k_mutual_n:float=0.01   # couplages phase‑neutre


def load_branches():
    branches: List[BranchParam] = []
    if os.path.exists(F_BRANCH):
        df = pd.read_csv(F_BRANCH)
        for _,r in df.iterrows():
            branches.append(BranchParam(
                i=int(r['from_bus']), j=int(r['to_bus']), length_m=float(r['length_m']),
                Rkm_a=float(r['R_ohm_km_a']), Rkm_b=float(r['R_ohm_km_b']), Rkm_c=float(r['R_ohm_km_c']), Rkm_n=float(r.get('R_ohm_km_n', r['R_ohm_km_a'])),
                Xkm_a=float(r['X_ohm_km_a']), Xkm_b=float(r['X_ohm_km_b']), Xkm_c=float(r['X_ohm_km_c']), Xkm_n=float(r.get('X_ohm_km_n', r['X_ohm_km_a'])),
                k_mutual=float(r.get('k_mutual', 0.01)), k_mutual_n=float(r.get('k_mutual_n', 0.01))
            ))
        edges = [(b.i,b.j) for b in branches]
        buses = sorted({x for e in edges for x in e})
    else:
        # défaut : mêmes impédances par tronçon, proportionnelles à la longueur totale/nb segments (valeurs indicatives)
        edges = EDGES_DEFAULT
        buses = ALL_BUSES
        # valeurs indicatives : cuivre 110 mm2 ≈ R≈0.30 Ω/km, X≈0.08 Ω/km ; 35 mm2 ≈ R≈0.9 Ω/km, X≈0.10 Ω/km
        def guess(ix):
            if ix in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]:
                return 110,0.30,0.08
            if ix in [(1,8),(8,9),(9,10),(10,11),(11,12)]:
                return 110,0.30,0.08
            if ix in [(1,13),(13,14)]:
                return 35,0.90,0.10
            if ix in [(1,15),(15,16),(16,17),(17,18)]:
                return 110,0.30,0.08
            return 110,0.30,0.08
        # répartir longueurs moyennes par départ
        lengths = {
            'D1':602.79/6,'D2':524.28/5,'D3':111.93/2,'D4':420.02/4,'D5':1145.28/10
        }
        for e in edges:
            sec,Rkm,Xkm = guess(e)
            if e in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]: L=lengths['D1']
            elif e in [(1,8),(8,9),(9,10),(10,11),(11,12)]: L=lengths['D2']
            elif e in [(1,13),(13,14)]: L=lengths['D3']
            elif e in [(1,15),(15,16),(16,17),(17,18)]: L=lengths['D4']
            else: L=lengths['D5']
            branches.append(BranchParam(e[0],e[1],L, Rkm,Rkm,Rkm,Rkm, Xkm,Xkm,Xkm,Xkm, 0.01,0.01))
    return branches, edges, buses

# ==================================
# 2) Construction Ybus (pu), 4 fils
# ==================================

def Z4x4_from_branch(b: BranchParam):
    Lkm = b.length_m/1000.0
    za = complex(b.Rkm_a*Lkm, b.Xkm_a*Lkm)
    zb = complex(b.Rkm_b*Lkm, b.Xkm_b*Lkm)
    zc = complex(b.Rkm_c*Lkm, b.Xkm_c*Lkm)
    zn = complex(b.Rkm_n*Lkm, b.Xkm_n*Lkm)
    # Matrice Z (ohm)
    Z = np.zeros((4,4), dtype=complex)
    Z[0,0],Z[1,1],Z[2,2],Z[3,3] = za,zb,zc,zn
    # mutuelles phase‑phase : valeur moyenne pondérée
    zpp = (abs(za)+abs(zb)+abs(zc))/3.0
    zpn = zpp
    for i in range(3):
        for j in range(3):
            if i!=j:
                Z[i,j] = b.k_mutual * zpp
    # phase‑neutre
    for i in range(3):
        Z[i,3] = b.k_mutual_n * zpn
        Z[3,i] = b.k_mutual_n * zpn
    return Z


def build_Ybus_pu(branches: List[BranchParam], buses: List[int]):
    N = len(buses)
    bus_index = {b:i for i,b in enumerate(buses)}
    Ybus = np.zeros((4*N,4*N), dtype=complex)
    line_data: Dict[Tuple[int,int], Dict[str,np.ndarray]] = {}

    for br in branches:
        Zi = Z4x4_from_branch(br)                           # ohm
        Zi_pu = Zi / BASE.Zbase_ohm                         # pu
        Yi_pu = np.linalg.inv(Zi_pu)                        # pu
        i = bus_index[br.i]; j = bus_index[br.j]
        # estampillage 4×4
        for a in range(4):
            for b in range(4):
                Ybus[i*4+a, i*4+b] += Yi_pu[a,b]
                Ybus[j*4+a, j*4+b] += Yi_pu[a,b]
                Ybus[i*4+a, j*4+b] -= Yi_pu[a,b]
                Ybus[j*4+a, i*4+b] -= Yi_pu[a,b]
        line_data[(br.i,br.j)] = {'Zpu':Zi_pu, 'Ypu':Yi_pu}

    # Ancrage slack (bus minimal = référence)
    slack = min(buses)
    for a in range(4):
        for b in range(4):
            Ybus[(slack-1)*4+a,(slack-1)*4+b] = 0.0
        for d in range(4):
            Ybus[(slack-1)*4+d,(slack-1)*4+d] = 1.0

    return Ybus, line_data, bus_index

# ==================================
# 3) Modèle Pyomo (I–V) en pu
# ==================================

def build_opf_model(mode: str='capacity', pv_cap_fixed: float|None=None,
                    ess_nodes: List[int]|None=None):
    assert mode in {'capacity','quality'}

    branches, edges, buses = load_branches()
    Ybus, Yline, bus_index = build_Ybus_pu(branches, buses)

    pv_vec, LoadP, LoadQ, Tset = load_profiles()

    m = pyo.ConcreteModel()
    m.BUS = pyo.Set(initialize=buses)
    m.T   = pyo.Set(initialize=Tset)
    m.PH  = pyo.Set(initialize=PHASES)

    # Variables de tension (pu)
    m.Ure = pyo.Var(m.BUS, m.PH, m.T, initialize=1.0)
    m.Uim = pyo.Var(m.BUS, m.PH, m.T, initialize=0.0)

    # Courants nodaux (pu)
    m.Ire = pyo.Var(m.BUS, m.PH, m.T, initialize=0.0)
    m.Iim = pyo.Var(m.BUS, m.PH, m.T, initialize=0.0)

    # Injections nettes par phase (pu de Sbase_kVA)
    m.Pnet = pyo.Var(m.BUS, m.PH, m.T, initialize=0.0)
    m.Qnet = pyo.Var(m.BUS, m.PH, m.T, initialize=0.0)

    # PV
    if mode=='capacity' and pv_cap_fixed is None:
        m.PPVmax = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0) # kW de crête par nœud (interprétation simple)
    else:
        m.PPVmax = pyo.Param(initialize=(pv_cap_fixed if pv_cap_fixed is not None else 0.0), mutable=False)

    m.Ppv = pyo.Var(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.Qpv = pyo.Var(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, initialize=0.0)

    # ESS
    ESS_NODES = ess_nodes if ess_nodes is not None else ESS_NODES_DEFAULT
    m.Pess = pyo.Var(m.BUS, m.T, initialize=0.0)  # + décharge, − charge (kW base=Sbase)
    m.SOC  = pyo.Var(m.BUS, m.T, bounds=(ESS_SOC_MIN, ESS_SOC_MAX), initialize=(ESS_SOC_MIN+ESS_SOC_MAX)/2)
    m.Pchg = pyo.Var(m.BUS, m.T, bounds=(0,None), initialize=0.0)
    m.Pdis = pyo.Var(m.BUS, m.T, bounds=(0,None), initialize=0.0)

    # Paramètres
    t_sorted = sorted(list(m.T.data()))
    pv_series = {t: float(pv_vec[t]) if t < len(pv_vec) else 0.0 for t in t_sorted}
    m.PVpu = pyo.Param(m.T, initialize=lambda m,tt: pv_series[int(tt)], mutable=True)

    # Charges (kW/kVAr) -> en pu : / Sbase
    def init_LoadP(m,b,p,tt):
        return LoadP.get((int(b),str(p),int(tt)), 0.0) / BASE.Sbase_kVA
    def init_LoadQ(m,b,p,tt):
        return LoadQ.get((int(b),str(p),int(tt)), 0.0) / BASE.Sbase_kVA
    m.LoadP = pyo.Param(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, initialize=init_LoadP, mutable=True)
    m.LoadQ = pyo.Param(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, initialize=init_LoadQ, mutable=True)

    # 3.1) I = Y·V (réel/imag séparés)
    G = Ybus.real; B = Ybus.imag
    bus_to_idx = lambda n,p: (buses.index(n))*4 + PHASE_IDX[p]

    def Ire_rule(m,n,p,tt):
        k = bus_to_idx(n,p)
        return m.Ire[n,p,tt] == sum(G[k, bus_to_idx(b, q)]*m.Ure[b,q,tt] - B[k, bus_to_idx(b, q)]*m.Uim[b,q,tt]
                                    for b in m.BUS for q in m.PH)
    def Iim_rule(m,n,p,tt):
        k = bus_to_idx(n,p)
        return m.Iim[n,p,tt] == sum(G[k, bus_to_idx(b, q)]*m.Uim[b,q,tt] + B[k, bus_to_idx(b, q)]*m.Ure[b,q,tt]
                                    for b in m.BUS for q in m.PH)
    m.Ire_con = pyo.Constraint(m.BUS, m.PH, m.T, rule=Ire_rule)
    m.Iim_con = pyo.Constraint(m.BUS, m.PH, m.T, rule=Iim_rule)

    # 3.2) Lien puissance‑constante : I*|U|² = [P,Q]·[U]
    def PQ_link_re(m,n,p,tt):
        if p=='n':
            return m.Ire[n,p,tt] == 0.0
        denom = m.Ure[n,p,tt]**2 + m.Uim[n,p,tt]**2 + 1e-7
        return m.Ire[n,p,tt]*denom == m.Pnet[n,p,tt]*m.Ure[n,p,tt] + m.Qnet[n,p,tt]*m.Uim[n,p,tt]
    def PQ_link_im(m,n,p,tt):
        if p=='n':
            return m.Iim[n,p,tt] == 0.0
        denom = m.Ure[n,p,tt]**2 + m.Uim[n,p,tt]**2 + 1e-7
        return m.Iim[n,p,tt]*denom == m.Pnet[n,p,tt]*m.Uim[n,p,tt] - m.Qnet[n,p,tt]*m.Ure[n,p,tt]
    m.PQ_link_re = pyo.Constraint(m.BUS, m.PH, m.T, rule=PQ_link_re)
    m.PQ_link_im = pyo.Constraint(m.BUS, m.PH, m.T, rule=PQ_link_im)

    # 3.3) Définition des injections nettes : PV + ESS − LOAD
    def pnet_rule(m,b,p,tt):
        if p=='n':
            return m.Pnet[b,p,tt] == 0.0
        ess_share = (m.Pess[b,tt]/3.0) if (int(b) in ESS_NODES) else 0.0
        return m.Pnet[b,p,tt] == m.Ppv[b,p,tt] + ess_share - m.LoadP[b,p,tt]
    def qnet_rule(m,b,p,tt):
        if p=='n':
            return m.Qnet[b,p,tt] == 0.0
        return m.Qnet[b,p,tt] == m.Qpv[b,p,tt] - m.LoadQ[b,p,tt]
    m.Pnet_def = pyo.Constraint(m.BUS, m.PH, m.T, rule=pnet_rule)
    m.Qnet_def = pyo.Constraint(m.BUS, m.PH, m.T, rule=qnet_rule)

    # 3.4) PV : bornes & cercle de capacité (pu de Sbase)
    def pv_cap_rule(m,b,p,tt):
        PPVmax = m.PPVmax if isinstance(m.PPVmax,pyo.Var) else pyo.value(m.PPVmax)
        return m.Ppv[b,p,tt] <= (PPVmax/BASE.Sbase_kVA) * m.PVpu[tt]
    m.PV_cap = pyo.Constraint(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, rule=pv_cap_rule)

    def pv_circle_rule(m,b,p,tt):
        PPVmax = m.PPVmax if isinstance(m.PPVmax,pyo.Var) else pyo.value(m.PPVmax)
        Smax = PV_INVERTER_S_OVER_P * (PPVmax/BASE.Sbase_kVA) * m.PVpu[tt]
        return m.Ppv[b,p,tt]**2 + m.Qpv[b,p,tt]**2 <= (Smax**2 + 1e-8)
    m.PV_circle = pyo.Constraint(m.BUS, pyo.Set(initialize=PHASES_ABC), m.T, rule=pv_circle_rule)

    # 3.5) ESS : split Pess = Pdis − Pchg ; limites ; dynamique SOC ; boucle
    def ess_split(m,b,tt):
        if int(b) not in ESS_NODES: return pyo.Constraint.Skip
        return m.Pess[b,tt] == m.Pdis[b,tt] - m.Pchg[b,tt]
    m.ess_split = pyo.Constraint(m.BUS, m.T, rule=ess_split)

    def ess_limits(m,b,tt):
        if int(b) not in ESS_NODES: return pyo.Constraint.Skip
        return pyo.inequality(-ESS_P_MAX_kW/BASE.Sbase_kVA, m.Pess[b,tt], ESS_P_MAX_kW/BASE.Sbase_kVA)
    m.ess_lims = pyo.Constraint(m.BUS, m.T, rule=ess_limits)

    def soc_evol(m,b,tt):
        if int(b) not in ESS_NODES: return pyo.Constraint.Skip
        if tt==t_sorted[0]: return pyo.Constraint.Skip
        return m.SOC[b,tt] == m.SOC[b,tt-1] + (ETA_CH*m.Pchg[b,tt] - m.Pdis[b,tt]/ETA_DIS) * DT_H / (CAP_kWh_default/BASE.Sbase_kVA)
    m.soc_dyn = pyo.Constraint(m.BUS, m.T, rule=soc_evol)

    def soc_cycle(m,b):
        if int(b) not in ESS_NODES: return pyo.Constraint.Skip
        return m.SOC[b,t_sorted[0]] == m.SOC[b,t_sorted[-1]]
    m.soc_loop = pyo.Constraint(m.BUS, rule=soc_cycle)

    # 3.6) Contraintes tension & VUF
    def vmag_rule(m,b,p,tt):
        if p=='n': return pyo.Constraint.Skip
        return pyo.inequality(U_MIN**2, m.Ure[b,p,tt]**2 + m.Uim[b,p,tt]**2, U_MAX**2)
    m.Vmag = pyo.Constraint(m.BUS, m.PH, m.T, rule=vmag_rule)

    alpha = complex(math.cos(2*math.pi/3), math.sin(2*math.pi/3))
    alpha2 = alpha**2
    def vuf_rule(m,b,tt):
        Va_re,Va_im = m.Ure[b,'a',tt], m.Uim[b,'a',tt]
        Vb_re,Vb_im = m.Ure[b,'b',tt], m.Uim[b,'b',tt]
        Vc_re,Vc_im = m.Ure[b,'c',tt], m.Uim[b,'c',tt]
        Vp_re = Va_re + (alpha2.real)*Vb_re - (alpha2.imag)*Vb_im + (alpha.real)*Vc_re - (alpha.imag)*Vc_im
        Vp_im = Va_im + (alpha2.real)*Vb_im + (alpha2.imag)*Vb_re + (alpha.real)*Vc_im + (alpha.imag)*Vc_re
        Vm_re = Va_re + (alpha.real)*Vb_re - (alpha.imag)*Vb_im + (alpha2.real)*Vc_re - (alpha2.imag)*Vc_im
        Vm_im = Va_im + (alpha.real)*Vb_im + (alpha.imag)*Vb_re + (alpha2.real)*Vc_im + (alpha2.imag)*Vc_re
        return Vm_re**2 + Vm_im**2 <= (VUF_MAX**2) * (Vp_re**2 + Vp_im**2) + 1e-10
    m.VUF = pyo.Constraint(m.BUS, m.T, rule=vuf_rule)

    # 3.7) Slack bus (bus min)
    slack = min(buses)  # Bus de référence

    def slack_fix_real(m, p, tt):
        if p == 'a':
            return m.Ure[slack, p, tt] == 1.0
        if p == 'b':
            return m.Ure[slack, p, tt] == -0.5
        if p == 'c':
            return m.Ure[slack, p, tt] == -0.5
        return m.Ure[slack, p, tt] == 0.0

    def slack_fix_imag(m, p, tt):
        if p == 'a':
            return m.Uim[slack, p, tt] == 0.0
        if p == 'b':
            return m.Uim[slack, p, tt] == -math.sqrt(3)/2
        if p == 'c':
            return m.Uim[slack, p, tt] == math.sqrt(3)/2
        return m.Uim[slack, p, tt] == 0.0

    # Définition des contraintes dans le modèle
    m.slack_real = pyo.Constraint(m.PH, m.T, rule=slack_fix_real)
    m.slack_imag = pyo.Constraint(m.PH, m.T, rule=slack_fix_imag)


    # 3.8) Objectif
    def devU_sum():
        return sum((pyo.sqrt(m.Ure[b,p,tt]**2 + m.Uim[b,p,tt]**2) - 1.0)**2 for b in m.BUS for p in PHASES_ABC for tt in m.T)

    if mode=='capacity' and isinstance(m.PPVmax,pyo.Var):
        m.obj = pyo.Objective(expr=m.PPVmax, sense=pyo.maximize)
    else:
        m.obj = pyo.Objective(expr=devU_sum(), sense=pyo.minimize)

    # Attacher des objets utiles pour post‑traitement
    m._Yline = Yline
    m._buses = buses
    m._edges = edges
    m._BASE = BASE
    m._ESSN = ESS_NODES
    m._t_sorted = t_sorted
    return m

# ==================================
# 4) Post‑traitement : tensions, VUF, pertes I²R & exports
# ==================================

def export_results(model: pyo.ConcreteModel, tag: str=''):
    m = model
    BASE = m._BASE
    buses = m._buses
    edges = m._edges
    Yline = m._Yline
    t_sorted = m._t_sorted

    # Voltages, VUF, SOC
    rows_v = []
    rows_vuf = []
    rows_soc = []

    alpha = complex(math.cos(2*math.pi/3), math.sin(2*math.pi/3))
    alpha2 = alpha**2

    for b in buses:
        for tt in t_sorted:
            # VUF
            Va = complex(pyo.value(m.Ure[b,'a',tt]), pyo.value(m.Uim[b,'a',tt]))
            Vb = complex(pyo.value(m.Ure[b,'b',tt]), pyo.value(m.Uim[b,'b',tt]))
            Vc = complex(pyo.value(m.Ure[b,'c',tt]), pyo.value(m.Uim[b,'c',tt]))
            Vp = Va + alpha2*Vb + alpha*Vc
            Vm = Va + alpha*Vb + alpha2*Vc
            VUF = abs(Vm)/max(abs(Vp),1e-9)
            rows_vuf.append({'bus':b,'t':tt,'VUF':VUF})
            # tensions par phase
            for p in PHASES_ABC:
                Ure = pyo.value(m.Ure[b,p,tt]); Uim = pyo.value(m.Uim[b,p,tt])
                rows_v.append({'bus':b,'phase':p,'t':tt,'U_mag_pu':math.sqrt(Ure*Ure+Uim*Uim),'U_re':Ure,'U_im':Uim})
            # SOC (si ESS)
            if b in m._ESSN:
                rows_soc.append({'bus':b,'t':tt,'SOC':pyo.value(m.SOC[b,tt])})

    dfv = pd.DataFrame(rows_v)
    dfvuf = pd.DataFrame(rows_vuf)
    dfsoc = pd.DataFrame(rows_soc) if rows_soc else pd.DataFrame(columns=['bus','t','SOC'])

    # Pertes par tronçon/phase
    rows_loss = []
    def Vvec(bus,tt):
        return np.array([
            pyo.value(m.Ure[bus,'a',tt]) + 1j*pyo.value(m.Uim[bus,'a',tt]),
            pyo.value(m.Ure[bus,'b',tt]) + 1j*pyo.value(m.Uim[bus,'b',tt]),
            pyo.value(m.Ure[bus,'c',tt]) + 1j*pyo.value(m.Uim[bus,'c',tt]),
            pyo.value(m.Ure[bus,'n',tt]) + 1j*pyo.value(m.Uim[bus,'n',tt]),
        ], dtype=complex)

    Sbase_kW = BASE.Sbase_kVA  # ~kW (pf≈1)

    for tt in t_sorted:
        for (i,j), dat in Yline.items():
            Zpu = dat['Zpu']; Ypu = dat['Ypu']
            Vi = Vvec(i,tt); Vj = Vvec(j,tt)
            dV = Vi - Vj
            I = Ypu @ dV  # 4×1 courant par conducteur (pu)
            for k,phase in enumerate(PHASES_ABC):
                Ipu = I[k]
                Rpu = Zpu[k,k].real
                Ploss_pu = (abs(Ipu)**2) * Rpu
                Ploss_kW = Ploss_pu * Sbase_kW
                rows_loss.append({'t':tt,'from':i,'to':j,'phase':phase,'I_mag_pu':abs(Ipu),'R_pu':Rpu,'P_loss_kW':Ploss_kW})
            # neutre (optionnel)
            k=3; Ipu = I[k]; Rpu = Zpu[k,k].real
            Ploss_kW_n = (abs(Ipu)**2)*Rpu*Sbase_kW
            rows_loss.append({'t':tt,'from':i,'to':j,'phase':'n','I_mag_pu':abs(Ipu),'R_pu':Rpu,'P_loss_kW':Ploss_kW_n})

    dfloss = pd.DataFrame(rows_loss)

    # Exports
    suf = (('_'+tag) if tag else '')
    dfv.to_csv(f'export_voltages{suf}.csv', index=False)
    dfvuf.to_csv(f'export_vuf{suf}.csv', index=False)
    dfsoc.to_csv(f'export_soc{suf}.csv', index=False)
    dfloss.to_csv(f'export_losses{suf}.csv', index=False)
    print('CSV écrits : export_voltages, export_vuf, export_soc, export_losses')

# ==================================
# 5) Exécution exemple
# ==================================
if __name__ == '__main__':
    # Passe 1 : capacité d’accueil
    m1 = build_opf_model(mode='capacity')
    solver = pyo.SolverFactory('ipopt')
    res1 = solver.solve(m1, tee=False)
    if isinstance(m1.PPVmax,pyo.Var):
        PPVmax_kW = pyo.value(m1.PPVmax)
    else:
        PPVmax_kW = pyo.value(m1.PPVmax)
    print(f"Capacité d’accueil PV ≈ {PPVmax_kW:.2f} kW (par nœud candidat, interprétation simple)")

    # Passe 2 : qualité avec PPVmax fixé
    m2 = build_opf_model(mode='quality', pv_cap_fixed=PPVmax_kW)
    res2 = solver.solve(m2, tee=False)
    print('Optimisation qualité terminée.')

    # Exports CSV
    export_results(m2, tag='qualite')

    # Exemple d’affichage
    b=8; t=12
    print('U(8,b,12) = ', pyo.value(m2.Ure[b,'b',t]), '+ j', pyo.value(m2.Uim[m2._buses[1],'b',t]))

import matplotlib.pyplot as plt

# Exemple de graphe pour la tension du bus 2
voltages_bus2 = [pyo.value(model.Ure['2','a',t])**2 + pyo.value(model.Uim['2','a',t])**2
                 for t in model.T]

plt.figure(figsize=(8,4))
plt.plot(list(model.T), voltages_bus2, marker='o')
plt.xlabel("Temps (h)")
plt.ylabel("Tension² (p.u.)")
plt.title("Profil de tension Bus 2 - Phase a")
plt.grid(True)
plt.show()

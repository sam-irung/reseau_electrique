#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sc_rx_elec_singlefile.py
Script unifié : données en dur (dictionnaires) -> réseau pandapower -> scénarios PV (0,20,50%) -> ESS (optimisation multi-periode via Pyomo ou heuristique)
Génère 14 graphiques dans outputs/
Labels/legendes en français, fichiers en anglais.
"""

from __future__ import annotations
import os, copy, math, json, traceback
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# heavy optional imports
try:
    import pandapower as pp
except Exception as e:
    pp = None
    _pp_err = e

try:
    import pyomo.environ as pyo
except Exception as e:
    pyo = None
    _pyo_err = e

# ---------------------------
# Simulation settings & DATA
# ---------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120

# SimulationSettings (weights from article -> user-specified)
SimulationSettings = {
    "weights": {"w1": 0.5, "w2": 0.4, "w3": 0.1},    # ω1, ω2, ω3
    "k_V_per_kW": 0.002,       # pu per kW ESS impact on local voltage (paramétrable)
    "k_loss_per_kW": 0.02,     # kW losses reduction per kW ESS dispatched (proxy)
    "timesteps": ["2025-09-22 06:00", "2025-09-22 09:00", "2025-09-22 12:00", "2025-09-22 15:00", "2025-09-22 18:00"]
}

# === données intégrées (tes dictionnaires) ===
buses = [
    {"bus": "B1", "depart": 0, "vn_kv": 0.4, "is_slack": 1},
    {"bus": "B2", "depart": 1, "vn_kv": 0.4, "is_slack": 0},
    {"bus": "B3", "depart": 2, "vn_kv": 0.4, "is_slack": 0},
    {"bus": "B4", "depart": 3, "vn_kv": 0.4, "is_slack": 0},
    {"bus": "B5", "depart": 4, "vn_kv": 0.4, "is_slack": 0},
    {"bus": "B6", "depart": 5, "vn_kv": 0.4, "is_slack": 0},
]

transformer = [
    {
        "trafo_id": "T1",
        "sn_kva": 630,
        "vn_hv_kv": 15,
        "vn_lv_kv": 0.4,
        "vk_percent": 6,
        "vkr_percent": 1.2,
        "pfe_kw": 1.2,
        "vector_group": "Dyn11",
    }
]

line = [
    {"line_id": "L1", "from_bus": "B1", "to_bus": "B2", "length_m": 607.8, "section_mm2": 110, "r_ohm_per_km": 0.65, "x_ohm_per_km": 0.412},
    {"line_id": "L2", "from_bus": "B1", "to_bus": "B3", "length_m": 524.3, "section_mm2": 110, "r_ohm_per_km": 0.65, "x_ohm_per_km": 0.412},
    {"line_id": "L3", "from_bus": "B1", "to_bus": "B4", "length_m": 111.9, "section_mm2": 35, "r_ohm_per_km": 0.65, "x_ohm_per_km": 0.412},
    {"line_id": "L4", "from_bus": "B1", "to_bus": "B5", "length_m": 420,   "section_mm2": 110, "r_ohm_per_km": 0.65, "x_ohm_per_km": 0.412},
    {"line_id": "L5", "from_bus": "B1", "to_bus": "B6", "length_m": 1145, "section_mm2": 110, "r_ohm_per_km": 0.65, "x_ohm_per_km": 0.412},
]

limits = {
    "vmin_pu": 0.95,
    "vmax_pu": 1.05,
    "i_loading_max_percent": 100,
    "vuf_max_percent": 3
}

pv_static = [
    {"pv_id": "PV1", "bus": "B1", "phase": "A", "p_stc_kw": 5, "s_max_kva": 5.5},
    {"pv_id": "PV2", "bus": "B1", "phase": "B", "p_stc_kw": 5, "s_max_kva": 5.5},
    {"pv_id": "PV3", "bus": "B1", "phase": "C", "p_stc_kw": 5, "s_max_kva": 5.5},
]

ess = [
    {"ess_id": "ESS1", "bus": "B1", "phase": "3ph", "e_cap_kwh": 20, "p_max_kw_per_phase": 4, "eta_charge": 0.94, "eta_discharge": 0.94, "soc_init_percent": 50},
    {"ess_id": "ESS2", "bus": "B1", "phase": "3ph", "e_cap_kwh": 20, "p_max_kw_per_phase": 4, "eta_charge": 0.94, "eta_discharge": 0.94, "soc_init_percent": 50},
]

load_timeseries = [
    {"time": "2025-09-22 06:00", "bus": "B5", "phase": "A", "p_kw": 20, "q_kvar": 5},
    {"time": "2025-09-22 07:00", "bus": "B5", "phase": "B", "p_kw": 22, "q_kvar": 5.2},
    {"time": "2025-09-22 08:00", "bus": "B5", "phase": "C", "p_kw": 18, "q_kvar": 4.8},
]

pv_timeseries = [
    {"time": "2025-09-22 06:00", "pv_id": "PV1", "p_kw": 0,   "q_kvar": 0},
    {"time": "2025-09-22 12:00", "pv_id": "PV1", "p_kw": 4.5, "q_kvar": 0},
    {"time": "2025-09-22 18:00", "pv_id": "PV1", "p_kw": 0.1, "q_kvar": 0},
]

results_before = [
    {"bus": "B12", "phase": "A", "v_pu": 0.93, "i_line_A": 138, "vuf_percent": 7.1, "losses_kW_branch": 1.2},
    {"bus": "B12", "phase": "B", "v_pu": 0.96, "i_line_A": 146, "vuf_percent": 7.1, "losses_kW_branch": 1.1},
    {"bus": "B12", "phase": "C", "v_pu": 0.94, "i_line_A": 132, "vuf_percent": 7.1, "losses_kW_branch": 1.05}
]

# ---------------------------
# Utilitaires
# ---------------------------
def debug(msg: str):
    print(msg)

def safe_float(x, default=0.0):
    try:
        if x is None: return default
        s = str(x).strip().replace(',', '.').replace(';', '.')
        return float(s)
    except Exception:
        return default

def check_pandapower():
    if pp is None:
        raise ImportError(f"pandapower non disponible: {_pp_err}")

# ---------------------------
# Construction réseau (pandapower) - données en dur
# ---------------------------
def build_network_from_dicts():
    """
    Construire un réseau pandapower à partir des dictionnaires définis plus haut.
    Retour: net, bus_map (label -> idx), pv_map (bus_idx -> kw), ess_map
    """
    if pp is None:
        debug("[WARN] pandapower non installé, le PF ne sera pas exécuté.")
        # return placeholders
        return None, {}, {}, {}
    net = pp.create_empty_network()
    bus_map: Dict[str,int] = {}
    # create buses
    for b in buses:
        try:
            idx = pp.create_bus(net, vn_kv=float(b.get("vn_kv", 0.4)), name=str(b["bus"]))
            bus_map[str(b["bus"])] = idx
            if int(b.get("is_slack", 0)) == 1:
                pp.create_ext_grid(net, bus=idx, vm_pu=1.0, name="Slack")
        except Exception as e:
            debug(f"[WARN] erreur création bus {b}: {e}")

    # lines
    lines_created = 0
    for l in line:
        try:
            fb = bus_map.get(l["from_bus"]); tb = bus_map.get(l["to_bus"])
            if fb is None or tb is None:
                debug(f"[WARN] ligne {l['line_id']} -> bus introuvable (from={l['from_bus']} to={l['to_bus']}). Skip.")
                continue
            pp.create_line_from_parameters(net,
                                           from_bus=fb, to_bus=tb,
                                           length_km=float(l.get("length_m", 100))/1000.0,
                                           r_ohm_per_km=float(l.get("r_ohm_per_km", 0.65)),
                                           x_ohm_per_km=float(l.get("x_ohm_per_km", 0.412)),
                                           c_nf_per_km=0.0, max_i_ka=0.2,
                                           name=str(l.get("line_id")))
            lines_created += 1
        except Exception as e:
            debug(f"[WARN] impossible de créer la ligne {l.get('line_id')}: {e}")
    debug(f"Created {lines_created} lines.")

    # static PV
    pv_map: Dict[int, float] = {}
    pv_created = 0
    for p in pv_static:
        try:
            bus_label = str(p["bus"])
            if bus_label not in bus_map:
                debug(f"[WARN] PV {p['pv_id']} -> bus {bus_label} introuvable. Skip.")
                continue
            p_kw = safe_float(p.get("p_stc_kw", 0.0))
            pp.create_sgen(net, bus=bus_map[bus_label], p_mw=p_kw/1000.0, q_mvar=0.0, name=p["pv_id"])
            pv_map[bus_map[bus_label]] = pv_map.get(bus_map[bus_label], 0.0) + p_kw
            pv_created += 1
        except Exception as e:
            debug(f"[WARN] create_sgen failed: {e}")
    debug(f"Created {pv_created} PV modules.")

    # create loads from load_timeseries aggregated (mean per bus)
    load_df = pd.DataFrame(load_timeseries)
    if not load_df.empty:
        grouped = load_df.groupby("bus").agg({"p_kw":"mean","q_kvar":"mean"}).reset_index()
        created_loads = 0
        for _, r in grouped.iterrows():
            bus_label = str(r["bus"])
            if bus_label not in bus_map:
                debug(f"[WARN] Load bus {bus_label} not found -> skip")
                continue
            p_mw = float(r["p_kw"])/1000.0
            q_mvar = float(r["q_kvar"])/1000.0
            try:
                pp.create_load(net, bus=bus_map[bus_label], p_mw=p_mw, q_mvar=q_mvar, name=f"Load_{bus_label}")
                created_loads += 1
            except Exception as e:
                debug(f"[WARN] create_load failed: {e}")
        debug(f"Created {created_loads} loads from load_timeseries.")

    # storages (ESS)
    ess_map: Dict[int, Dict[str, float]] = {}
    ess_created = 0
    for s in ess:
        try:
            bus_label = str(s["bus"])
            if bus_label not in bus_map:
                debug(f"[WARN] ESS {s['ess_id']} -> bus {bus_label} introuvable. Skip.")
                continue
            cap_kwh = safe_float(s.get("e_cap_kwh", 20.0))
            pmax_kw = safe_float(s.get("p_max_kw_per_phase", 4.0))
            pp.create_storage(net, bus=bus_map[bus_label], p_mw=0.0,
                              max_e_mwh=cap_kwh/1000.0, sn_mva=pmax_kw/1000.0,
                              controllable=True, name=s["ess_id"])
            ess_map[bus_map[bus_label]] = {"cap_kwh":cap_kwh, "pmax_kw":pmax_kw, "id": s["ess_id"]}
            ess_created += 1
        except Exception as e:
            debug(f"[WARN] create_storage failed: {e}")
    debug(f"Created {ess_created} ESS entries (pandapower storages).")

    return net, bus_map, pv_map, ess_map

# ---------------------------
# Powerflow runner & helpers
# ---------------------------
def run_powerflow_for_timestep(net, bus_map, pv_scale=1.0, timestep:str=None, verbose=False):
    """
    Met à jour sgens et loads si possible pour le timestep (si time series fournie),
    execute runpp (robuste), et retourne résultats agrégés par bus:
      DataFrame with columns: ['t','bus_label','bus_idx','V_pu','P_loss_kW_total_line_share']
    """
    if pp is None:
        # produce synthetic outputs (fallback) - flat voltages 1.0, zero losses
        rows=[]
        for label, idx in bus_map.items():
            rows.append({"t":timestep, "Départ": label, "bus_idx": idx, "V_pu": 1.0, "P_loss_kW": 0.0})
        return pd.DataFrame(rows)

    # Try to apply pv_timeseries and load_timeseries for given timestep (if matches)
    # Update sgen: scale static PV by pv_scale and override if pv_timeseries exists
    try:
        # scale static sgens uniformly by pv_scale: multiply existing p_mw in net.sgen by scale
        if "sgen" in net and not net.sgen.empty:
            for idx in net.sgen.index:
                original = float(net.sgen.at[idx,"p_mw"])
                net.sgen.at[idx,"p_mw"] = original * pv_scale
        # override from pv_timeseries if present for specific pv_id
        pv_ts_df = pd.DataFrame(pv_timeseries)
        if timestep is not None and not pv_ts_df.empty:
            # find entries matching timestep
            rows = pv_ts_df[pv_ts_df["time"]==timestep]
            # for each entry, find sgen with same name/pv_id and set p_mw
            for _, r in rows.iterrows():
                pv_id = str(r["pv_id"])
                p_kw = safe_float(r["p_kw"], 0.0)
                # find sgen row in net.sgen with name pv_id
                if "sgen" in net and not net.sgen.empty:
                    matches = net.sgen[net.sgen["name"]==pv_id]
                    for i in matches.index:
                        net.sgen.at[i,"p_mw"] = p_kw/1000.0
    except Exception as e:
        debug(f"[WARN] erreur mise à jour PV pour timestep {timestep}: {e}")

    # Update loads for timestep if available
    try:
        load_ts_df = pd.DataFrame(load_timeseries)
        if timestep is not None and not load_ts_df.empty:
            rows = load_ts_df[load_ts_df["time"]==timestep]
            # we approximate by setting aggregate load at bus to sum p_kw for that time
            if not rows.empty:
                # set all loads at bus to new value (note: pandapower may have multiple loads)
                for _, r in rows.iterrows():
                    bus_label = str(r["bus"])
                    p_kw = safe_float(r["p_kw"], 0.0)
                    # find loads for that bus and update p_mw
                    if "load" in net and not net.load.empty:
                        bus_idx = None
                        if bus_label in net.bus["name"].values:
                            bus_idx = net.bus[net.bus["name"]==bus_label].index[0]
                        else:
                            # try our bus_map mapping
                            bus_idx = bus_map.get(bus_label)
                        if bus_idx is not None:
                            idxs = net.load[net.load["bus"]==bus_idx].index
                            for i in idxs:
                                net.load.at[i,"p_mw"] = p_kw/1000.0
    except Exception as e:
        debug(f"[WARN] erreur mise à jour charges pour timestep {timestep}: {e}")

    # run powerflow with fallbacks
    success = False
    try:
        pp.runpp(net)
        success = True
    except Exception as e:
        debug(f"[WARN] runpp standard failed: {e}")
    if not success:
        try:
            pp.runpp(net, algorithm="nr")
            success = True
        except Exception as e:
            debug(f"[WARN] runpp nr failed: {e}")
    if not success:
        try:
            pp.runpp(net, enforce_q_lims=True)
            success = True
        except Exception as e:
            debug(f"[WARN] runpp enforce_q_lims failed: {e}")
    if not success:
        debug(f"[ERROR] Powerflow impossible pour timestep {timestep} (on continuera).")

    # collect results
    rows = []
    # compute line losses total pl_mw per line -> distribute half to each end (approx)
    total_line_losses_kw = 0.0
    if hasattr(net, "res_line") and not net.res_line.empty:
        for idx in net.res_line.index:
            pl_mw = safe_float(net.res_line.at[idx, "pl_mw"], 0.0)
            total_line_losses_kw += pl_mw * 1000.0

    # get bus voltage magnitudes
    for label, idx in bus_map.items():
        V_pu = 1.0
        if hasattr(net, "res_bus") and "vm_pu" in net.res_bus.columns:
            if idx in net.res_bus.index:
                V_pu = float(net.res_bus.at[idx, "vm_pu"])
        # approximate loss share per bus (equal share of total_line_losses_kw / nb_buses)
        P_loss_share = float(total_line_losses_kw) / max(1, len(bus_map))
        rows.append({"t":timestep, "Départ": label, "bus_idx": idx, "V_pu": V_pu, "P_loss_kW": P_loss_share})
    return pd.DataFrame(rows)

# ---------------------------
# Run scenarios (PV scales + ESS)
# ---------------------------
def run_all_scenarios(pv_scales = [0.0, 0.2, 0.5], include_ess=False):
    """
    Pour chaque scenario PV scale, pour chaque timestep, exécute PF et collecte résultats.
    Retour: DataFrame results_all columns: ['Scenario','t','Départ','bus_idx','V_pu','P_loss_kW']
    """
    net_base, bus_map, pv_map, ess_map = build_network_from_dicts()
    results_list = []
    # if pandapower missing, build_network returns None net; our run function will handle fallback
    for scale in pv_scales:
        scen_name = f"PV_{int(scale*100)}%"
        debug(f"Running scenario {scen_name} ...")
        # deep copy network safely (use python copy)
        if net_base is None:
            net_for_scenario = None
        else:
            net_for_scenario = copy.deepcopy(net_base)
        for t in SimulationSettings["timesteps"]:
            try:
                df = run_powerflow_for_timestep(net_for_scenario, bus_map, pv_scale=scale, timestep=t)
                df["Scenario"] = scen_name
                df["t"] = t
                results_list.append(df)
            except Exception as e:
                debug(f"[ERROR] scenario {scen_name} timestep {t} failed: {e}")
                traceback.print_exc()
    # if include_ess -> we will append PV_x%_ESS scenarios (post-hoc; ESS effect applied later)
    results_all = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame(columns=["Scenario","t","Départ","bus_idx","V_pu","P_loss_kW"])
    if include_ess:
        # we'll copy PV_50% and mark ESS present (effect applied later after optimization)
        ess_scen = []
        for t in SimulationSettings["timesteps"]:
            subset = results_all[(results_all["Scenario"]==f"PV_{int(50)}%") & (results_all["t"]==t)].copy()
            if not subset.empty:
                subset["Scenario"] = f"PV_{int(50)}%_ESS"
                ess_scen.append(subset)
        if ess_scen:
            results_all = pd.concat([results_all] + ess_scen, ignore_index=True)
    return results_all, bus_map, pv_map, ess_map

# ---------------------------
# Summarize & metrics
# ---------------------------
def summarize_by_scenario(results_all: pd.DataFrame):
    # compute per scenario summary: avg V, total losses, avg VUF (not provided -> 0)
    rows=[]
    for scen in results_all["Scenario"].unique():
        df = results_all[results_all["Scenario"]==scen]
        rows.append({
            "Scenario": scen,
            "V_mean": float(df["V_pu"].mean()) if not df.empty else np.nan,
            "P_loss_kW_total": float(df["P_loss_kW"].sum()) if not df.empty else 0.0,
            "VUF_mean_percent": 0.0
        })
    return pd.DataFrame(rows)

def compute_gains(summary_df: pd.DataFrame):
    # baseline = PV_0%
    base = summary_df[summary_df["Scenario"]=="PV_0%"]
    if base.empty:
        baseline_V = summary_df["V_mean"].mean()
        baseline_loss = summary_df["P_loss_kW_total"].mean()
    else:
        baseline_V = base["V_mean"].iloc[0]
        baseline_loss = base["P_loss_kW_total"].iloc[0]
    rows=[]
    for _, r in summary_df.iterrows():
        gain_loss = 100.0 * (baseline_loss - r["P_loss_kW_total"]) / max(abs(baseline_loss), 1e-9)
        gain_v = 100.0 * (r["V_mean"] - baseline_V) / max(abs(baseline_V), 1e-9) if baseline_V!=0 else 0.0
        rows.append({"Scenario": r["Scenario"], "ΔPertes_%": gain_loss, "ΔTension_%": gain_v, "V_mean": r["V_mean"], "P_loss_kW_total": r["P_loss_kW_total"]})
    return pd.DataFrame(rows)

# ---------------------------
# ESS optimisation (Pyomo multi-periode proxy)
# ---------------------------
def optimize_ess_pyomo(results_all: pd.DataFrame, bus_map: Dict[str,int], ess_map:Dict[int,Dict[str,float]]):
    """
    Optimisation multi-periode: variables P_dis, P_ch et SOC par bus/t.
    Proxy linking: V_after = V_before + k_V_per_kW * Pess (Pess in kW, positive = discharge)
    Losses after eaten by ESS approximated by loss_after = loss_before - k_loss_per_kW * Pess_total
    Objective = weighted sum from SimulationSettings["weights"] combining voltage deviation and losses.
    """
    if pyo is None:
        raise ImportError(f"Pyomo non installé: {_pyo_err}")
    # Check solver GLPK
    solver = pyo.SolverFactory("glpk")
    if not solver.available():
        raise RuntimeError("GLPK solver non disponible. Installe GLPK pour utiliser l'optimiseur Pyomo.")
    # prepare data: baseline V and losses per bus per timestep from results_all (PV_50%)
    target_scen = "PV_50%"
    df_target = results_all[results_all["Scenario"]==target_scen]
    if df_target.empty:
        raise ValueError(f"Aucun résultat pour le scénario {target_scen} — impossible d'optimiser.")
    timesteps = sorted(df_target["t"].unique())
    buses_labels = sorted(bus_map.keys())
    # map bus label -> baseline V mean per timestep
    V0 = {(b,t): 1.0 for b in buses_labels for t in timesteps}
    Loss0 = {(b,t): 0.0 for b in buses_labels for t in timesteps}
    for _, r in df_target.iterrows():
        b = r["Départ"]; t = r["t"]
        V0[(b,t)] = float(r["V_pu"])
        Loss0[(b,t)] = float(r["P_loss_kW"])
    # build model
    model = pyo.ConcreteModel()
    model.B = pyo.Set(initialize=buses_labels, ordered=True)
    model.T = pyo.Set(initialize=timesteps, ordered=True)
    # Pdis (kW), Pchg (kW), SOC (kWh)
    def pdis_index(m):
        return ((b,t) for b in model.B for t in model.T)
    model.Pdis = pyo.Var(model.B, model.T, domain=pyo.Reals, bounds=lambda m,b,t: (-ess_map.get(bus_map.get(b, -999), {}).get("pmax_kw", 0)*1.0, ess_map.get(bus_map.get(b, -999), {}).get("pmax_kw", 0)*1.0))  # positive discharge
    model.SOC = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals, bounds=(0, max((v.get("cap_kwh",20) for v in ess_map.values()), default=20)))
    # Some ESS may not exist at certain buses; enforce zero bounds as needed
    # initial SOC as half capacity or as provided
    # SOC dynamics
    EFF_CH = 0.94; EFF_DIS = 0.94
    cap_by_bus = {b: ess_map.get(bus_map[b], {}).get("cap_kwh", 0.0) for b in buses_labels}
    pmax_by_bus = {b: ess_map.get(bus_map[b], {}).get("pmax_kw", 0.0) for b in buses_labels}

    # override Pdis bounds for buses without ESS (force 0)
    for b in buses_labels:
        if pmax_by_bus[b] <= 0:
            # create constraint that Pdis[b,t] == 0
            pass

    # constraints SOC dynamics
    def soc_rule(m, b, t):
        # if no ESS: SOC fixed 0
        cap = cap_by_bus[b]
        if cap <= 0:
            return m.SOC[b,t] == 0.0
        t_list = list(model.T)
        if t == t_list[0]:
            # init to soc_init ~ 50% of cap
            return m.SOC[b,t] == cap * 0.5 + 0.0
        else:
            prev = t_list[t_list.index(t)-1]
            # SOC_t = SOC_{t-1} - Pdis(t)/EFF_DIS * dt + Pch*EFF_CH*dt
            # But we only model Pdis (net positive = discharge). For simplicity, treat Pdis positive=>discharge reduces SOC.
            return m.SOC[b,t] == m.SOC[b,prev] - (m.Pdis[b,t] / EFF_DIS) * 1.0  # dt=1h, Pdis in kW, SOC in kWh
    model.soc_cons = pyo.Constraint(model.B, model.T, rule=soc_rule)

    # bound Pdis to pmax
    def pdis_bounds_rule(m, b, t):
        return (-pmax_by_bus[b], pmax_by_bus[b])
    # Note: pyomo Var bounds via domain already used; create explicit constraint if needed
    # Objective: weights on voltage deviation and losses
    kV = SimulationSettings["k_V_per_kW"]
    k_loss = SimulationSettings["k_loss_per_kW"]
    w = SimulationSettings["weights"]

    def obj_rule(m):
        expr = 0.0
        for b in buses_labels:
            for t in timesteps:
                V_before = V0[(b,t)]
                Pess = m.Pdis[b,t]  # positive discharge increases local V
                V_after = V_before + kV * Pess
                loss_before = Loss0[(b,t)]
                loss_after = pyo.maximize(loss_before - k_loss * Pess, 0) if False else (loss_before - k_loss * Pess)
                # penalize voltage deviation from 1.0 (squared) and losses
                expr += w["w1"] * (V_after - 1.0)**2 + w["w2"] * (loss_after) + w["w3"] * (Pess**2)
        return expr
    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # solve
    debug("Lancement solveur GLPK (Pyomo) pour optimisation ESS (proxy)...")
    res = solver.solve(model, tee=False)
    debug(f"Pyomo solve status: {res.solver.status}, termination: {res.solver.termination_condition}")

    # extract schedule
    rows=[]
    for b in buses_labels:
        for t in timesteps:
            try:
                p_ess = float(pyo.value(model.Pdis[b,t]))
                soc = float(pyo.value(model.SOC[b,t]))
            except Exception:
                p_ess = 0.0; soc = 0.0
            rows.append({"Départ": b, "t": t, "P_ESS_kW_opt": p_ess, "SOC_kWh_opt": soc, "SOC_%_opt": (soc / max(1e-6, cap_by_bus[b]))*100.0 if cap_by_bus[b]>0 else 0.0})
    df_opt = pd.DataFrame(rows)
    csv_out = os.path.join(OUTPUT_DIR, "summary_ess_pyomo.csv")
    df_opt.to_csv(csv_out, index=False)
    debug(f"Pyomo optimisation results saved -> {csv_out}")
    return df_opt

# ---------------------------
# ESS heuristic fallback
# ---------------------------
def heuristic_ess_schedule(results_all: pd.DataFrame, bus_map:Dict[str,int], ess_map:Dict[int,Dict[str,float]]):
    """
    Simple heuristic: charge when V high, discharge when V low.
    Return DataFrame similar to optimize_ess_pyomo output.
    """
    df_base = results_all[results_all["Scenario"]=="PV_50%"].copy()
    if df_base.empty:
        df_base = results_all.groupby(["t","Départ"]).mean().reset_index()
    rows=[]
    for (t, group) in df_base.groupby("t"):
        for _, r in group.iterrows():
            b = r["Départ"]
            V = float(r["V_pu"])
            cap = ess_map.get(bus_map.get(b,-1),{}).get("cap_kwh", 0.0)
            pmax = ess_map.get(bus_map.get(b,-1),{}).get("pmax_kw", 0.0)
            if cap <= 0 or pmax <=0:
                rows.append({"Départ": b, "t": t, "P_ESS_kW_heur": 0.0, "SOC_kWh_heur": 0.0, "SOC_%_heur": 0.0})
                continue
            # if V < 0.98 -> discharge, if V >1.02 -> charge (negative power)
            if V < 0.98:
                p = min(pmax, cap*0.2)   # discharge modest amount
                soc = max(0.0, cap*0.5 - p)
            elif V > 1.02:
                p = -min(pmax, cap*0.2)  # charge negative
                soc = min(cap, cap*0.5 + abs(p))
            else:
                p = 0.0; soc = cap*0.5
            rows.append({"Départ": b, "t": t, "P_ESS_kW_heur": p, "SOC_kWh_heur": soc, "SOC_%_heur": (soc/cap)*100.0 if cap>0 else 0.0})
    dfh = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "summary_ess_heuristic.csv")
    dfh.to_csv(out, index=False)
    debug(f"Heuristic ESS schedule saved -> {out}")
    return dfh

# ---------------------------
# Apply ESS proxy (voltage correction & loss correction)
# ---------------------------
def apply_ess_effects(results_all: pd.DataFrame, df_ess_schedule: pd.DataFrame, mode:str="opt"):
    """
    Adds columns: P_ESS_kW, SOC_kWh, V_pu_after, P_loss_kW_after
    mode: 'opt' expects columns P_ESS_kW_opt, 'heur' expects P_ESS_kW_heur
    """
    kV = SimulationSettings["k_V_per_kW"]
    k_loss = SimulationSettings["k_loss_per_kW"]
    df = results_all.copy()
    # merge schedules
    keycols = ["Départ","t"]
    if mode=="opt":
        schedule = df_ess_schedule.rename(columns={"P_ESS_kW_opt":"P_ESS_kW","SOC_kWh_opt":"SOC_kWh","SOC_%_opt":"SOC_%"})
    else:
        schedule = df_ess_schedule.rename(columns={"P_ESS_kW_heur":"P_ESS_kW","SOC_kWh_heur":"SOC_kWh","SOC_%_heur":"SOC_%"})
    merged = pd.merge(df, schedule[keycols+["P_ESS_kW","SOC_kWh","SOC_%"]], on=keycols, how="left")
    merged["P_ESS_kW"] = merged["P_ESS_kW"].fillna(0.0)
    merged["SOC_kWh"] = merged["SOC_kWh"].fillna(0.0)
    merged["SOC_%"] = merged["SOC_%"].fillna(0.0)
    merged["V_pu_after"] = merged["V_pu"] + kV * merged["P_ESS_kW"]
    merged["P_loss_kW_after"] = merged["P_loss_kW"] - k_loss * merged["P_ESS_kW"]
    merged["P_loss_kW_after"] = merged["P_loss_kW_after"].clip(lower=0.0)
    return merged

# ---------------------------
# Plotting functions (14 figures)
# ---------------------------
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    try:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        debug(f"Saved figure -> {path}")
    except Exception as e:
        debug(f"[WARN] cannot save figure {name}: {e}")

def plot_voltage_profile_by_scenario(results_df):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=results_df, x="Départ", y="V_pu", hue="Scenario", marker="o", ax=ax)
    ax.axhline(1.05, color="red", linestyle="--", label="Vmax")
    ax.axhline(0.95, color="red", linestyle="--", label="Vmin")
    ax.set_title("Profil de tension par scénario")
    ax.set_ylabel("Tension (pu)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "voltage_profile_by_scenario")
    plt.show()

def plot_losses_by_scenario(results_df):
    # aggregate per scenario & departure
    agg = results_df.groupby(["Scenario","Départ"]).P_loss_kW.mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=agg, x="Départ", y="P_loss_kW", hue="Scenario", ax=ax)
    ax.set_title("Pertes actives par scénario")
    ax.set_ylabel("P pertes (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "losses_by_scenario")
    plt.show()

def plot_voltage_heatmap(results_df):
    try:
        pivot = results_df.groupby(["Scenario","Départ"]).V_pu.mean().unstack(level=0).T
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", ax=ax, cbar_kws={"label":"V_pu"})
        ax.set_title("Heatmap de la tension par scénario et départ")
        plt.tight_layout()
        save_fig(fig, "voltage_heatmap_by_scenario_depart")
        plt.show()
    except Exception as e:
        debug(f"[WARN] plot_voltage_heatmap failed: {e}")

def plot_pv_capacity_by_depart(pv_map, bus_map):
    items = []
    for bus_idx, kw in pv_map.items():
        # find label
        label = next((lbl for lbl, idx in bus_map.items() if idx==bus_idx), str(bus_idx))
        items.append({"Départ": label, "PV_kW": kw})
    if not items:
        debug("[INFO] no PV capacity to plot.")
        return
    dfpv = pd.DataFrame(items)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=dfpv, x="Départ", y="PV_kW", ax=ax)
    ax.set_title("Capacité d'accueil PV par départ (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "pv_capacity_by_depart")
    plt.show()

def plot_comparative_gains(gains_df):
    if gains_df.empty:
        debug("[INFO] no gains to plot.")
        return
    melted = gains_df.melt(id_vars="Scenario", value_vars=["ΔPertes_%","ΔTension_%"], var_name="Metric", value_name="Gain_%")
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(data=melted, x="Metric", y="Gain_%", hue="Scenario", ax=ax)
    ax.set_title("Comparaison des gains par scénario")
    plt.tight_layout()
    save_fig(fig, "comparative_gains")
    plt.show()

def plot_gains_heatmap(gains_df):
    if gains_df.empty: return
    pivot = gains_df.set_index("Scenario")[["ΔPertes_%","ΔTension_%"]]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Heatmap des gains par scénario et métrique")
    plt.tight_layout()
    save_fig(fig, "gains_heatmap")
    plt.show()

def plot_soc_and_pess(df_ess, mode="opt"):
    col_p = "P_ESS_kW_opt" if "P_ESS_kW_opt" in df_ess.columns else ("P_ESS_kW_heur" if "P_ESS_kW_heur" in df_ess.columns else "P_ESS_kW")
    col_s = "SOC_%_opt" if "SOC_%_opt" in df_ess.columns else ("SOC_%_heur" if "SOC_%_heur" in df_ess.columns else "SOC_%")
    if col_s not in df_ess.columns or col_p not in df_ess.columns:
        debug("[INFO] pas de schedule ESS pour plot SOC/PESS")
        return
    # SOC line
    fig, ax = plt.subplots(figsize=(9,4))
    sns.lineplot(data=df_ess, x="t", y=col_s, hue="Départ", marker="o", ax=ax)
    ax.set_title("SOC ESS optimisé (%)" if mode=="opt" else "SOC ESS (heuristique)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "soc_ess_optimized" if mode=="opt" else "soc_ess_heuristic")
    plt.show()
    # Pess bar
    fig, ax = plt.subplots(figsize=(9,4))
    sns.barplot(data=df_ess, x="t", y=col_p, hue="Départ", ax=ax)
    ax.set_title("Puissance ESS (kW) — positive = décharge")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, "pess_ess_optimized" if mode=="opt" else "pess_ess_heuristic")
    plt.show()

def plot_before_after_losses_and_voltage(results_df_with_ess):
    # grouped losses before/after
    try:
        df = results_df_with_ess.copy()
        agg = df.groupby(["Scenario","Départ"]).agg({"P_loss_kW":"mean","P_loss_kW_after":"mean","V_pu":"mean","V_pu_after":"mean"}).reset_index()
        # losses grouped bar
        fig, ax = plt.subplots(figsize=(10,5))
        x = np.arange(len(agg["Départ"].unique()))
        sns.barplot(data=agg.melt(id_vars=["Scenario","Départ"], value_vars=["P_loss_kW","P_loss_kW_after"]), x="Départ", y="value", hue="variable", ax=ax)
        ax.set_title("Pertes avant / après ESS (moyennes)")
        ax.set_ylabel("P (kW)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_fig(fig, "loss_before_after_ess")
        plt.show()
        # voltage before/after line
        fig, ax = plt.subplots(figsize=(10,5))
        for dep in agg["Départ"].unique():
            sub = agg[agg["Départ"]==dep]
            ax.plot(["before","after"], [sub["V_pu"].iloc[0], sub["V_pu_after"].iloc[0]], marker="o", label=str(dep))
        ax.set_title("Comparaison des tensions avant/après ESS")
        ax.set_ylabel("V (pu)")
        ax.legend(bbox_to_anchor=(1.02,1.0))
        plt.tight_layout()
        save_fig(fig, "voltage_before_after_ess")
        plt.show()
    except Exception as e:
        debug(f"[WARN] plot_before_after_losses_and_voltage failed: {e}")

def plot_pv_capacity_phase_heatmap(pv_static_list, bus_map):
    # build matrix departures x phase
    rows=[]
    for p in pv_static_list:
        label = p["bus"]
        rows.append({"Départ":label, "phase":p.get("phase","A"), "PV_kW": safe_float(p.get("p_stc_kw",0.0))})
    if not rows:
        debug("[INFO] no PV per phase to plot")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="phase", columns="Départ", values="PV_kW", aggfunc="sum").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Capacité PV par départ et par phase (kW)")
    plt.tight_layout()
    save_fig(fig, "pv_capacity_by_depart_phase_heatmap")
    plt.show()

def plot_radar_metrics(summary_df):
    # metrics: V_mean, P_loss_kW_total, VUF (set to 0)
    try:
        df = summary_df.copy()
        metrics = ["V_mean","P_loss_kW_total"]
        labels = metrics
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        for _, r in df.iterrows():
            vals = [r[m] for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, label=r["Scenario"])
            ax.fill(angles, vals, alpha=0.15)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Radar: métriques par scénario")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        save_fig(fig, "radar_metrics_by_scenario")
        plt.show()
    except Exception as e:
        debug(f"[WARN] plot_radar_metrics failed: {e}")

def plot_surface_3d(results_df):
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        scenarios = list(sorted(results_df["Scenario"].unique()))
        departures = list(sorted(results_df["Départ"].unique()))
        Z = np.zeros((len(scenarios), len(departures)))
        for i, scen in enumerate(scenarios):
            sub = results_df[results_df["Scenario"]==scen].groupby("Départ").V_pu.mean()
            for j, dep in enumerate(departures):
                Z[i,j] = sub.get(dep, np.nan)
        X, Y = np.meshgrid(range(len(departures)), range(len(scenarios)))
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_xticks(range(len(departures))); ax.set_xticklabels(departures, rotation=45)
        ax.set_yticks(range(len(scenarios))); ax.set_yticklabels(scenarios)
        ax.set_xlabel("Départ"); ax.set_ylabel("Scenario"); ax.set_zlabel("V_pu")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        save_fig(fig, "surface_3D_tension")
        plt.show()
    except Exception as e:
        debug(f"[WARN] plot_surface_3d failed: {e}")

def plot_time_series_2d(results_all):
    # if load_timeseries / pv_timeseries are provided, plot simple time series aggregated
    try:
        # plot total PV production for PV_20% and PV_50% scenarios over timesteps (approx)
        df = results_all.copy()
        df_time = df.groupby(["t","Scenario"]).V_pu.mean().reset_index()
        fig, ax = plt.subplots(figsize=(10,4))
        sns.lineplot(data=df_time, x="t", y="V_pu", hue="Scenario", marker="o", ax=ax)
        ax.set_title("Time series: tension moyenne par scénario")
        plt.xticks(rotation=45)
        save_fig(fig, "lineplots_time_series_2D")
        plt.show()
    except Exception as e:
        debug(f"[WARN] plot_time_series_2d failed: {e}")

# ---------------------------
# Orchestration main
# ---------------------------
def main():
    debug("=== Début exécution ===")
    # 1) run scenarios PV_0%, PV_20%, PV_50% and PV_50%_ESS placeholder
    results_all, bus_map, pv_map, ess_map = run_all_scenarios(pv_scales=[0.0, 0.2, 0.5], include_ess=True)
    if results_all.empty:
        debug("[ERROR] Aucun résultat de PF — vérifie l'installation de pandapower ou les données.")
    # 2) summaries
    summary = summarize_by_scenario(results_all)
    gains = compute_gains(summary)
    # 3) try optimize ESS via Pyomo (proxy) otherwise heuristic
    df_ess_schedule = None
    try:
        if pyo is not None:
            df_ess_schedule = optimize_ess_pyomo(results_all, bus_map, ess_map)
            ess_mode = "opt"
        else:
            raise ImportError("Pyomo non installé")
    except Exception as e:
        debug(f"[WARN] optimisation Pyomo échouée ({e}) -> fallback heuristique.")
        try:
            df_ess_schedule = heuristic_ess_schedule(results_all, bus_map, ess_map)
            ess_mode = "heur"
        except Exception as e2:
            debug(f"[ERROR] heuristique aussi échouée: {e2}")
            df_ess_schedule = pd.DataFrame(columns=["Départ","t","P_ESS_kW_heur","SOC_kWh_heur","SOC_%_heur"])
            ess_mode = "none"

    # 4) apply ESS effects to PV_50%_ESS scenarios only (post-hoc)
    results_with_ess = apply_ess_effects(results_all, df_ess_schedule, mode=("opt" if ess_mode=="opt" else "heur"))
    # 5) Export main CSVs
    try:
        results_all.to_csv(os.path.join(OUTPUT_DIR,"network_results.csv"), index=False)
        results_all.to_json(os.path.join(OUTPUT_DIR,"network_results.json"), orient="records")
        results_with_ess.to_csv(os.path.join(OUTPUT_DIR,"network_results_with_ess.csv"), index=False)
        summary.to_csv(os.path.join(OUTPUT_DIR,"summary_by_scenario.csv"), index=False)
        gains.to_csv(os.path.join(OUTPUT_DIR,"gains_summary.csv"), index=False)
        debug("Exports CSV/JSON écrits dans outputs/")
    except Exception as e:
        debug(f"[WARN] Erreur export CSV/JSON: {e}")

    # 6) Generate all plots
    try:
        plot_voltage_profile_by_scenario(results_all)
        plot_losses_by_scenario(results_all)
        plot_voltage_heatmap(results_all)
        plot_pv_capacity_by_depart(pv_map, bus_map)
        plot_comparative_gains(gains)
        plot_gains_heatmap(gains)
        # ESS plots
        if not df_ess_schedule.empty:
            plot_soc_and_pess(df_ess_schedule, mode=("opt" if ess_mode=="opt" else "heur"))
        plot_before_after_losses_and_voltage(results_with_ess)
        plot_pv_capacity_phase_heatmap(pv_static, bus_map)
        plot_radar_metrics(summary)
        plot_surface_3d(results_all)
        plot_time_series_2d(results_all)
    except Exception as e:
        debug(f"[WARN] Erreur lors génération figures: {e}")
        traceback.print_exc()

    debug("=== Fin exécution. Vérifie le dossier outputs/ pour les PNG et CSV. ===")

if __name__ == "__main__":
    main()

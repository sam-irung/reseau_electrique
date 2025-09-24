#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse réseau unifiée (Module1 pandapower + Module2 ESS (Pyomo ou heuristique))
- Lecture multi-feuilles depuis Donnees.xlsx
- Construction réseau (buses, transfo, lignes, charges, PV, ESS)
- Scénarios : base (0%), PV, PV+ESS
- Option optimisation ESS Pyomo (si GLPK dispo) ou heuristique
- Visualisations (seaborn/matplotlib) : line, comparative, bar, heatmap, radar, 3D surface, 2D
- Sauvegarde PNG dans outputs/
- Exports JSON/CSV
"""
from __future__ import annotations
import os, sys, math, json, traceback
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional heavy imports (load lazily / with clear messages)
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

sns.set(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120

# -----------------------
# Configuration
# -----------------------
INPUT_XLSX = "data.xlsx"
INPUT_CSV_FALLBACK = "data.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PV_RATIO_DEFAULT = 0.20   # 20% injection default
ESS_CAP_kWh_DEFAULT = 20.0
ESS_Pmax_kW_DEFAULT = 4.0
SAVE_FIGURES = True
VERBOSE = True

# -----------------------
# Utilities
# -----------------------
def debug(msg: str):
    if VERBOSE:
        print(msg)

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip()
        s = s.replace(";", ".").replace(" ", "")
        # handle European thousands
        if s.count('.') > 1 and ',' in s:
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '.')
        return float(s)
    except Exception:
        return default

def find_input():
    if os.path.exists(INPUT_XLSX):
        return INPUT_XLSX
    if os.path.exists(INPUT_CSV_FALLBACK):
        return INPUT_CSV_FALLBACK
    raise FileNotFoundError(f"Data file not found: {INPUT_XLSX} or {INPUT_CSV_FALLBACK}")

def save_fig(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        debug(f"Saved figure -> {path}")
    except Exception as e:
        debug(f"[WARN] Could not save figure {name}: {e}")

# -----------------------
# Data loading
# -----------------------
def load_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    debug(f"Reading input file: {path}")
    if path.lower().endswith(".csv"):
        # fallback: single-sheet csv -> put in 'data_phase'
        df = pd.read_csv(path, dtype=str)
        df = df.applymap(lambda v: safe_float(v, np.nan))
        return {"data_phase": df}
    # excel
    xls = pd.ExcelFile(path)
    sheets = {}
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
            # normalize whitespace column names
            df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
            # convert numeric-like strings to floats where possible
            df = df.applymap(lambda v: safe_float(v, np.nan) if isinstance(v, str) else v)
            sheets[sheet.strip().lower()] = df
            debug(f" - loaded sheet '{sheet}' shape={df.shape}")
        except Exception as e:
            debug(f"[WARN] Could not read sheet {sheet}: {e}")
    return sheets

# -----------------------
# Build pandapower network
# -----------------------
def check_pandapower():
    if pp is None:
        raise ImportError(f"pandapower not available: {_pp_err}")

def check_pandapower():
    if pp is None:
        raise ImportError(f"pandapower not available: {_pp_err}")

def create_network_from_sheets(sheets: Dict[str, pd.DataFrame]):
    """
    Build pandapower network using provided sheets. Expect sheets:
    data_phase, transformer, buses, line, limits, pv_static, ess, load_timeseries, pv_timeseries
    """
    check_pandapower()
    net = pp.create_empty_network()

    # Buses sheet
    buses_df = sheets.get("buses")
    bus_map = {}



    if buses_df is None:
        debug("[WARN] 'buses' sheet not found -> building buses from unique 'Départ' in data_phase.")
        data_phase = sheets.get("data_phase")
        if data_phase is None:
            raise ValueError("No data_phase and no buses sheet.")
        departures = sorted(list(data_phase["Départ"].dropna().unique()))
        for i, dep in enumerate(departures):
            bus_map[dep] = pp.create_bus(net, vn_kv=0.4, name=f"Bus_{dep}")
        debug(f"Created {len(bus_map)} buses from data_phase.")
    else:
        # Normaliser colonnes
        buses_df.columns = [c.strip().lower() for c in buses_df.columns]

        # Conversion des nombres avec virgules en points
        for col in ["vn_kv", "is_slack"]:
            if col in buses_df.columns:
                buses_df[col] = buses_df[col].astype(str).str.replace(",", ".").astype(float)

        for _, r in buses_df.iterrows():
            try:
                label = r.get("bus")
                if pd.isna(label):
                    continue
                vn_kv = r.get("vn_kv", 0.4)
                bus_idx = pp.create_bus(net, vn_kv=float(vn_kv), name=str(label))
                bus_map[label] = bus_idx

                if int(r.get("is_slack", 0)) == 1:
                    pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="Slack")

            except Exception as e:
                print(f"[WARN] Ligne bus ignorée: {r.to_dict()} | Erreur: {e}")

        if len(bus_map) == 0:
            print("[WARN] Aucun bus valide trouvé → génération automatique depuis data_phase.")
            data_phase = sheets.get("data_phase")
            if data_phase is not None:
                for i, dep in enumerate(sorted(list(data_phase["Départ"].dropna().unique()))):
                    bus_map[dep] = pp.create_bus(net, vn_kv=0.4, name=f"Bus_{dep}")
                pp.create_ext_grid(net, bus=list(bus_map.values())[0], vm_pu=1.0, name="Slack")
            else:
                raise ValueError("Impossible de créer des bus: aucune donnée disponible.")
        debug(f"Created {len(bus_map)} buses from 'buses' sheet.")


    # Transformer
    trafo_df = sheets.get("transformer")
    if trafo_df is not None and not trafo_df.empty:
        row = trafo_df.iloc[0]
        sn_kva = safe_float(row.get("sn_kva", row.get("sn_kva", None)), None)
        vn_hv_kv = safe_float(row.get("vn_hv_kv", 15.0), 15.0)
        vn_lv_kv = safe_float(row.get("vn_lv_kv", 0.4), 0.4)
        vk_percent = safe_float(row.get("vk_percent", 6.0), 6.0)
        vkr_percent = safe_float(row.get("vkr_percent", 0.5), 0.5)
        # create high-voltage and low-voltage bus if needed
        # For simplicity, map HV to external grid bus and LV to bus of first entry
        # create an ext_grid at the LV bus of first bus in bus_map
        try:
            # create external grid later after buses exist
            debug("Transformer data present; will reference in net as ext_grid on first LV bus.")
        except Exception:
            pass

    # create ext_grid on first bus
    if len(bus_map) == 0:
        raise ValueError("No buses available to create external grid.")
    first_bus = list(bus_map.values())[0]
    try:
        pp.create_ext_grid(net, bus=first_bus, vm_pu=1.0, name="Slack")
    except Exception as e:
        debug(f"[WARN] create_ext_grid failed: {e}")

    # Lines sheet
    line_df = sheets.get("line")
    if line_df is not None:
        for _, r in line_df.iterrows():
            try:
                from_bus_label = r.get("from_bus")
                to_bus_label = r.get("to_bus")
                # resolve if bus label provided like 'B1'
                if from_bus_label not in bus_map:
                    # try string match
                    keys = [k for k in bus_map.keys() if str(k) == str(from_bus_label)]
                    if keys:
                        from_idx = bus_map[keys[0]]
                    else:
                        debug(f"[WARN] line from_bus {from_bus_label} not found in bus_map; skipping")
                        continue
                else:
                    from_idx = bus_map[from_bus_label]
                if to_bus_label not in bus_map:
                    keys = [k for k in bus_map.keys() if str(k) == str(to_bus_label)]
                    if keys:
                        to_idx = bus_map[keys[0]]
                    else:
                        debug(f"[WARN] line to_bus {to_bus_label} not found in bus_map; skipping")
                        continue
                else:
                    to_idx = bus_map[to_bus_label]

                length_m = safe_float(r.get("length_m", 50.0), 50.0)
                length_km = max(0.001, length_m / 1000.0)
                r_ohm_per_km = safe_float(r.get("r_ohm_per_km", r.get("r_ohm_per_km", 0.65)), 0.65)
                x_ohm_per_km = safe_float(r.get("x_ohm_per_km", r.get("x_ohm_per_km", 0.412)), 0.412)
                c_nf_per_km = safe_float(r.get("c_nf_per_km", 0.0), 0.0)
                max_i_ka = safe_float(r.get("max_i_ka", 0.2), 0.2)
                pp.create_line_from_parameters(net, from_bus=from_idx, to_bus=to_idx,
                                               length_km=length_km,
                                               r_ohm_per_km=r_ohm_per_km,
                                               x_ohm_per_km=x_ohm_per_km,
                                               c_nf_per_km=c_nf_per_km,
                                               max_i_ka=max_i_ka,
                                               name=str(r.get("line_id", f"line_{from_idx}_{to_idx}")))
            except Exception as e:
                debug(f"[WARN] cannot create line row {_}: {e}")

    # Loads from load_timeseries sheet (we'll aggregate per bus/phase mean or sum)
    load_ts = sheets.get("load_timeseries")
    if load_ts is not None and not load_ts.empty:
        # aggregate by bus and phase summing P_kw
        try:
            # ensure column names standardized
            load_ts = load_ts.rename(columns={c: c.strip() for c in load_ts.columns})
            grouped = load_ts.groupby(["bus", "phase"], dropna=True).agg({"p_kw": "mean", "q_kvar": "mean"}).reset_index()
            for _, r in grouped.iterrows():
                bus_label = r["bus"]
                if bus_label not in bus_map:
                    debug(f"[WARN] load bus {bus_label} not in bus_map -> skipping load")
                    continue
                p_mw = safe_float(r["p_kw"], 0.0) / 1000.0
                q_mvar = safe_float(r["q_kvar"], 0.0) / 1000.0
                if p_mw > 0 or q_mvar > 0:
                    pp.create_load(net, bus=bus_map[bus_label], p_mw=p_mw, q_mvar=q_mvar, name=f"Load_{bus_label}")
        except Exception as e:
            debug(f"[WARN] error processing load_timeseries: {e}")

    else:
        # fallback: try to use data_phase sheet 'P (W)' columns
        data_phase = sheets.get("data_phase")
        if data_phase is not None:
            # try to pick P (W), Unnamed: 2, Unnamed: 3 as earlier
            for _, r in data_phase.iterrows():
                dep = r.get("Départ")
                if dep is None or pd.isna(dep):
                    continue
                # map dep to bus label (if buses are labels like B1..)
                # assume departure value equals bus label in buses sheet or index
                bus_label = None
                # first try exact match
                if dep in bus_map:
                    bus_label = dep
                else:
                    # try "B{dep}" match
                    key = f"B{int(dep)}" if not pd.isna(dep) else None
                    if key in bus_map:
                        bus_label = key
                if bus_label is None:
                    # try first bus
                    bus_idx = list(bus_map.values())[0]
                else:
                    bus_idx = bus_map[bus_label]
                P_w = 0.0
                for colname in ["P (W)", "Unnamed: 2", "Unnamed: 3"]:
                    P_w += safe_float(r.get(colname, 0.0), 0.0)
                p_mw = max(0.0, P_w) / 1e6
                if p_mw > 0:
                    pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=0.0, name=f"Load_dep_{dep}")

    # PV static sheet
    pv_static = sheets.get("pv_static")
    pv_map = {}
    if pv_static is not None and not pv_static.empty:
        for _, r in pv_static.iterrows():
            bus_label = r.get("bus")
            if bus_label not in bus_map:
                debug(f"[WARN] pv bus {bus_label} not in bus_map; skipping PV")
                continue
            p_kw = safe_float(r.get("p_stc_kw", 0.0), 0.0)
            q_kvar = safe_float(r.get("q_kvar", 0.0), 0.0) if "q_kvar" in r.index else 0.0
            s_max_kva = safe_float(r.get("s_max_kva", r.get("s_max_kva", p_kw*1.1)), p_kw*1.1)
            if p_kw > 0:
                try:
                    pp.create_sgen(net, bus=bus_map[bus_label], p_mw=p_kw/1000.0, q_mvar=q_kvar/1000.0,
                                   name=str(r.get("pv_id", f"PV_{bus_label}")))
                    pv_map[bus_map[bus_label]] = pv_map.get(bus_map[bus_label], 0.0) + p_kw
                except Exception as e:
                    debug(f"[WARN] create_sgen failed: {e}")

    # ESS sheet: create storages (note: pandapower storage usage may require external controllers)
    ess_df = sheets.get("ess")
    ess_map = {}
    if ess_df is not None and not ess_df.empty:
        for _, r in ess_df.iterrows():
            bus_label = r.get("bus")
            if bus_label not in bus_map:
                debug(f"[WARN] ESS bus {bus_label} not in bus_map; skipping ESS creation")
                continue
            e_cap_kwh = safe_float(r.get("e_cap_kwh", ESS_CAP_kWh_DEFAULT), ESS_CAP_kWh_DEFAULT)
            p_max_kw = safe_float(r.get("p_max_kw_per_phase", ESS_Pmax_kW_DEFAULT), ESS_Pmax_kW_DEFAULT)
            try:
                pp.create_storage(net, bus=bus_map[bus_label],
                                  p_mw=0.0,
                                  max_e_mwh=max(0.0, e_cap_kwh/1000.0),
                                  sn_mva=max(0.0, p_max_kw/1000.0),
                                  controllable=True,
                                  name=str(r.get("ess_id", f"ESS_{bus_label}")))
                ess_map[bus_map[bus_label]] = {"cap_kwh": e_cap_kwh, "pmax_kw": p_max_kw}
            except Exception as e:
                debug(f"[WARN] create_storage failed: {e}")

    return net, bus_map, pv_map, ess_map

# -----------------------
# Powerflow run with robustness
# -----------------------
def run_pf_robust(net):
    """
    Execute runpp with a few fallback strategies on failure (nr, enforce_q_lims, calculate_voltage_angles).
    """
    try:
        pp.runpp(net)
        return True
    except Exception as e:
        debug(f"[WARN] runpp failed (default): {e}")
    # try Newton-Raphson
    try:
        pp.runpp(net, algorithm="nr")
        return True
    except Exception as e:
        debug(f"[WARN] runpp algorithm='nr' failed: {e}")
    # try enforce Q limits
    try:
        pp.runpp(net, enforce_q_lims=True)
        return True
    except Exception as e:
        debug(f"[WARN] runpp enforce_q_lims failed: {e}")
    # try with voltage angles calculation
    try:
        pp.runpp(net, calculate_voltage_angles=True)
        return True
    except Exception as e:
        debug(f"[ERROR] runpp failed finally: {e}")
        return False

# -----------------------
# Collect results
# -----------------------
def collect_results(net, bus_map) -> pd.DataFrame:
    """
    Collect Vm_pu per bus and losses per line -> DataFrame with columns (Départ, bus_idx, V_pu, P_loss_kW)
    """
    # vm_pu
    vm_series = {}
    if hasattr(net, "res_bus") and "vm_pu" in net.res_bus:
        for idx, v in net.res_bus["vm_pu"].items():
            vm_series[int(idx)] = float(v)
    else:
        # fallback default
        for b in bus_map.values():
            vm_series[b] = 1.0

    # losses
    losses_per_bus = {b: 0.0 for b in bus_map.values()}
    if hasattr(net, "res_line") and "pl_mw" in net.res_line:
        for idx, row in net.line.iterrows():
            pl_mw = float(net.res_line.loc[idx, "pl_mw"]) if idx in net.res_line.index else 0.0
            loss_kw = pl_mw * 1000.0
            # distribute evenly
            fbus = int(row["from_bus"]); tbus = int(row["to_bus"])
            losses_per_bus[fbus] = losses_per_bus.get(fbus, 0.0) + loss_kw/2.0
            losses_per_bus[tbus] = losses_per_bus.get(tbus, 0.0) + loss_kw/2.0

    rows = []
    # reverse map bus index -> departure key
    bus_to_dep = {bus_idx: dep for dep, bus_idx in bus_map.items()}
    for bus_idx, dep in bus_to_dep.items():
        rows.append({
            "Départ": dep,
            "bus_idx": int(bus_idx),
            "V_pu": vm_series.get(bus_idx, 1.0),
            "P_loss_kW": losses_per_bus.get(bus_idx, 0.0)
        })
    df = pd.DataFrame(rows)
    return df

# -----------------------
# Visualizations (many types)
# -----------------------
def plot_line_voltage_by_scenario(df_all: pd.DataFrame, name="voltage_by_scenario"):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=df_all, x="Départ", y="V_pu", hue="Scenario", marker="o", ax=ax)
    ax.axhline(1.05, color="red", linestyle="--", label="Vmax")
    ax.axhline(0.95, color="red", linestyle="--", label="Vmin")
    ax.set_title("Profil de tension par scénario")
    ax.set_ylabel("V (pu)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_bar_losses(df_all: pd.DataFrame, name="losses_bar"):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=df_all, x="Départ", y="P_loss_kW", hue="Scenario", ax=ax)
    ax.set_title("Pertes actives par scénario")
    ax.set_ylabel("P pertes (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_heatmap_voltage(df_all: pd.DataFrame, name="voltage_heatmap"):
    try:
        pivot = df_all.pivot(index="Scenario", columns="Départ", values="V_pu")
    except Exception as e:
        debug(f"[WARN] heatmap pivot failed: {e}")
        return
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label":"V_pu"}, ax=ax)
    ax.set_title("Heatmap de la tension par scénario et départ")
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_pv_capacity_by_depart(pv_map: Dict[int, float], bus_map: Dict[Any,int], name="pv_capacity"):
    if not pv_map:
        debug("[INFO] no PV capacity to plot.")
        return
    # invert bus_map to label by bus_idx
    items = [{"bus_idx": k, "PV_kW": v, "label": next((str(dep) for dep,b in bus_map.items() if b==k), str(k))}
             for k,v in pv_map.items()]
    dfpv = pd.DataFrame(items)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=dfpv, x="label", y="PV_kW", ax=ax, palette="crest")
    ax.set_title("Capacité d'accueil PV par départ (kW)")
    ax.set_xlabel("Départ")
    ax.set_ylabel("PV (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_comparative_gains(summary_df: pd.DataFrame, name="comparative_gains"):
    if summary_df.empty:
        debug("[INFO] no summary to plot comparative gains.")
        return
    melted = summary_df.melt(id_vars="Scenario", var_name="Metric", value_name="Gain_%")
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(data=melted, x="Metric", y="Gain_%", hue="Scenario", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Comparaison des gains par scénario")
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_gains_heatmap(summary_df: pd.DataFrame, name="gains_heatmap"):
    if summary_df.empty:
        return
    pivot = summary_df.set_index("Scenario")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label":"Gain (%)"})
    ax.set_title("Heatmap des gains par scénario et métrique")
    plt.tight_layout()
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_radar_metrics(summary_df: pd.DataFrame, metrics=None, name="radar_metrics"):
    if summary_df.empty:
        return
    if metrics is None:
        metrics = [c for c in summary_df.columns if c != "Scenario"]
    labels = metrics
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    # close the plot
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    for _, r in summary_df.iterrows():
        values = [float(r[m]) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=r["Scenario"])
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar: gains / metrics par scénario")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    if SAVE_FIGURES: save_fig(fig, name)
    plt.show()

def plot_3d_surface_voltage(df_all: pd.DataFrame, name="surface_voltage_3d"):
    """
    Build a surface: x = departure index, y = scenario index, z = V_pu
    """
    try:
        scenarios = df_all["Scenario"].unique().tolist()
        departures = df_all["Départ"].unique().tolist()
        Z = np.zeros((len(scenarios), len(departures)))
        for i, scen in enumerate(scenarios):
            row = df_all[df_all["Scenario"]==scen].set_index("Départ")
            for j, dep in enumerate(departures):
                Z[i, j] = float(row.loc[dep, "V_pu"]) if dep in row.index else np.nan
        X, Y = np.meshgrid(np.arange(len(departures)), np.arange(len(scenarios)))
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xticks(np.arange(len(departures))); ax.set_xticklabels(departures, rotation=45)
        ax.set_yticks(np.arange(len(scenarios))); ax.set_yticklabels(scenarios)
        ax.set_xlabel("Départ"); ax.set_ylabel("Scenario"); ax.set_zlabel("V_pu")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_title("Surface 3D: tension (V_pu) par départ & scénario")
        if SAVE_FIGURES: save_fig(fig, name)
        plt.show()
    except Exception as e:
        debug(f"[WARN] 3D surface plot failed: {e}")

# -----------------------
# Comparisons summary
# -----------------------
def summarize_gains(results_all: pd.DataFrame) -> pd.DataFrame:
    base = results_all[results_all["Scenario"]=="PV_0%"]
    if base.empty:
        debug("[WARN] baseline PV_0% not found; using overall mean as baseline fallback.")
        baseline_loss = results_all["P_loss_kW"].mean()
        baseline_v = results_all["V_pu"].mean()
    else:
        baseline_loss = base["P_loss_kW"].mean()
        baseline_v = base["V_pu"].mean()
    rows = []
    for scen in results_all["Scenario"].unique():
        df = results_all[results_all["Scenario"]==scen]
        loss_mean = df["P_loss_kW"].mean()
        v_mean = df["V_pu"].mean()
        gain_loss = 100.0 * (baseline_loss - loss_mean) / max(abs(baseline_loss), 1e-9)
        gain_v = 100.0 * (v_mean - baseline_v) / max(abs(baseline_v), 1e-9)
        rows.append({"Scenario": scen, "ΔPertes_%": gain_loss, "ΔTension_%": gain_v, "ΔVUF_%": 0.0})
    return pd.DataFrame(rows)

# -----------------------
# Module2 - ESS optimisation (Pyomo)
# -----------------------
def run_ess_pyomo_opt(results_json: str, ess_cap_kwh=ESS_CAP_kWh_DEFAULT, ess_pmax_kw=ESS_Pmax_kW_DEFAULT) -> pd.DataFrame:
    if pyo is None:
        raise ImportError(f"Pyomo not installed: {_pyo_err}")
    # simple solver check GLPK
    solver = pyo.SolverFactory("glpk")
    if not solver.available():
        raise RuntimeError("GLPK solver not available in PATH. Install GLPK to use Pyomo optimisation.")
    # load results
    df_all = pd.read_json(results_json)
    target = f"PV_{int(PV_RATIO_DEFAULT*100)}%"
    df = df_all[df_all["Scenario"]==target].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No scenario {target} found in results for Pyomo optimization.")
    N = len(df)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, N-1)
    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, ess_pmax_kw))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, ess_pmax_kw))
    model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, ess_cap_kwh))
    def p_ess_expr(m, t):
        return m.P_dis[t] - m.P_ch[t]
    model.P_ess = pyo.Expression(model.T, rule=p_ess_expr)
    EFF_CH = 0.95; EFF_DIS = 0.95
    def soc_balance(m, t):
        if t == 0:
            return m.SOC[t] == ess_cap_kwh/2.0
        return m.SOC[t] == m.SOC[t-1] + EFF_CH * m.P_ch[t] - m.P_dis[t] / EFF_DIS
    model.soc_con = pyo.Constraint(model.T, rule=soc_balance)
    P_losses = df["P_loss_kW"].tolist()
    def obj_rule(m):
        # minimize losses - small reward for ESS usage (heuristic)
        return sum(P_losses[t] for t in m.T) - 0.01 * sum(m.P_ess[t] for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    # solve
    debug("Running GLPK solver for ESS optimisation (Pyomo)...")
    res = solver.solve(model, tee=False)
    debug(f"GLPK status: {res.solver.status}, termination: {res.solver.termination_condition}")
    P_ess = [pyo.value(model.P_ess[t]) for t in model.T]
    SOC = [pyo.value(model.SOC[t]) for t in model.T]
    df_out = df.copy()
    df_out["P_ESS_kW_opt"] = P_ess
    df_out["SOC_kWh_opt"] = SOC
    df_out["SOC_%_opt"] = 100.0 * df_out["SOC_kWh_opt"] / ess_cap_kwh
    df_out["P_loss_kW_afterESS_opt"] = np.maximum(df_out["P_loss_kW"] - 0.05 * df_out["P_ESS_kW_opt"], 0.0)
    out_path = os.path.join(OUTPUT_DIR, "summary_ess_pyomo.csv")
    df_out.to_csv(out_path, index=False)
    debug(f"Pyomo optimisation results -> {out_path}")
    return df_out

# -----------------------
# Module2 - ESS heuristic
# -----------------------
def run_ess_heuristic(results_json: str, ess_cap_kwh=ESS_CAP_kWh_DEFAULT, ess_pmax_kw=ESS_Pmax_kW_DEFAULT) -> pd.DataFrame:
    df_all = pd.read_json(results_json)
    target = f"PV_{int(PV_RATIO_DEFAULT*100)}%"
    df = df_all[df_all["Scenario"]==target].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No scenario {target} found for heuristic ESS.")
    V_LOW = 0.97; V_HIGH = 1.03; EFF = 0.95
    soc = ess_cap_kwh/2.0
    soc_list = []
    p_ess_list = []
    for _, r in df.iterrows():
        v = float(r.get("V_pu", 1.0))
        p_now = 0.0
        if v < V_LOW and soc > 0:
            p = min(ess_pmax_kw, soc)
            p_now = p
            soc -= p / EFF
        elif v > V_HIGH and soc < ess_cap_kwh:
            p = min(ess_pmax_kw, ess_cap_kwh - soc)
            p_now = -p
            soc += p * EFF
        soc = max(0.0, min(ess_cap_kwh, soc))
        soc_list.append(soc)
        p_ess_list.append(p_now)
    df_out = df.copy()
    df_out["P_ESS_kW_heur"] = p_ess_list
    df_out["SOC_kWh_heur"] = soc_list
    df_out["SOC_%_heur"] = 100.0 * df_out["SOC_kWh_heur"] / ess_cap_kwh
    df_out["P_loss_kW_afterESS_heur"] = np.maximum(df_out["P_loss_kW"] - 0.05 * df_out["P_ESS_kW_heur"], 0.0)
    out_path = os.path.join(OUTPUT_DIR, "summary_ess_heuristic.csv")
    df_out.to_csv(out_path, index=False)
    debug(f"Heuristic ESS results -> {out_path}")
    return df_out

# -----------------------
# Full orchestration
# -----------------------
def module1_and_plots(sheets: Dict[str, pd.DataFrame]):
    # build net, run three scenarios: PV_0%, PV_x%, PV_x%_ESS
    net0, bus_map, pv_map0, ess_map = create_network_from_sheets(sheets)
    ok0 = run_pf_robust(net0)
    res0 = collect_results(net0, bus_map)
    res0["Scenario"] = "PV_0%"

    # scenario with PV injection: use pv_static + additional scaling
    net1, bus_map1, pv_map1, ess_map1 = create_network_from_sheets(sheets)
    # add dynamic PV scaling from pv_timeseries if present: apply PV_RATIO_DEFAULT fraction of static or timeseries mean
    # We'll simply add additional SGEN proportional to PV_RATIO_DEFAULT * existing or derived loads
    # if pv_static exists we already created static sgen in create_network_from_sheets
    # To simulate larger PV, create new sgens per bus proportional to load if needed
    # We'll add small sgen per bus based on pv_map0 keys
    try:
        # create additional PV sgens if pv_map0 empty: base on loads
        if pv_map0:
            for bus_idx, kw in pv_map0.items():
                p_mw = (kw * PV_RATIO_DEFAULT) / 1000.0
                if p_mw > 0:
                    pp.create_sgen(net1, bus=bus_idx, p_mw=p_mw, q_mvar=0.0, name=f"PV_add_{bus_idx}")
        else:
            # fallback: create PV proportional to mean load per bus (if loads exist)
            # try to compute per-bus load from net
            if hasattr(net1, "load") and not net1.load.empty:
                grouped = net1.load.groupby("bus").p_mw.sum().to_dict()
                for bus_idx, p_mw in grouped.items():
                    add = p_mw * PV_RATIO_DEFAULT
                    pp.create_sgen(net1, bus=bus_idx, p_mw=add, q_mvar=0.0, name=f"PV_add_{bus_idx}")
    except Exception as e:
        debug(f"[WARN] adding PV for scenario 1: {e}")

    ok1 = run_pf_robust(net1)
    res1 = collect_results(net1, bus_map1)
    res1["Scenario"] = f"PV_{int(PV_RATIO_DEFAULT*100)}%"

    # scenario with ESS added (storage components)
    net2, bus_map2, pv_map2, ess_map2 = create_network_from_sheets(sheets)
    # add PV like above
    try:
        if pv_map0:
            for bus_idx, kw in pv_map0.items():
                p_mw = (kw * PV_RATIO_DEFAULT) / 1000.0
                if p_mw > 0:
                    pp.create_sgen(net2, bus=bus_idx, p_mw=p_mw, q_mvar=0.0, name=f"PV_add_{bus_idx}")
    except Exception as e:
        debug(f"[WARN] adding PV for net2: {e}")
    # add ESS storages
    try:
        for bus_idx in bus_map2.values():
            pp.create_storage(net2, bus=bus_idx, p_mw=0.0, max_e_mwh=ESS_CAP_kWh_DEFAULT/1000.0,
                              sn_mva=ESS_Pmax_kW_DEFAULT/1000.0, controllable=True, name=f"ESS_{bus_idx}")
    except Exception as e:
        debug(f"[WARN] adding storages: {e}")
    ok2 = run_pf_robust(net2)
    res2 = collect_results(net2, bus_map2)
    res2["Scenario"] = f"PV_{int(PV_RATIO_DEFAULT*100)}%_ESS"

    results_all = pd.concat([res0, res1, res2], ignore_index=True)
    # export
    json_out = os.path.join(OUTPUT_DIR, "network_results.json")
    csv_out = os.path.join(OUTPUT_DIR, "network_results.csv")
    results_all.to_json(json_out, orient="records")
    results_all.to_csv(csv_out, index=False)
    debug(f"Module1: exported -> {json_out}, {csv_out}")

    # Summaries and plots
    summary = summarize_gains(results_all)
    # plots
    plot_line_voltage_by_scenario(results_all)
    plot_bar_losses(results_all)
    plot_heatmap_voltage(results_all)
    plot_pv_capacity_by_depart(pv_map1 if pv_map1 else pv_map0, bus_map)
    plot_comparative_gains(summary)
    plot_gains_heatmap(summary)
    plot_radar_metrics(summary)
    plot_3d_surface_voltage(results_all)

    # Additional 2D line per bus
    try:
        fig, ax = plt.subplots(figsize=(10,5))
        for dep in results_all["Départ"].unique():
            subset = results_all[results_all["Départ"]==dep]
            sns.lineplot(x="Scenario", y="V_pu", data=subset, label=str(dep), marker="o", ax=ax)
        ax.set_title("Tension par départ (ligne par départ) - scenarios")
        plt.xticks(rotation=45)
        if SAVE_FIGURES: save_fig(fig, "voltage_by_depart_lines")
        plt.show()
    except Exception as e:
        debug(f"[WARN] additional 2D line plot failed: {e}")

    return json_out, csv_out, results_all

# -----------------------
# Main CLI
# -----------------------
def main():
    try:
        path = find_input()
    except Exception as e:
        print(f"[FATAL] Input file not found: {e}")
        sys.exit(1)

    sheets = load_all_sheets(path)

    # run module1 and plots
    try:
        json_out, csv_out, results_all = module1_and_plots(sheets)
        debug("Module1 completed.")
    except Exception as e:
        debug(f"[ERROR] module1 failed: {e}")
        traceback.print_exc()
        results_all = pd.DataFrame()

    # Module2: ESS optimisation if available else heuristic
    pyomo_ok = (pyo is not None)
    glpk_ok = False
    if pyomo_ok:
        try:
            solver = pyo.SolverFactory("glpk")
            glpk_ok = solver.available()
        except Exception:
            glpk_ok = False

    try:
        if pyomo_ok and glpk_ok:
            debug("Running Pyomo optimisation (Module2).")
            df_opt = run_ess_pyomo_opt(os.path.join(OUTPUT_DIR, "network_results.json"))
            # plots: SOC and ESS pow
            fig, ax = plt.subplots(figsize=(8,4))
            sns.lineplot(data=df_opt, x="Départ", y="SOC_%_opt", marker="o", ax=ax)
            ax.set_title("SOC ESS (optimisé)")
            plt.xticks(rotation=45); plt.tight_layout()
            if SAVE_FIGURES: save_fig(fig, "SOC_opt")
            plt.show()

            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=df_opt, x="Départ", y="P_ESS_kW_opt", ax=ax)
            ax.set_title("Puissance ESS (optimisée)")
            plt.xticks(rotation=45); plt.tight_layout()
            if SAVE_FIGURES: save_fig(fig, "PESS_opt")
            plt.show()
        else:
            debug("Pyomo/GLPK not available; running heuristic ESS (Module2).")
            df_heur = run_ess_heuristic(os.path.join(OUTPUT_DIR, "network_results.json"))
            fig, ax = plt.subplots(figsize=(8,4))
            sns.lineplot(data=df_heur, x="Départ", y="SOC_%_heur", marker="o", ax=ax)
            ax.set_title("SOC ESS (heuristique)")
            plt.xticks(rotation=45); plt.tight_layout()
            if SAVE_FIGURES: save_fig(fig, "SOC_heur")
            plt.show()

            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=df_heur, x="Départ", y="P_ESS_kW_heur", ax=ax)
            ax.set_title("Puissance ESS (heuristique)")
            plt.xticks(rotation=45); plt.tight_layout()
            if SAVE_FIGURES: save_fig(fig, "PESS_heur")
            plt.show()
    except Exception as e:
        debug(f"[ERROR] Module2 failed: {e}")
        traceback.print_exc()

    debug("All done. Check outputs/ for PNG and CSV files.")

if __name__ == "__main__":
    main()

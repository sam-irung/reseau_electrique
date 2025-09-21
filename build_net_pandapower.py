#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1 - Analyse réseau avec pandapower
Lecture Donnees.xlsx -> Construction réseau -> Simulation scénarios
Comparaison avant/après PV + ESS
Visualisations (matplotlib + seaborn) + Export JSON + Sauvegarde PNG
"""

import os, json
import numpy as np
import pandas as pd
import pandapower as pp
import seaborn as sns
import matplotlib.pyplot as plt

# --- PARAMÈTRES ---
F_EXCEL = "Donnees.xlsx"
PV_RATIO_DEFAULT = 0.2
ESS_CAP_kWh = 20.0
ESS_Pmax_kW = 4.0
SAVE_FIGURES = True  # passer à False si tu ne veux pas sauvegarder les graphiques

sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 110


# --- FONCTIONS UTILES ---
def clean_string_number(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(';', '.').replace(' ', '')
        if '.' in s and ',' in s:
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '.')
        return float(s)
    except:
        return np.nan


def load_data():
    if not os.path.exists(F_EXCEL):
        raise FileNotFoundError(f"❌ Fichier introuvable : {F_EXCEL}")
    df = pd.read_excel(F_EXCEL, dtype=str)
    if df.empty:
        raise ValueError("❌ Fichier vide ! Ajoute des données avant de lancer le script.")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].apply(clean_string_number)
    if "Départ" not in df.columns:
        raise KeyError("❌ Colonne 'Départ' manquante dans le fichier.")
    df["Départ"] = df["Départ"].ffill()
    df = df.fillna(0.0)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# --- RÉSEAU ---
def build_net(df):
    net = pp.create_empty_network()
    buses = {}
    for dep in df["Départ"].unique():
        buses[dep] = pp.create_bus(net, vn_kv=0.4, name=f"Bus_{dep}")

    # Slack sur le premier bus
    pp.create_ext_grid(net, bus=buses[df["Départ"].iloc[0]], vm_pu=1.0)

    dep_list = list(df["Départ"].unique())
    for i in range(len(dep_list) - 1):
        dep_from, dep_to = dep_list[i], dep_list[i + 1]
        length_km = df.loc[df["Départ"] == dep_to, "L (m)"].mean() / 1000 if "L (m)" in df.columns else 0.05
        S = df.loc[df["Départ"] == dep_to, "S (mm²)"].mean() if "S (mm²)" in df.columns else 95.0
        if S <= 0: S = 95.0
        if length_km <= 0: length_km = 0.05
        R_km = 17.24 / S
        X_km = 0.3 * R_km
        try:
            pp.create_line_from_parameters(net, buses[dep_from], buses[dep_to],
                                           length_km=length_km,
                                           r_ohm_per_km=R_km, x_ohm_per_km=X_km,
                                           c_nf_per_km=0, max_i_ka=0.2,
                                           name=f"Ligne_{dep_from}_{dep_to}")
        except Exception as e:
            print(f"[WARN] Ligne {dep_from}->{dep_to} ignorée : {e}")

    for _, row in df.iterrows():
        bus = buses[row["Départ"]]
        P = (row.get("P (W)", 0) + row.get("Unnamed: 2", 0) + row.get("Unnamed: 3", 0)) / 1e6
        if P > 0:
            pp.create_load(net, bus=bus, p_mw=P, q_mvar=0.0, name=f"Load_{bus}")

    return net, buses


def add_pv(net, buses, df, ratio=PV_RATIO_DEFAULT):
    total_pv = {}
    for _, row in df.iterrows():
        bus = buses[row["Départ"]]
        P_pv = (row.get("P (W)", 0) + row.get("Unnamed: 2", 0) + row.get("Unnamed: 3", 0)) / 1e6 * ratio
        if P_pv > 0:
            pp.create_sgen(net, bus=bus, p_mw=P_pv, q_mvar=0.0, name=f"PV_{bus}")
        total_pv[bus] = total_pv.get(bus, 0) + P_pv * 1000
    return total_pv


def add_ess(net, buses):
    for bus in buses.values():
        pp.create_storage(net, bus=bus, p_mw=0.0, max_e_mwh=ESS_CAP_kWh / 1000,
                          sn_mva=ESS_Pmax_kW / 1000, name=f"ESS_{bus}", controllable=True)


def compute_vuf(Uabc):
    Ua, Ub, Uc = Uabc
    Uavg = (Ua + Ub + Uc) / 3
    if Uavg == 0: return 0.0
    return max(abs(Ua - Uavg), abs(Ub - Uavg), abs(Uc - Uavg)) / Uavg


def run_scenario(df, ratio=0.0, use_ess=False):
    net, buses = build_net(df)
    pv_by_bus = {}
    if ratio > 0:
        pv_by_bus = add_pv(net, buses, df, ratio=ratio)
    if use_ess:
        add_ess(net, buses)

    try:
        pp.runpp(net)
    except Exception as e:
        print(f"[WARN] runpp a échoué : {e}")

    vuf_vals = []
    for b in buses.values():
        Ua = Ub = Uc = net.res_bus.vm_pu.get(b, 1.0) if hasattr(net.res_bus, "vm_pu") else 1.0
        vuf_vals.append(compute_vuf([Ua, Ub, Uc]))

    P_loss_kW = {b: 0.0 for b in buses.values()}
    if hasattr(net, "res_line") and hasattr(net.res_line, "pl_mw"):
        for idx, line in net.line.iterrows():
            from_bus = line["from_bus"]
            to_bus = line["to_bus"]
            loss = net.res_line.pl_mw[idx] * 1000
            P_loss_kW[from_bus] += loss / 2
            P_loss_kW[to_bus] += loss / 2

    res = pd.DataFrame({
        "Départ": list(buses.keys()),
        "V_pu": [net.res_bus.vm_pu.get(b, 1.0) for b in buses.values()],
        "P_loss_kW": [P_loss_kW[b] for b in buses.values()],
        "VUF": vuf_vals
    })
    res["Scenario"] = f"PV_{ratio*100:.0f}%" + ("_ESS" if use_ess else "")
    return res, pv_by_bus, net


# --- COMPARAISONS & VISUALISATIONS ---
def compare_scenarios(results):
    summary = []
    base = results[results["Scenario"] == "PV_0%"]
    for scen in results["Scenario"].unique():
        df = results[results["Scenario"] == scen]
        gain_loss = 100 * (base["P_loss_kW"].mean() - df["P_loss_kW"].mean()) / max(base["P_loss_kW"].mean(), 1e-9)
        gain_v = 100 * (df["V_pu"].mean() - base["V_pu"].mean()) / max(base["V_pu"].mean(), 1e-9)
        red_vuf = 100 * (base["VUF"].mean() - df["VUF"].mean()) / max(base["VUF"].mean(), 1e-9)
        summary.append({"Scenario": scen, "ΔPertes_%": gain_loss, "ΔTension_%": gain_v, "ΔVUF_%": red_vuf})
    return pd.DataFrame(summary)


def save_or_show(fig, name):
    if SAVE_FIGURES:
        fig.savefig(f"{name}.png", dpi=150)
        print(f"📁 Figure sauvegardée : {name}.png")
    plt.show()


def plot_voltage_comparison(results):
    fig = plt.figure(figsize=(10, 5))
    sns.lineplot(data=results, x="Départ", y="V_pu", hue="Scenario", marker="o")
    plt.axhline(1.05, color="red", linestyle="--")
    plt.axhline(0.95, color="red", linestyle="--")
    plt.title("Profil de tension par scénario")
    plt.ylabel("Tension [pu]")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_or_show(fig, "voltage_comparison")


def plot_losses(results):
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(data=results, x="Départ", y="P_loss_kW", hue="Scenario")
    plt.title("Pertes actives par scénario")
    plt.ylabel("Pertes [kW]")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_or_show(fig, "losses_comparison")


def plot_voltage_heatmap(results):
    pivot = results.pivot(index="Scenario", columns="Départ", values="V_pu")
    fig = plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Tension [pu]'})
    plt.title("Carte thermique de la tension par scénario et départ")
    plt.tight_layout()
    save_or_show(fig, "voltage_heatmap")


def plot_pv_capacity(pv_by_bus):
    if not pv_by_bus:
        return
    pv_df = pd.DataFrame(list(pv_by_bus.items()), columns=["Bus", "PV_kW"])
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(data=pv_df, x="Bus", y="PV_kW", palette="crest")
    plt.title("Capacité d'accueil PV par départ")
    plt.ylabel("PV injecté [kW]")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_or_show(fig, "pv_capacity")


def plot_comparative_gains(summary):
    fig = plt.figure(figsize=(8, 5))
    summary_melted = summary.melt(id_vars="Scenario", var_name="Metric", value_name="Gain_%")
    sns.barplot(data=summary_melted, x="Metric", y="Gain_%", hue="Scenario", palette="viridis")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Comparaison des gains par scénario")
    plt.ylabel("Gain / Amélioration [%]")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_or_show(fig, "comparative_gains")

def plot_comparative_heatmap(summary):
    """Affiche une heatmap des gains en % pour chaque scénario et chaque métrique."""
    if summary.empty:
        print("⚠️ Aucun résultat pour la heatmap comparative.")
        return

    pivot = summary.set_index("Scenario")
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                cbar_kws={'label': 'Gain [%]'})
    plt.title("Heatmap des gains par scénario et métrique")
    plt.tight_layout()
    save_or_show(fig, "gains_heatmap")



def export_json(results, summary):
    with open("network_results.json", "w") as f:
        json.dump(results.to_dict(orient="list"), f, indent=4)
    with open("network_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="list"), f, indent=4)
    print("📁 Résultats exportés -> network_results.json & network_summary.json")


# --- MAIN ---
def main():
    df = load_data()
    results_all, pv_capacity = [], {}

    res0, _, _ = run_scenario(df, ratio=0.0)
    res1, pv_capacity, _ = run_scenario(df, ratio=PV_RATIO_DEFAULT)
    res2, _, _ = run_scenario(df, ratio=PV_RATIO_DEFAULT, use_ess=True)

    results_all.extend([res0, res1, res2])
    results = pd.concat(results_all, ignore_index=True)

    print("📊 Résultats détaillés :")
    print(results)

    summary = compare_scenarios(results)
    print("\n📊 Résumé comparatif (gains) :")
    print(summary)

    # Visualisations
    plot_voltage_comparison(results)
    plot_losses(results)
    plot_voltage_heatmap(results)
    plot_pv_capacity(pv_capacity)
    plot_comparative_gains(summary)
    plot_comparative_heatmap(summary)


    export_json(results, summary)


if __name__ == "__main__":
    main()

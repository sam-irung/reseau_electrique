#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse réseau avancée avec PV et ESS
Version robuste : toutes les colonnes numériques sont forcées float, valeurs manquantes remplacées par 0.0.
Visualisations avancées avec Seaborn et Matplotlib.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 110

F_EXCEL = "Donnees.xlsx"
USE_CSV_FALLBACK = True
PV_INJECTION_RATIO_DEFAULT = 0.5
ESS_CAP_kWh_DEFAULT = 20.0
ESS_Pmax_kW_DEFAULT = 4.0

# --- Fonctions utilitaires ---
def find_file():
    if os.path.exists(F_EXCEL):
        return F_EXCEL
    if USE_CSV_FALLBACK and os.path.exists(F_EXCEL.replace('.xlsx', '.csv')):
        return F_EXCEL.replace('.xlsx', '.csv')
    raise FileNotFoundError(f"Fichier introuvable : {F_EXCEL}.")

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

def safe_float_col(df, col):
    """Force une colonne en float, remplace NaN par 0.0"""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    else:
        df[col] = 0.0
        print(f"⚠️ Colonne {col} manquante, remplacée par 0.0")

# --- Chargement & nettoyage ---
def load_and_clean():
    path = find_file()
    print(f"Lecture du fichier : {path}")
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)

    # Nettoyage noms colonnes
    df.columns = [str(c).strip().replace(' ', '_') if c is not None else "Unnamed" for c in df.columns]

    # Nettoyage des valeurs
    for col in df.columns:
        df[col] = df[col].apply(clean_string_number)

    # Départ forward fill
    if 'Départ' in df.columns:
        df['Départ'] = df['Départ'].ffill()
    else:
        raise KeyError("Colonne 'Départ' introuvable dans le fichier.")

    df = df.fillna(0.0)
    print(f"Données nettoyées, shape={df.shape}")
    return df

# --- Détection colonnes phases ---
def detect_phase_columns(df):
    cols = {'P':{}, 'U':{}, 'I':{}, 'L':{}, 'S':{}}
    for ph in ['a','b','c']:
        for t in ['P','U','I','L','S']:
            candidates = [c for c in df.columns if t.lower() in c.lower() and ph in c.lower()]
            cols[t][ph] = candidates[0] if candidates else None
    print("Colonnes détectées par phase :", cols)
    return cols

# --- Calculs principaux ---
def compute_vuf_row(Ua, Ub, Uc):
    if any(pd.isna(x) for x in [Ua, Ub, Uc]):
        return 0.0
    Uavg = (Ua + Ub + Uc) / 3.0
    return max(abs(Ua - Uavg), abs(Ub - Uavg), abs(Uc - Uavg)) / Uavg if Uavg != 0 else 0.0

def compute_Rkm_from_section(S_mm2):
    return 17.24 / S_mm2 if S_mm2 > 0 else 0.0

def compute_losses_per_line(I_A, L_m, S_mm2):
    if I_A==0 or L_m==0 or S_mm2==0:
        return 0.0
    R_line = compute_Rkm_from_section(S_mm2) * (L_m/1000.0)
    return (I_A**2) * R_line / 1000.0

# --- Scénarios PV et ESS ---
def simulate_pv_injection(df, cols, ratio=PV_INJECTION_RATIO_DEFAULT):
    for ph in ['a','b','c']:
        pcol = cols['P'].get(ph)
        pvcol = f'PV_{ph}_kW'
        if pcol and pcol in df.columns:
            df[pvcol] = (df[pcol] / 1000.0) * ratio
        else:
            df[pvcol] = 0.0
    # Pnet after PV
    for ph in ['a','b','c']:
        df[f'Pnet_{ph}_afterPV'] = df.get(f'P_{ph}_kW', 0.0) - df.get(f'PV_{ph}_kW', 0.0)
    # SOC simulé simple ESS
    df['ESS_SOC_%'] = np.clip((ESS_CAP_kWh_DEFAULT - df[[f'Pnet_{ph}_afterPV' for ph in ['a','b','c']]].sum(axis=1)/ESS_Pmax_kW_DEFAULT)*5,0,100)
    return df

# --- Visualisations ---
def plot_vuf_bar(df):
    if 'VUF' not in df.columns:
        return
    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x='Départ', y='VUF')
    plt.title("VUF par départ")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pv_vs_pnet(df):
    plt.figure(figsize=(10,5))
    phases = ['a','b','c']
    for ph in phases:
        plt.plot(df['Départ'], df[f'PV_{ph}_kW'], label=f'PV {ph}')
        plt.plot(df['Départ'], df[f'Pnet_{ph}_afterPV'], '--', label=f'Pnet {ph}')
    plt.title("Comparatif PV injecté et Pnet")
    plt.xticks(rotation=45)
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_voltage_heatmap(df, cols):
    U_cols = [cols['U'].get(ph) for ph in ['a','b','c']]
    voltages = df[U_cols].copy()
    voltages.index = df['Départ']
    plt.figure(figsize=(10,5))
    sns.heatmap(voltages.T, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Heatmap tensions par phase")
    plt.ylabel("Phase")
    plt.xlabel("Départ")
    plt.tight_layout()
    plt.show()

def plot_ess_hist(df):
    if 'ESS_SOC_%' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['ESS_SOC_%'], bins=20, kde=True, color='orange')
        plt.title("Histogramme SOC ESS simulé")
        plt.xlabel("SOC (%)")
        plt.tight_layout()
        plt.show()

# --- Main ---
def main():
    df = load_and_clean()
    cols = detect_phase_columns(df)

    # Assurer toutes les colonnes P en kW
    for ph in ['a','b','c']:
        pcol = cols['P'].get(ph)
        safe_float_col(df, pcol)
        df[f'P_{ph}_kW'] = df.get(pcol, 0.0)/1000.0

    df = simulate_pv_injection(df, cols)

    # Calcul VUF
    u_a = cols['U'].get('a'); u_b = cols['U'].get('b'); u_c = cols['U'].get('c')
    safe_float_col(df, u_a); safe_float_col(df, u_b); safe_float_col(df, u_c)
    df['VUF'] = df.apply(lambda r: compute_vuf_row(r.get(u_a,0.0), r.get(u_b,0.0), r.get(u_c,0.0)), axis=1)

    # Losses
    for ph in ['a','b','c']:
        safe_float_col(df, cols['I'].get(ph))
        safe_float_col(df, cols['L'].get(ph))
        safe_float_col(df, cols['S'].get(ph))
    df['Loss_kW'] = df.apply(lambda r: sum(compute_losses_per_line(
        r.get(cols['I'].get(ph),0.0),
        r.get(cols['L'].get(ph),0.0),
        r.get(cols['S'].get(ph),0.0)
    ) for ph in ['a','b','c']), axis=1)

    # Affichage rapide
    display_cols = ['Départ','VUF','Loss_kW','PV_a_kW','Pnet_a_afterPV','ESS_SOC_%']
    print(df[display_cols].head())

    # Graphiques
    plot_vuf_bar(df)
    plot_pv_vs_pnet(df)
    plot_voltage_heatmap(df, cols)
    plot_ess_hist(df)

if __name__ == "__main__":
    main()

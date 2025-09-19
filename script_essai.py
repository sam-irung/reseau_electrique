#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse et visualisations réseau (Seaborn) à partir de Donnees.xlsx
- Lecture robuste des données (virgules, ;, NaN)
- Détection automatique des colonnes par phase (P, U, I, L, S)
- Calculs : VUF (approx.), pertes (si I & S disponibles), capacité PV simulée, SOC ESS (simple)
- Scénarios : avant / après injection PV ; avec / sans ESS
- Visualisations : line VUF, tensions avant/après, bar PV capacity, heatmap PV par depart×phase,
  comparatifs pertes avant/après, SOC, gains VUF et pertes
"""

import os
import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration seaborn / matplotlib
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 110

# --- Paramètres modifiables ---
F_EXCEL = "Donnees.xlsx"        # nom du fichier Excel (ou CSV si tu préfères)
USE_CSV_FALLBACK = True         # si Excel absent et CSV existe, utilisera Donnees.csv
PV_INJECTION_RATIO_DEFAULT = 0.5   # fraction de la charge pouvant être injectée comme PV si pas de donnée
ESS_CAP_kWh_DEFAULT = 20.0         # capacité ESS par départ (par défaut si simulation)
ESS_Pmax_kW_DEFAULT = 4.0          # puissance max ESS (kW)
BASE_SBASE_kVA = 100.0             # pour conversions pu si besoin (non strictement utilisé ici)

# --- Fonctions utilitaires ---

def find_file():
    """Retourne le chemin du fichier Donnees.xlsx ou Donnees.csv si présent."""
    if os.path.exists(F_EXCEL):
        return F_EXCEL
    if USE_CSV_FALLBACK and os.path.exists(F_EXCEL.replace('.xlsx', '.csv')):
        return F_EXCEL.replace('.xlsx', '.csv')
    raise FileNotFoundError(f"Fichier introuvable : {F_EXCEL}. Place-le dans le répertoire courant.")

def clean_string_number(x):
    """Nettoie une valeur pouvant contenir '1.234' ou '1,234' ou '1;234' et renvoie float ou NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # Remplacer les séparateurs mixtes ; -> .
    s = s.replace(';', '.')
    # Si plusieurs séparateurs décimaux (ex: "1.234,56"), remplacer les milliers
    # approche conservative: remplacer ',' par '.' si pas déjà un '.'
    # mais d'abord supprimer espaces
    s = s.replace(' ', '')
    # If string contains both '.' and ',' assume '.' thousands and ',' decimal -> swap
    if '.' in s and ',' in s:
        # e.g. "1.234,56" -> "1234.56"
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return np.nan

def safe_print_df_info(df, name="DataFrame"):
    print(f"{name} shape: {df.shape}")
    print("Colonnes:", list(df.columns[:30]))
    print(df.head(3).to_string(index=False))

# --- Chargement & nettoyage ---

def load_and_clean():
    """Lit Donnees.xlsx (ou .csv), nettoie les colonnes numériques et normalise les noms."""
    path = find_file()
    print(f"Lecture du fichier : {path}")
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path, header=0, dtype=str)
    else:
        df = pd.read_excel(path, header=0, dtype=str)

    # Nettoyage noms colonnes : trim & retirer espaces multiples, remplacer caractères spéciaux
    orig_cols = list(df.columns)
    new_cols = []
    for c in orig_cols:
        if c is None:
            new_cols.append("Unnamed")
            continue
        s = str(c).strip()
        s = " ".join(s.split())  # normalize spaces
        new_cols.append(s)
    df.columns = new_cols

    # Remplacer valeurs ; , etc dans toutes les cellules (string), puis tenter numeric conversion
    # Appliquer clean_string_number pour colonnes plausiblement numériques
    # On détecte colonnes numériques candidate: celles qui contiennent au moins une valeur numérique after cleaning
    for col in df.columns:
        # try quick detection: if any char digit in column values -> apply cleaning
        sample = df[col].dropna().astype(str).head(20).tolist()
        has_digit = any(any(ch.isdigit() for ch in v) for v in sample)
        if has_digit:
            df[col] = df[col].apply(clean_string_number)

    # Gestion colonne 'Départ' : forward fill si NaN
    if 'Départ' in df.columns:
        # if values numeric convert to int where possible
        df['Départ'] = df['Départ'].ffill()
        # if still floats, cast to int when safe
        try:
            df['Départ'] = df['Départ'].astype(float)
            df['Départ'] = df['Départ'].apply(lambda x: int(x) if not np.isnan(x) else np.nan)
        except:
            pass
    else:
        raise KeyError("Colonne 'Départ' introuvable dans le fichier. Renomme la colonne correspondante en 'Départ'.")

    # Supprimer colonnes entièrement vides
    df = df.dropna(axis=1, how='all')

    safe_print_df_info(df, "Données nettoyées")
    return df

# --- Détection des colonnes par phase ---
def detect_phase_columns(df):
    """
    Recherche automatiquement les colonnes P (W), U (V), I (A), L (m), S (mm²) par phase.
    Retourne dictionnaire de la forme:
    cols = {
      'P': {'a':colname_a, 'b':colname_b, 'c':colname_c},
      'U': {...}, 'I': {...}, 'L': {...}, 'S': {...}
    }
    """
    cols = {'P':{}, 'U':{}, 'I':{}, 'L':{}, 'S':{}}
    ph_labels = {'a':['a','ph a','ph_a','ph.a','pha'],
                 'b':['b','ph b','ph_b','ph.b','phb'],
                 'c':['c','ph c','ph_c','ph.c','phc']}

    # Normalize column names to lower for matching
    colmap = {c: c.lower() for c in df.columns}

    for c_orig, c_low in colmap.items():
        # try to identify type and phase
        # type keywords
        for tkey, tnames in [('P', ['p (w)','p (kw)','p (w)_','p_','p(w)','p(w)_','p (w)']),
                             ('U', ['u (v)','u (v)_','u_','v (v)','v(v)','u(v)']),
                             ('I', ['i (a)','i (a)_','i_','i(a)']),
                             ('L', ['l (m)','l(m)','l_','length','length_m','l (m)']),
                             ('S', ['s (mm²)','s (mm2)','s (mm²)_','section','s_','s (mm²)','mm2','mm²'])]:
            for name in tnames:
                if name in c_low:
                    # then detect phase by checking presence of phase tokens
                    for ph, tokens in ph_labels.items():
                        if any(tok in c_low for tok in tokens):
                            cols[tkey][ph] = c_orig
                    # if no explicit phase in column name, maybe it's a block P (W) with phases in subsequent columns
                    # we'll handle missing above after scanning
    # Fallback heuristics:
    # If there exist exactly 3 columns with 'P' but without explicit ph detected, attempt to map by column order:
    def find_unmapped(df, key):
        # columns that look like they belong to key type
        possible = [c for c in df.columns if any(s in c.lower() for s in ['p (w)','p (kw)','p(','p_','p (w)_','p']) or key=='P' and 'p' in c.lower()]
        return possible

    # If mapping incomplete, attempt to map by patterns like "Ph a" etc.
    for key in cols.keys():
        # count detected
        detected = cols[key]
        if len(detected) < 3:
            # attempt to find columns containing 'ph a','ph b','ph c' and containing type keywords like 'p','u','i','l','s'
            for ph in ['a','b','c']:
                if ph not in cols[key]:
                    candidate = None
                    # search for columns containing both ph token and type token
                    for c in df.columns:
                        c_low = c.lower()
                        if f'ph {ph}' in c_low or f'ph{ph}' in c_low or f' {ph}' in c_low:
                            # check type keywords
                            if key == 'P' and ('p' in c_low or 'p (w)' in c_low):
                                candidate = c; break
                            if key == 'U' and ('u' in c_low or 'v (v)' in c_low):
                                candidate = c; break
                            if key == 'I' and ('i' in c_low or 'i (a)' in c_low):
                                candidate = c; break
                            if key == 'L' and ('l' in c_low or 'l (m)' in c_low or 'length' in c_low):
                                candidate = c; break
                            if key == 'S' and ('s' in c_low or 'mm' in c_low):
                                candidate = c; break
                    if candidate:
                        cols[key][ph] = candidate

    # Final fallback: if still missing, attempt to assign by ordering groups: find groups of 3 numeric columns that look similar
    for key in cols.keys():
        if len(cols[key]) < 3:
            # find numeric columns not yet assigned
            assigned = set(sum([list(d.keys()) for d in [cols[key]]], []))
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            # heuristic: find three consecutive numeric columns not used
            # simpler: if exactly 3 numeric columns exist for this key pattern by name, map them in order a,b,c
            pattern_keywords = {
                'P': ['p (w)','p (kw)','p '],
                'U': ['u (v)','v (v)','u '],
                'I': ['i (a)','i '],
                'L': ['l (m)','length','l '],
                'S': ['s (mm','section','mm2']
            }
            cand = [c for c in df.columns if any(k in c.lower() for k in pattern_keywords[key])]
            if len(cand) >= 3:
                # map first three to a,b,c
                for ph, ccol in zip(['a','b','c'], cand[:3]):
                    cols[key][ph] = ccol

    # Print summary
    print("Détection colonnes par type/phase :")
    for k, mapping in cols.items():
        print(f"  {k}: {mapping}")
    return cols

# --- Calculs principaux ---

def compute_vuf_row(Ua, Ub, Uc):
    """Calcul VUF approché : max deviation from average divided by average (ratio)."""
    if any(pd.isna(x) for x in [Ua, Ub, Uc]):
        return np.nan
    Uavg = (Ua + Ub + Uc) / 3.0
    if Uavg == 0:
        return np.nan
    return float(max(abs(Ua - Uavg), abs(Ub - Uavg), abs(Uc - Uavg)) / Uavg)

def compute_Rkm_from_section(S_mm2):
    """
    Calcule R (ohm/km) à partir de la section en mm² (cuivre).
    R_km ≈ ρ*1000 / A where A in m².
    Avec ρ (Cu) ≈ 1.724e-8 Ω·m => Rkm ≈ 17.24 / S_mm2 (Ω/km)
    """
    try:
        if np.isnan(S_mm2) or S_mm2 <= 0:
            return np.nan
        return 17.24 / float(S_mm2)
    except:
        return np.nan

def compute_losses_per_line(I_A, L_m, S_mm2):
    """
    Calcule pertes sur un tronçon: Ploss_W = I^2 * R_line
    R_line = R_km * L_km
    Retourne Ploss_kW
    """
    if any(pd.isna(x) for x in [I_A, L_m, S_mm2]):
        return np.nan
    Rkm = compute_Rkm_from_section(S_mm2)
    if np.isnan(Rkm):
        return np.nan
    L_km = L_m / 1000.0
    R_line = Rkm * L_km
    Ploss_W = (I_A ** 2) * R_line
    return Ploss_W / 1000.0  # kW

# --- Scénarios PV & ESS (simples / adaptatifs) ---

def simulate_pv_injection(df, cols, ratio=PV_INJECTION_RATIO_DEFAULT):
    """
    Simule une injection PV par défaut : PV_inj_kW_phase = ratio * load_kW_phase (si charge P disponible).
    Si déjà existe une colonne 'PV...' on la respecte.
    Retourne df avec colonnes ajoutées 'PV_a_kW', 'PV_b_kW', 'PV_c_kW' et 'P_net_after' etc.
    """
    for ph in ['a','b','c']:
        pcol = cols['P'].get(ph)
        pvcol = f'PV_{ph}_kW'
        if pcol:
            # P is likely W; convert to kW
            df[pvcol] = (df[pcol] / 1000.0) * ratio
        else:
            df[pvcol] = 0.0
    # puissance nette après injection (per phase) = load - pv_inj
    df['Pnet_a_kW_afterPV'] = (df[cols['P'].get('a')] / 1000.0).fillna(0) - df['PV_a_kW']
    df['Pnet_b_kW_afterPV'] = (df[cols['P'].get('b')] / 1000.0).fillna(0) - df['PV_b_kW']
    df['Pnet_c_kW_afterPV'] = (df[cols['P'].get('c')] / 1000.0).fillna(0) - df['PV_c_kW']
    return df

def simulate_ess_smoothing(df, cols, ess_cap_kwh=ESS_CAP_kWh_DEFAULT, ess_pmax_kW=ESS_Pmax_kW_DEFAULT):
    """
    Simulation heuristique d'un ESS par départ (statique/simplifiée).
    Si serie temporelle (col 't') existante : on lisse la courbe Pnet_afterPV en simulant charge/discharge.
    Si snapshot unique : on applique un effet proportionnel de réduction des déséquilibres (approx.).
    Ajoute colonnes 'Pnet_*_afterPV_ESS', 'soc_end' (approx).
    """
    # On détecte si dataframe contient une colonne 't' -> time series
    if 't' in df.columns:
        # Pour chaque départ indépendamment, simuler ESS time series
        results = []
        for dep, g in df.groupby('Départ'):
            g = g.sort_values('t').copy()
            # combine phases into total net load (kW)
            g['Pnet_total_afterPV'] = g.get('Pnet_a_kW_afterPV', 0).fillna(0) + g.get('Pnet_b_kW_afterPV', 0).fillna(0) + g.get('Pnet_c_kW_afterPV', 0).fillna(0)
            # simple battery dispatch: charge when Pnet negative (export) up to Pmax, discharge when Pnet high to cap Pmax
            soc = ess_cap_kwh * 0.5  # start at 50%
            soc_series = []
            Pchg = []
            Pdis = []
            dt = 1.0  # assuming hourly steps if time is hourly, best-effort
            for idx,row in g.iterrows():
                p = row['Pnet_total_afterPV']
                # if p > threshold, discharge to reduce peak
                if p > ess_pmax_kW:
                    discharge = min(p - ess_pmax_kW, ess_pmax_kW, soc / dt)
                    soc -= discharge * dt
                    Pdis.append(discharge)
                    Pchg.append(0.0)
                elif p < -ess_pmax_kW:
                    charge = min(-p - ess_pmax_kW, ess_pmax_kW)
                    soc += charge * dt
                    Pchg.append(charge)
                    Pdis.append(0.0)
                else:
                    Pchg.append(0.0)
                    Pdis.append(0.0)
                soc = max(0.0, min(ess_cap_kwh, soc))
                soc_series.append(soc)
            g['ESS_Pchg_kW'] = Pchg
            g['ESS_Pdis_kW'] = Pdis
            g['ESS_SOC_kWh'] = soc_series
            # compute adjusted Pnet_total after ESS
            g['Pnet_total_afterPV_ESS'] = g['Pnet_total_afterPV'] - g['ESS_Pdis_kW'] + g['ESS_Pchg_kW']
            results.append(g)
        df_out = pd.concat(results, ignore_index=True)
        # split adjusted Pnet back to phases proportionally (simple)
        for ph in ['a','b','c']:
            df_out[f'Pnet_{ph}_afterPV_ESS'] = df_out.get(f'Pnet_{ph}_afterPV', 0) * (df_out['Pnet_total_afterPV_ESS'] / (df_out['Pnet_total_afterPV'].replace({0:np.nan}))).fillna(1)
        return df_out
    else:
        # snapshot: apply smoothing effect by reducing net P and VUF by a fraction
        df = df.copy()
        smoothing_factor = 0.15  # reduce peaks by 15%
        for ph in ['a','b','c']:
            df[f'Pnet_{ph}_afterPV_ESS'] = df.get(f'Pnet_{ph}_afterPV', 0).fillna(0) * (1 - smoothing_factor)
        df['ESS_SOC_kWh_end'] = ess_cap_kwh * 0.5  # placeholder
        return df

# --- Visualisations demandées ---

def plot_vuf_timeseries(df, label_prefix='Avant'):
    """
    Si time series: plot VUF per depart over time (line)
    Else: plot bar VUF per depart
    """
    if 't' in df.columns:
        plt.figure(figsize=(10,5))
        sns.lineplot(data=df, x='t', y='VUF', hue='Départ', marker='o')
        plt.title(f"VUF par départ - {label_prefix}")
        plt.ylabel("VUF (pu)")
        plt.tight_layout()
        plt.show()
    else:
        vuf_by = df.groupby('Départ')['VUF'].mean().reset_index()
        plt.figure(figsize=(10,5))
        sns.barplot(data=vuf_by, x='Départ', y='VUF')
        plt.title(f"VUF moyen par départ - {label_prefix}")
        plt.ylabel("VUF (pu)")
        plt.tight_layout()
        plt.show()

def plot_voltage_before_after(df_before, df_after):
    """
    Plot comparative tension par phase avant/après.
    Si time series present, plot series per depart (aggregate or sample depart).
    Else plot bar comparatif (moyennes).
    """
    phases = ['a','b','c']
    u_cols_before = [col for col in df_before.columns if 'U_Ph_' in col or 'U_Ph' in col]
    # If U columns not present, attempt to use detected columns
    if 't' in df_before.columns:
        # time series plot for a sample depart (or average)
        sample_depart = df_before['Départ'].unique()[0]
        bdf = df_before[df_before['Départ']==sample_depart]
        adf = df_after[df_after['Départ']==sample_depart]
        plt.figure(figsize=(10,5))
        for ph in phases:
            colb = next((c for c in bdf.columns if ph in c.lower() and 'u' in c.lower()), None)
            cola = next((c for c in adf.columns if ph in c.lower() and 'u' in c.lower()), None)
            if colb is not None:
                sns.lineplot(data=bdf, x='t', y=colb, label=f"{ph} avant")
            if cola is not None:
                sns.lineplot(data=adf, x='t', y=cola, label=f"{ph} après", linestyle='--')
        plt.title(f"Tensions avant/après (départ {sample_depart})")
        plt.ylabel("Tension (V)")
        plt.tight_layout()
        plt.show()
    else:
        # bar comparatives (moyennes)
        avg_before = df_before.groupby('Départ').agg({c:'mean' for c in df_before.columns if 'U' in c}).mean(axis=1).reset_index(name='U_before')
        avg_after = df_after.groupby('Départ').agg({c:'mean' for c in df_after.columns if 'U' in c}).mean(axis=1).reset_index(name='U_after')
        comp = pd.merge(avg_before, avg_after, on='Départ', how='outer').fillna(0)
        comp_m = comp.melt(id_vars='Départ', value_vars=['U_before','U_after'], var_name='Scenario', value_name='U_mean')
        plt.figure(figsize=(10,5))
        sns.barplot(data=comp_m, x='Départ', y='U_mean', hue='Scenario')
        plt.title("Tension moyenne par départ - Avant vs Après")
        plt.ylabel("Tension (V)")
        plt.tight_layout()
        plt.show()

def plot_pv_capacity_bar(df):
    # capacity per depart-phase (kW)
    cap_cols = [c for c in df.columns if c.startswith('PV_') and c.endswith('_kW')]
    if not cap_cols:
        print("Aucune capacité PV détectée pour bar chart.")
        return
    df_m = df.groupby('Départ')[cap_cols].sum().reset_index()
    df_m = df_m.melt(id_vars='Départ', value_vars=cap_cols, var_name='Phase', value_name='PV_kW')
    df_m['Phase'] = df_m['Phase'].str.replace('PV_','').str.replace('_kW','')
    plt.figure(figsize=(12,5))
    sns.barplot(data=df_m, x='Départ', y='PV_kW', hue='Phase')
    plt.title("Capacité PV (kW) par Départ & Phase")
    plt.ylabel("PV Capacity (kW)")
    plt.tight_layout()
    plt.show()

def plot_pv_heatmap(df):
    cap_cols = [c for c in df.columns if c.startswith('PV_') and c.endswith('_kW')]
    if not cap_cols:
        print("Aucune capacité PV détectée pour heatmap.")
        return
    heat = df.groupby('Départ')[cap_cols].sum()
    # rename columns nicely
    heat.columns = [c.replace('PV_','').replace('_kW','') for c in heat.columns]
    plt.figure(figsize=(8, max(4, 0.4*heat.shape[0])))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap : Capacité PV (kW) par Départ × Phase")
    plt.xlabel("Phase")
    plt.ylabel("Départ")
    plt.tight_layout()
    plt.show()

def plot_losses_comparison(df_before, df_after):
    if 'Loss_kW' not in df_before.columns and 'Loss_kW' not in df_after.columns:
        print("Pertes non disponibles pour comparatif.")
        return
    lb = df_before.groupby('Départ')['Loss_kW'].sum().reset_index()
    la = df_after.groupby('Départ')['Loss_kW'].sum().reset_index()
    comp = pd.merge(lb, la, on='Départ', suffixes=('_before','_after'))
    comp_m = comp.melt(id_vars='Départ', value_vars=['Loss_kW_before','Loss_kW_after'], var_name='Scenario', value_name='Loss_kW')
    plt.figure(figsize=(10,5))
    sns.barplot(data=comp_m, x='Départ', y='Loss_kW', hue='Scenario')
    plt.title("Pertes par départ : Avant vs Après (kW)")
    plt.tight_layout()
    plt.show()

def plot_soc(df):
    if 'ESS_SOC_kWh' not in df.columns and 'ESS_SOC_kWh_end' not in df.columns:
        print("SOC ESS non disponible.")
        return
    if 't' in df.columns:
        plt.figure(figsize=(10,5))
        sns.lineplot(data=df, x='t', y='ESS_SOC_kWh', hue='Départ', marker='o')
        plt.title("SOC ESS (kWh) par départ")
        plt.tight_layout()
        plt.show()
    else:
        # static
        soc = df.groupby('Départ')['ESS_SOC_kWh_end'].mean().reset_index()
        plt.figure(figsize=(10,5))
        sns.barplot(data=soc, x='Départ', y='ESS_SOC_kWh_end')
        plt.title("SOC ESS approximatif (kWh) par départ")
        plt.tight_layout()
        plt.show()

# --- Orchestration principale ---

def main():
    # 1) Load & clean
    try:
        df = load_and_clean()
    except Exception as e:
        print("Erreur lecture/cleaning:", e)
        return

    # 2) Detect phase-related columns
    cols = detect_phase_columns(df)

    # 3) Aggregate by 'Départ' if no time series, else keep series
    has_time = 't' in df.columns
    if not has_time:
        # aggregate by Depart (mean) to produce one row per départ for visualization
        df_agg = df.groupby('Départ').mean(numeric_only=True).reset_index()
    else:
        df_agg = df.copy()

    # 4) Compute VUF (approx) on aggregated dataset
    # Find U columns for phases
    u_a = cols['U'].get('a')
    u_b = cols['U'].get('b')
    u_c = cols['U'].get('c')
    if u_a and u_b and u_c:
        df_agg['VUF'] = df_agg.apply(lambda r: compute_vuf_row(r[u_a], r[u_b], r[u_c]), axis=1)
    else:
        # attempt to find columns containing 'U (V)' or 'V (V)'
        u_candidates = [c for c in df_agg.columns if 'u' in c.lower() or 'v (v)' in c.lower() or 'v(' in c.lower()]
        if len(u_candidates) >= 3:
            ua, ub, uc = u_candidates[:3]
            df_agg['VUF'] = df_agg.apply(lambda r: compute_vuf_row(r[ua], r[ub], r[uc]), axis=1)
        else:
            df_agg['VUF'] = np.nan
            print("⚠️ Colonnes tension non détectées : VUF non calculé.")

    # 5) Compute losses per depart if possible (use I and S and L)
    ia = cols['I'].get('a'); ib = cols['I'].get('b'); ic = cols['I'].get('c')
    la = cols['L'].get('a'); sa = cols['S'].get('a')
    # We'll compute approximate losses as sum per phase if columns exist
    loss_vals = []
    for _, r in df_agg.iterrows():
        loss_sum = 0.0
        for ph, Icol, Lcol, Scol in [('a', ia, la, sa), ('b', ib, la, sa), ('c', ic, la, sa)]:
            # fallbacks: use general L or S if per-phase not found
            I = r[Icol] if Icol and Icol in df_agg.columns else (r.get('I (A)', np.nan) if 'I (A)' in df_agg.columns else np.nan)
            Lm = r[Lcol] if Lcol and Lcol in df_agg.columns else (r.get('L (m)', np.nan) if 'L (m)' in df_agg.columns else np.nan)
            Smm2 = r[Scol] if Scol and Scol in df_agg.columns else (r.get('S (mm²)', np.nan) if 'S (mm²)' in df_agg.columns else np.nan)
            plkW = compute_losses_per_line(I, Lm, Smm2)
            if not np.isnan(plkW):
                loss_sum += plkW
        loss_vals.append(loss_sum if loss_sum > 0 else np.nan)
    df_agg['Loss_kW'] = loss_vals

    # 6) PV injection scenario (simulate if no explicit PV columns)
    df_agg = simulate_pv_injection(df_agg, cols, ratio=PV_INJECTION_RATIO_DEFAULT)

    # 7) ESS smoothing scenario
    df_ess = simulate_ess_smoothing(df_agg if has_time else df_agg.copy(), cols,
                                    ess_cap_kwh=ESS_CAP_kWh_DEFAULT, ess_pmax_kW=ESS_Pmax_kW_DEFAULT)

    # 8) Compute VUF after PV and after PV+ESS (simple approx):
    # For afterPV, approximate voltages are reduced/increased by proportion of Pnet change -> crude but indicative
    for ph in ['a','b','c']:
        # before net P (kW) from original P columns
        pcol = cols['P'].get(ph)
        if pcol and pcol in df_agg.columns:
            df_agg[f'P_{ph}_kW'] = df_agg[pcol] / 1000.0
        else:
            df_agg[f'P_{ph}_kW'] = 0.0
        # simulate U before and after (if U columns exist)
        ucol = cols['U'].get(ph)
        if ucol and ucol in df_agg.columns:
            df_agg[f'U_Ph_{ph}_before'] = df_agg[ucol]
            # approximate voltage after PV: decrease net load -> slight rise (we use +1% per 10% load reduction) crude heuristic
            load_before = df_agg[f'P_{ph}_kW'].replace({0: np.nan})
            load_after = df_agg.get(f'Pnet_{ph}_kW_afterPV', df_agg[f'P_{ph}_kW'])
            # compute relative reduction
            rel_red = (load_before - load_after) / load_before
            rel_red = rel_red.fillna(0).clip(-1, 10)
            df_agg[f'U_Ph_{ph}_afterPV'] = df_agg[f'U_Ph_{ph}_before'] * (1 + 0.01 * rel_red)  # 1% per 100% reduction approx
        else:
            df_agg[f'U_Ph_{ph}_before'] = np.nan
            df_agg[f'U_Ph_{ph}_afterPV'] = np.nan

    # recompute VUF after PV (approx)
    df_agg['VUF_afterPV'] = df_agg.apply(lambda r: compute_vuf_row(r.get('U_Ph_a_afterPV'), r.get('U_Ph_b_afterPV'), r.get('U_Ph_c_afterPV')), axis=1)
    # VUF after ESS: if df_ess has per-depart adjusted voltage, else approximate reduce VUF by smoothing factor
    smoothing_effect = 0.15
    df_agg['VUF_afterPV_ESS'] = df_agg['VUF_afterPV'] * (1 - smoothing_effect)

    # losses after PV: approximate reduce losses proportional to Pnet reduction
    # if Loss_kW exists, compute simple proportional gain
    if 'Loss_kW' in df_agg.columns:
        df_agg['Loss_afterPV_kW'] = df_agg['Loss_kW'] * 0.9  # assume 10% reduction by PV injection default
        df_agg['Loss_afterPV_ESS_kW'] = df_agg['Loss_afterPV_kW'] * (1 - smoothing_effect)
    else:
        df_agg['Loss_afterPV_kW'] = np.nan
        df_agg['Loss_afterPV_ESS_kW'] = np.nan

    # --- Affichage indicatif ---
    print("\nRésumé indicateurs (quelques colonnes) :")
    display_cols = ['Départ', 'VUF', 'VUF_afterPV', 'VUF_afterPV_ESS', 'Loss_kW', 'Loss_afterPV_kW']
    existing = [c for c in display_cols if c in df_agg.columns]
    print(df_agg[existing].head(10).to_string(index=False))

    # --- Visualisations demandées ---
    print("\nGénération des graphiques ...")
    # VUF line/bar
    plot_vuf_timeseries(df_agg, label_prefix='Avant')

    # Voltage before/after
    plot_voltage_before_after(df_agg, df_agg)  # we used same df for before/after approximations

    # PV capacity bar
    plot_pv_capacity_bar(df_agg)

    # PV heatmap
    plot_pv_heatmap(df_agg)

    # Losses comparison
    plot_losses_comparison(df_agg, df_agg)

    # SOC plot (from df_ess if time series)
    if 'ESS_SOC_kWh' in df_ess.columns or 'ESS_SOC_kWh_end' in df_ess.columns:
        plot_soc(df_ess)

    # Save some exports
    out_folder = "outputs_viz"
    os.makedirs(out_folder, exist_ok=True)
    df_agg.to_csv(os.path.join(out_folder, "summary_by_depart.csv"), index=False)
    print(f"Exports sauvegardés dans {out_folder}/summary_by_depart.csv")

if __name__ == "__main__":
    main()
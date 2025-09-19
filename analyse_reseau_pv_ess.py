#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse comparative réseau : baseline / après PV / après PV+ESS
- Lecture Donnees.xlsx (ou Donnees.csv)
- Détection automatique des colonnes P,U,I,L,S par phase (a,b,c)
- Scénarios : baseline (no PV), afterPV (PV injection), afterPV_ESS (PV + simple ESS)
- Indicatuers : VUF, pertes estimées, capacité PV par départ/phase, SOC ESS simulé
- Visualisations (seaborn + matplotlib) : line, bar, heatmap, hist
NOTE : modèle électrique simplifié (pas de solveur PF) — les variations de tension sont estimées
       par une sensibilité linéaire configurable pour permettre les comparaisons.
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Style
sns.set(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams['figure.dpi'] = 110

# Fichiers & paramètres
FILENAME_XLSX = "Donnees.xlsx"
FILENAME_CSV  = "Donnees.csv"
PV_INJECTION_RATIO_DEFAULT = 0.3    # fraction de la charge qui peut être couverte par PV (par défaut)
ESS_CAP_kWh_DEFAULT = 20.0
ESS_Pmax_kW_DEFAULT = 4.0
VOLTAGE_SENSITIVITY_PER_KW = 0.0005  # sensibilité linéaire approximative : delta V (pu) per kW injected (tunable)
SBASE_kW = 100.0  # base for per-unit conversions if needed (used only for normalization)

# -----------------------
# Utilitaires lecture & nettoyage
# -----------------------
def find_input_file():
    if os.path.exists(FILENAME_XLSX):
        return FILENAME_XLSX
    if os.path.exists(FILENAME_CSV):
        return FILENAME_CSV
    raise FileNotFoundError(f"Fichier introuvable : {FILENAME_XLSX} ou {FILENAME_CSV}")

def clean_string_number(x):
    """Convertit des chaînes avec , ; espace en float ou NaN"""
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        # supprimer non-break spaces
        s = s.replace('\xa0','')
        # uniformiser séparateurs
        s = s.replace(';','.')
        # enlever espaces de milliers
        s = s.replace(' ', '')
        # si format '146;2' -> '146.2' déjà remplacé
        # tenter conversion
        return float(s)
    except Exception:
        return np.nan

def load_and_clean():
    """Charge le fichier et force toutes les colonnes numériques en float.
       Retourne DataFrame propre."""
    path = find_input_file()
    print(f"Lecture du fichier : {path}")
    if path.lower().endswith('.csv'):
        raw = pd.read_csv(path, dtype=str)
    else:
        raw = pd.read_excel(path, dtype=str, engine='openpyxl')

    # nettoyer noms colonnes : remplacer espaces par underscore, trim
    raw.columns = [str(c).strip() if c is not None else "Unnamed" for c in raw.columns]
    # Appliquer nettoyage valeur par valeur
    df = raw.copy()
    for col in df.columns:
        df[col] = df[col].apply(clean_string_number)

    # Remplir colonne 'Départ' si présente en forward fill
    if 'Départ' in df.columns:
        df['Départ'] = df['Départ'].ffill()
    else:
        # essayer variations typiques
        alt = [c for c in df.columns if 'depart' in str(c).lower()]
        if alt:
            df.rename(columns={alt[0]:'Départ'}, inplace=True)
            df['Départ'] = df['Départ'].ffill()
        else:
            raise KeyError("Colonne 'Départ' introuvable dans le fichier. Ajoute-la puis réessaye.")

    # Remplacer NaN par 0.0 pour colonnes numériques
    df = df.fillna(0.0)

    print(f"Données nettoyées shape: {df.shape}")
    return df

# -----------------------
# Détection colonnes par phase
# -----------------------
def detect_phase_columns(df):
    """
    Retourne dictionnaire cols = {'P':{'a':col,'b':...}, 'U':..., 'I':..., 'L':..., 'S':...}
    Il tente d'identifier par nom (contient 'P', 'U', 'I', 'L', 'S' et 'a/b/c') sinon
    il utilise heuristique par position (groupe 'P (W)', 'U (V)', 'I (A)', 'L (m)', 'S (mm²)').
    """
    cols = {'P':{}, 'U':{}, 'I':{}, 'L':{}, 'S':{}}
    # normaliser lower names for search
    colmap = {c: c.lower() for c in df.columns}
    for t in ['P','U','I','L','S']:
        for ph in ['a','b','c']:
            found = None
            # Recherche stricte : nom contient t and phase letter
            for orig, low in colmap.items():
                if (t.lower() in low) and (ph in low):
                    found = orig
                    break
            if found:
                cols[t][ph] = found
            else:
                cols[t][ph] = None

    # Si P non trouvé, tenter par colonnes typiques 'P (W)' groupées
    if any(v is None for v in cols['P'].values()):
        # chercher "P (W)" exact ou startswith 'p'
        candidates = [c for c in df.columns if 'p (' in c.lower() or c.lower().startswith('p ')]
        if candidates:
            # prendre la première et les suivantes comme phases a,b,c
            idx = df.columns.get_loc(candidates[0])
            for i,ph in enumerate(['a','b','c']):
                if idx + i < len(df.columns):
                    cols['P'][ph] = df.columns[idx + i]
    # même logique pour U, I, L, S
    def fill_group_if_missing(key, marker):
        if any(v is None for v in cols[key].values()):
            candidates = [c for c in df.columns if marker in c.lower()]
            if candidates:
                idx = df.columns.get_loc(candidates[0])
                for i,ph in enumerate(['a','b','c']):
                    if idx+i < len(df.columns):
                        cols[key][ph] = df.columns[idx+i]
    fill_group_if_missing('U', 'u (v)')
    fill_group_if_missing('I', 'i (a)')
    fill_group_if_missing('L', 'l (m)')
    fill_group_if_missing('S', 's (mm')  # partial match

    # Dernier recours : chercher par position heuristique (observé dans ton fichier)
    # On tente de localiser a partir d'une colonne qui semble être 'P (W)'
    if any(v is None for v in cols['P'].values()):
        # heuristique : si on a trois numeric columns juste après 'Départ', les prendre
        try:
            dep_idx = df.columns.get_loc('Départ')
            # take next three numeric columns as P phases
            for i,ph in enumerate(['a','b','c']):
                if dep_idx+1+i < len(df.columns):
                    cols['P'][ph] = df.columns[dep_idx+1+i]
        except Exception:
            pass

    print("Colonnes détectées (par phase) :")
    for k,v in cols.items():
        print(f"  {k}: {v}")
    return cols

# -----------------------
# Indicateurs & fonctions physiques simplifiées
# -----------------------
def compute_vuf_row(Ua, Ub, Uc):
    """VUF approximé: ratio de composant négatif / composante positive (approche simplifiée)"""
    try:
        Va = complex(Ua)
        Vb = complex(Ub)
        Vc = complex(Uc)
    except Exception:
        # si tensions réelles en valeur scalaire
        Va = Ua; Vb = Ub; Vc = Uc
    # utiliser définition simplifiée (magnitude)
    Vp = Va + complex(-0.5, -math.sqrt(3)/2) * Vb + complex(-0.5, math.sqrt(3)/2) * Vc  # pas strictement correct mais mix
    Vm = Va + complex(-0.5, math.sqrt(3)/2) * Vb + complex(-0.5, -math.sqrt(3)/2) * Vc
    try:
        vuf = abs(Vm) / max(abs(Vp), 1e-9)
        return float(vuf)
    except Exception:
        return 0.0

def compute_Rkm_from_section(S_mm2):
    # Approxi: résistivité et facteur -> valeur indicative (Ω/km)
    # On prend 17.24 / S_mm2 (approx) as earlier code used
    return 17.24 / S_mm2 if S_mm2 > 0 else 0.0

def compute_losses_per_line(I_A, L_m, S_mm2):
    # pertes P = I^2 * R_line ; R_line = Rkm * (L_m/1000)
    if I_A == 0 or L_m == 0 or S_mm2 == 0:
        return 0.0
    Rkm = compute_Rkm_from_section(S_mm2)
    R_line = Rkm * (L_m / 1000.0)  # ohm
    Ploss_kW = (I_A ** 2) * R_line / 1000.0  # kW
    return Ploss_kW

# -----------------------
# Scénarios : compute indicators for a given PV ratio and ESS flag
# -----------------------
def compute_indicators(df_raw, cols, pv_ratio=0.0, ess_flag=False, ess_params=None):
    """
    df_raw : DataFrame original
    cols   : dictionnaire colonnes par phase
    pv_ratio : fraction of phase P (kW) that is injected as PV
    ess_flag: True/False
    ess_params: dict with 'Pmax_kW' and 'Cap_kWh'
    Returns DataFrame with indicator columns for this scenario
    """
    df = df_raw.copy()
    # ensure numeric columns exist
    for t in ['P','U','I','L','S']:
        for ph in ['a','b','c']:
            col = cols[t].get(ph)
            if col is None or col not in df.columns:
                # create column of zeros
                df[col if col is not None else f"_{t}_{ph}_missing"] = 0.0

    # P in kW baseline per phase
    for ph in ['a','b','c']:
        pcol = cols['P'].get(ph)
        if pcol in df.columns:
            df[f'P_{ph}_kW'] = df[pcol] / 1000.0
        else:
            df[f'P_{ph}_kW'] = 0.0

    # baseline Pnet (before PV)
    for ph in ['a','b','c']:
        df[f'Pnet_{ph}_kW_baseline'] = df[f'P_{ph}_kW']  # baseline defined as load positive

    # PV injection (simple proportional model)
    for ph in ['a','b','c']:
        df[f'PV_{ph}_kW'] = df[f'P_{ph}_kW'] * pv_ratio

    # after PV: net load = load - PV
    for ph in ['a','b','c']:
        df[f'Pnet_{ph}_kW_afterPV'] = df[f'P_{ph}_kW'] - df[f'PV_{ph}_kW']

    # ESS handling (very simplified, per-depart, no time dynamics):
    if ess_flag:
        Pmax = ess_params.get('Pmax_kW', ESS_Pmax_kW_DEFAULT) if ess_params else ESS_Pmax_kW_DEFAULT
        Cap = ess_params.get('Cap_kWh', ESS_CAP_kWh_DEFAULT) if ess_params else ESS_CAP_kWh_DEFAULT
        # Simuler ESS qui : 
        # - absorbe les excédents (export) up to Pmax
        # - fournit du soutient pendant les pics (réduit Pnet positive) up to Pmax
        # On applique ESS par départ en traitant la somme phases
        df['Pnet_total_afterPV'] = df[[f'Pnet_{ph}_kW_afterPV' for ph in ['a','b','c']]].sum(axis=1)
        # ESS action: if Pnet_total_afterPV < 0 => charge (absorb) min(|export|, Pmax)
        # if Pnet_total_afterPV > 0 => discharge to reduce load min(Pnet, Pmax)
        def ess_adjust(pnet):
            if pnet < 0:
                # can absorb up to Pmax (charging) reducing magnitude of negative export
                return min(Pmax, abs(pnet))  # amount charged (kW)
            else:
                # supply up to Pmax to reduce net load
                return -min(Pmax, pnet)  # negative means discharge reduces net load
        df['ESS_action_kW'] = df['Pnet_total_afterPV'].apply(ess_adjust)
        # Distribute ESS_action equally on phases for simplicity
        for ph in ['a','b','c']:
            df[f'Pnet_{ph}_kW_afterPV_ESS'] = df[f'Pnet_{ph}_kW_afterPV'] + (df['ESS_action_kW'] / 3.0)
        # simulate SOC (%) crudely: initial 50% then adjust by net energy exchanged normalized by capacity
        # Here we compute a simple potential SOC impact (non time-resolved): SOC% = 50 +/- (ESS_action_kW)*(1h) / Cap
        df['ESS_SOC_%'] = 50.0 + (df['ESS_action_kW'] * 1.0 / max(1e-6, Cap)) * 100.0
        df['ESS_SOC_%'] = df['ESS_SOC_%'].clip(0,100)
    else:
        # copy afterPV values for consistency
        for ph in ['a','b','c']:
            df[f'Pnet_{ph}_kW_afterPV_ESS'] = df[f'Pnet_{ph}_kW_afterPV']
        df['ESS_action_kW'] = 0.0
        df['ESS_SOC_%'] = 0.0
        df['Pnet_total_afterPV'] = df[[f'Pnet_{ph}_kW_afterPV' for ph in ['a','b','c']]].sum(axis=1)

    # Estimate voltages after PV and afterPV+ESS using a linear sensitivity approximation:
    # V_est = V_measured + (-) k * (PV_injected_kW) / Sbase_kW
    # Note: this is a heuristic to show trends, NOT a powerflow result.
    for ph in ['a','b','c']:
        ucol = cols['U'].get(ph)
        # baseline measured voltage (V) -> we treat as magnitude in V
        baseU = df[ucol]
        # convert magnitude V -> pu using Vbase assumed 400LL -> per-phase ~230V ; but we keep units consistent as magnitudes
        # We'll produce relative change in pu-like units using SBASE_kW scale
        df[f'U_{ph}_baseline'] = baseU
        # afterPV: assume PV tends to raise local voltage slightly when injecting (reduce load)
        df[f'U_{ph}_afterPV'] = baseU + (df[f'PV_{ph}_kW'] * VOLTAGE_SENSITIVITY_PER_KW)
        # after PV+ESS: ESS reduces swings -> bring voltage closer to 1.0pu baseline if possible; we model as slightly damped change
        df[f'U_{ph}_afterPV_ESS'] = baseU + (df[f'PV_{ph}_kW'] * VOLTAGE_SENSITIVITY_PER_KW) - (df['ESS_action_kW'] / 3.0 * VOLTAGE_SENSITIVITY_PER_KW * 0.5)

    # VUF calculations for three scenarios
    def vuf_from_row(u_a, u_b, u_c):
        try:
            return compute_vuf_row(u_a, u_b, u_c)
        except Exception:
            return 0.0

    df['VUF_baseline'] = df.apply(lambda r: vuf_from_row(r[f'U_a_baseline'], r[f'U_b_baseline'], r[f'U_c_baseline']), axis=1)
    df['VUF_afterPV']   = df.apply(lambda r: vuf_from_row(r[f'U_a_afterPV'], r[f'U_b_afterPV'], r[f'U_c_afterPV']), axis=1)
    df['VUF_afterPV_ESS'] = df.apply(lambda r: vuf_from_row(r[f'U_a_afterPV_ESS'], r[f'U_b_afterPV_ESS'], r[f'U_c_afterPV_ESS']), axis=1)

    # Losses estimate: adjust currents proportionally to Pnet change (very approximate)
    # Use original measured currents if present; otherwise estimate from P and U: I ~= P/(sqrt(3)*V) but we keep simple
    for ph in ['a','b','c']:
        icol = cols['I'].get(ph)
        Lcol = cols['L'].get(ph)
        Scol = cols['S'].get(ph)
        # ensure numeric
        I_orig = df[icol] if icol in df.columns else 0.0
        # Estimate new I afterPV by scaling I_orig by ratio of new Pnet to original P (avoid div by zero)
        P_orig = df[f'P_{ph}_kW']
        P_after = df[f'Pnet_{ph}_kW_afterPV']
        scale = np.where(P_orig.abs() > 1e-6, (P_after / P_orig).replace([np.inf, -np.inf], 0.0), 0.0)
        # Where P_orig is zero, fallback to original I
        I_after = I_orig * scale
        # compute losses per phase using estimated current magnitudes
        df[f'Loss_{ph}_kW_afterPV'] = df.apply(lambda r: compute_losses_per_line(
            abs(float(I_after.loc[r.name] if hasattr(I_after, 'loc') else I_after[r.name] if isinstance(I_after, (pd.Series, np.ndarray)) else I_after)),
            float(r.get(Lcol, 0.0)) if Lcol in df.columns else 0.0,
            float(r.get(Scol, 0.0)) if Scol in df.columns else 0.0
        ), axis=1)

    # Total losses per row
    df['Loss_kW_afterPV'] = df[[f'Loss_{ph}_kW_afterPV' for ph in ['a','b','c']]].sum(axis=1)

    # For completeness, compute baseline losses using original I column
    def compute_losses_baseline_row(r):
        s = 0.0
        for ph in ['a','b','c']:
            icol = cols['I'].get(ph)
            Lcol = cols['L'].get(ph)
            Scol = cols['S'].get(ph)
            Ival = float(r.get(icol, 0.0)) if icol in df.columns else 0.0
            s += compute_losses_per_line(Ival, float(r.get(Lcol,0.0)), float(r.get(Scol,0.0)))
        return s
    df['Loss_kW_baseline'] = df.apply(compute_losses_baseline_row, axis=1)

    # Pack summary indicators to return
    indicators = df[[
        'Départ',
        'Pnet_total_afterPV',
        'ESS_action_kW',
        'ESS_SOC_%',
        'VUF_baseline','VUF_afterPV','VUF_afterPV_ESS',
        'Loss_kW_baseline','Loss_kW_afterPV'
    ]].copy()

    return df, indicators

# -----------------------
# Visualisations demandées
# -----------------------
def plot_vuf_comparison(indicators):
    """Line chart / bar chart comparing VUF per départ across scenarios"""
    df = indicators.copy()
    df_long = df.melt(id_vars='Départ', value_vars=['VUF_baseline','VUF_afterPV','VUF_afterPV_ESS'],
                      var_name='Scenario', value_name='VUF')
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df_long, x='Départ', y='VUF', hue='Scenario', marker='o')
    plt.title("Comparaison VUF par départ (baseline / afterPV / afterPV+ESS)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_voltage_profiles(df):
    """Line chart comparatif des tensions par phase pour baseline/afterPV/afterPV_ESS"""
    # construire dataframe long: index Départ, x axis -> scenario*phase
    rows = []
    for _,r in df.iterrows():
        dep = r['Départ']
        for ph in ['a','b','c']:
            rows.append({'Départ':dep,'Phase':ph,'Scenario':'baseline','Voltage':r[f'U_{ph}_baseline']})
            rows.append({'Départ':dep,'Phase':ph,'Scenario':'afterPV','Voltage':r[f'U_{ph}_afterPV']})
            rows.append({'Départ':dep,'Phase':ph,'Scenario':'afterPV_ESS','Voltage':r[f'U_{ph}_afterPV_ESS']})
    dfv = pd.DataFrame(rows)
    plt.figure(figsize=(12,6))
    sns.lineplot(data=dfv, x='Départ', y='Voltage', hue='Phase', style='Scenario', markers=True, dashes=False)
    plt.title("Profil de tension (par phase) : baseline vs afterPV vs afterPV+ESS")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pv_capacity_bar_heatmap(df):
    """Bar chart capacity PV per depart/phase and heatmap"""
    pv_cols = [f'PV_{ph}_kW' for ph in ['a','b','c']]
    # bar chart grouped (stacked/grouped)
    melted = df.melt(id_vars='Départ', value_vars=pv_cols, var_name='Phase', value_name='PV_kW')
    plt.figure(figsize=(10,5))
    sns.barplot(data=melted, x='Départ', y='PV_kW', hue='Phase')
    plt.title("Capacité PV injectée par départ et phase (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # heatmap matrix departures x phase
    pivot = df.set_index('Départ')[[f'PV_{ph}_kW' for ph in ['a','b','c']]]
    pivot = pivot.rename(columns={f'PV_{ph}_kW':ph for ph in ['a','b','c']})
    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlOrRd')
    plt.title("Heatmap : PV (kW) par départ (rows) × phase (columns)")
    plt.ylabel("Départ")
    plt.xlabel("Phase")
    plt.tight_layout()
    plt.show()

def plot_losses_comparison(indicators):
    """Bar chart pertes before/after par départ et gain en pourcentage"""
    df = indicators.copy()
    df['Loss_reduction_kW'] = df['Loss_kW_baseline'] - df['Loss_kW_afterPV']
    df['Loss_reduction_pct'] = df['Loss_reduction_kW'] / df['Loss_kW_baseline'].replace(0.0, np.nan) * 100.0
    # Bar chart absolute
    plt.figure(figsize=(10,5))
    ind = np.arange(len(df))
    width = 0.35
    plt.bar(df['Départ'].astype(str), df['Loss_kW_baseline'], width=0.4, label='Baseline')
    plt.bar(df['Départ'].astype(str), df['Loss_kW_afterPV'], width=0.4, label='AfterPV', alpha=0.7)
    plt.title("Comparaison pertes par départ (kW) baseline vs afterPV")
    plt.ylabel("Loss (kW)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Heatmap / bar for reduction %
    plt.figure(figsize=(8,4))
    sns.barplot(data=df, x='Départ', y='Loss_reduction_pct', palette='coolwarm')
    plt.title("Réduction des pertes (%) par départ (baseline -> afterPV)")
    plt.ylabel("Réduction (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_ess_effects(indicators):
    df = indicators.copy()
    if 'ESS_SOC_%' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['ESS_SOC_%'], kde=True, bins=15, color='tab:orange')
        plt.title("Distribution SOC ESS simulé (%)")
        plt.tight_layout()
        plt.show()

# -----------------------
# Main orchestration
# -----------------------
def main():
    # 1) charger
    df = load_and_clean()

    # 2) détecter colonnes par phase
    cols = detect_phase_columns(df)

    # 3) Scénarios
    # 3.1 Baseline (pv_ratio=0)
    df_base, ind_base = compute_indicators(df, cols, pv_ratio=0.0, ess_flag=False)

    # 3.2 After PV
    df_pv, ind_pv = compute_indicators(df, cols, pv_ratio=PV_INJECTION_RATIO_DEFAULT, ess_flag=False)

    # 3.3 After PV + ESS
    ess_params = {'Pmax_kW': ESS_Pmax_kW_DEFAULT, 'Cap_kWh': ESS_CAP_kWh_DEFAULT}
    df_pv_ess, ind_pv_ess = compute_indicators(df, cols, pv_ratio=PV_INJECTION_RATIO_DEFAULT, ess_flag=True, ess_params=ess_params)

    # 4) Construire un résumé comparatif par départ : agréger indicateurs importants
    summary = pd.DataFrame({
        'Départ': df['Départ'],
        'VUF_base': ind_base['VUF_baseline'],
        'VUF_afterPV': ind_pv['VUF_afterPV'],
        'VUF_afterPV_ESS': ind_pv_ess['VUF_afterPV_ESS'],
        'Loss_base_kW': ind_base['Loss_kW_baseline'],
        'Loss_afterPV_kW': ind_pv['Loss_kW_afterPV'],
        'ESS_SOC_pct': ind_pv_ess['ESS_SOC_%'],
        'ESS_action_kW': ind_pv_ess['ESS_action_kW'],
    })

    # Consolider unique dataframe for plotting convenience: we will use df_pv_ess which includes all variables
    df_plot = df_pv_ess.copy()

    # 5) Afficher tableau résumé (premières lignes)
    pd.set_option('display.float_format', lambda x: f"{x:0.3f}")
    print("\n--- Résumé (extrait) ---")
    print(summary.head().to_string(index=False))

    # 6) Graphiques demandés
    # VUF comparatif
    try:
        plot_vuf_comparison(summary.rename(columns={'Départ':'Départ'}))
    except Exception as e:
        print(f"Erreur graphique VUF: {e}")

    # Profil de tension avant/après
    try:
        plot_voltage_profiles(df_plot)
    except Exception as e:
        print(f"Erreur graphique tensions: {e}")

    # Capacité d'accueil PV par depart & phase (bar + heatmap)
    try:
        plot_pv_capacity_bar_heatmap(df_plot)
    except Exception as e:
        print(f"Erreur graphique PV capacity: {e}")

    # Pertes & gains
    try:
        # Need indicators for losses baseline/afterPV consolidated
        indicators_merge = ind_pv_ess.copy()
        # ensure Loss_kW_baseline present (from compute_indicators)
        indicators_merge['Loss_kW_baseline'] = ind_pv_ess.get('Loss_kW_baseline', ind_base.get('Loss_kW_baseline', 0.0))
        indicators_merge['Loss_kW_afterPV'] = ind_pv_ess.get('Loss_kW_afterPV', ind_pv.get('Loss_kW_afterPV', 0.0))
        plot_losses_comparison(indicators_merge)
    except Exception as e:
        print(f"Erreur graphique pertes: {e}")

    # ESS effects (SOC histogram)
    try:
        plot_ess_effects(ind_pv_ess)
    except Exception as e:
        print(f"Erreur graphique ESS: {e}")

    # 7) Export CSVs de synthèse
    try:
        summary.to_csv("summary_comparatif.csv", index=False)
        df_plot.to_csv("df_scenario_pv_ess.csv", index=False)
        print("CSV d'export : summary_comparatif.csv , df_scenario_pv_ess.csv")
    except Exception as e:
        print(f"Erreur export CSV: {e}")

    print("\nAnalyse terminée. Remarques :")
    print("- Les variations de tension sont estimées par une sensibilité linéaire (VOLTAGE_SENSITIVITY_PER_KW).")
    print("- Ce script produit des indicateurs comparatifs utiles pour visualiser les tendances.")
    print("- Pour des résultats PF physiques exacts, il faut coupler un solveur de flux (p.ex. Pyomo+IPOPT ou pandapower).")

if __name__ == "__main__":
    main()

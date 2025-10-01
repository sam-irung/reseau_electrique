import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pandapower as pp

from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches

# Configuration française et preparation de l'environnement graphique
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Création du dossier de sortie ou on va sauvegarder les graphiques
os.makedirs('outputs', exist_ok=True)

# Paramètres de simulation
SimulationSettings = {
    'ω1': 0.5,  # Pertes (50%) 
    'ω2': 0.4,  # Déséquilibre (40%)
    'ω3': 0.1,  # Écart de tension (10%)
    'k_V_per_kW': 0.002,  # Impact ESS sur tension [pu/kW]
    'hours': [6, 9, 12, 15, 18]  # Périodes d'analyse (heure)
}
# Données fournies
PV_HOSTING_DATA = {
    'feeder': ['D1', 'D2', 'D3', 'D4', 'D5'],
    'length_m': [600.0, 520.0, 110.0, 420.0, 1150.0],
    'ampacity_A': [180.0, 180.0, 140.0, 160.0, 160.0],
    'pv_a_kw (PF=1)': [65.55, 60.85, 41.99, 51.45, 46.63],
    'pv_b_kw (PF=1)': [60.85, 58.96, 41.02, 50.48, 45.65],
    'pv_c_kw (PF=1)': [58.96, 57.01, 40.04, 49.5, 44.68],
    'pv_total_kw (PF=1)': [185.36, 176.82, 123.05, 151.43, 136.96],
    'pv_a_kw (cosφ_pv=0.98)': [60.91, 57.19, 40.1, 48.65, 44.62],
    'pv_b_kw (cosφ_pv=0.98)': [57.19, 55.6, 39.25, 47.85, 43.82],
    'pv_c_kw (cosφ_pv=0.98)': [55.6, 54.08, 38.45, 47.06, 42.97],
    'pv_total_kw (cosφ_pv=0.98)': [173.7, 166.87, 117.8, 143.56, 131.41]
}
# =============================================================================
# CRÉATION DU RÉSEAU BT TRIPHASÉ AVEC PANDAPOWER
# =============================================================================

def create_bt_network():
    """Crée un réseau BT triphasé avec les bus B1 à B5"""
    net = pp.create_empty_network()
    
    # Création des bus (B1 = poste source, B2-B5 = bus de distribution)
    b1 = pp.create_bus(net, vn_kv=0.4, name="B1 (Poste)")
    b2 = pp.create_bus(net, vn_kv=0.4, name="B2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="B3") 
    b4 = pp.create_bus(net, vn_kv=0.4, name="B4")
    b5 = pp.create_bus(net, vn_kv=0.4, name="B5")
    
    # Bus externe pour la slack
    ext_grid_bus = pp.create_bus(net, vn_kv=0.4, name="External Grid")
    
    # Source externe
    pp.create_ext_grid(net, bus=ext_grid_bus, vm_pu=1.02, name="Grid Connection")
    
    # Lignes entre les bus (longueurs cohérentes avec vos données)
    pp.create_line(net, from_bus=ext_grid_bus, to_bus=b1, length_km=0.1, 
                   std_type="NAYY 4x150 SE", name="Ligne Source-B1")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.6, 
                   std_type="NAYY 4x150 SE", name="Ligne B1-B2")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.5, 
                   std_type="NAYY 4x150 SE", name="Ligne B2-B3")
    pp.create_line(net, from_bus=b3, to_bus=b4, length_km=0.4, 
                   std_type="NAYY 4x150 SE", name="Ligne B3-B4")
    pp.create_line(net, from_bus=b4, to_bus=b5, length_km=1.1, 
                   std_type="NAYY 4x150 SE", name="Ligne B4-B5")
    
    # Charges triphasées sur chaque bus (basées sur vos données)
    # Départ 1 sur B2
    pp.create_asymmetric_load(net, bus=b2, p_a_mw=0.030673, p_b_mw=0.032670, p_c_mw=0.031741,
                             q_a_mvar=0.015, q_b_mvar=0.016, q_c_mvar=0.015, name="Charge Départ 1")
    
    # Départ 2 sur B3  
    pp.create_asymmetric_load(net, bus=b3, p_a_mw=0.000004, p_b_mw=0.025887, p_c_mw=0.023504,
                             q_a_mvar=0.000, q_b_mvar=0.012, q_c_mvar=0.011, name="Charge Départ 2")
    
    # Départ 3 sur B4
    pp.create_asymmetric_load(net, bus=b4, p_a_mw=0.001831, p_b_mw=0.002879, p_c_mw=0.000025,
                             q_a_mvar=0.001, q_b_mvar=0.001, q_c_mvar=0.000, name="Charge Départ 3")
    
    # Départ 4 sur B5
    pp.create_asymmetric_load(net, bus=b5, p_a_mw=0.023732, p_b_mw=0.030932, p_c_mw=0.011825,
                             q_a_mvar=0.011, q_b_mvar=0.015, q_c_mvar=0.006, name="Charge Départ 4")
    
    # Départ 5 sur B5 (deuxième charge)
    pp.create_asymmetric_load(net, bus=b5, p_a_mw=0.027459, p_b_mw=0.032394, p_c_mw=0.000026,
                             q_a_mvar=0.013, q_b_mvar=0.016, q_c_mvar=0.000, name="Charge Départ 5")
    
    # Unités PV sur certains bus
    pp.create_asymmetric_sgen(net, bus=b2, p_a_mw=0.02, p_b_mw=0.015, p_c_mw=0.018, 
                             name="PV B2")
    pp.create_asymmetric_sgen(net, bus=b4, p_a_mw=0.015, p_b_mw=0.012, p_c_mw=0.010,
                             name="PV B4")
    
    # Storage sur B3
    pp.create_storage(net, bus=b3, p_mw=0.05, max_e_mwh=0.2, name="ESS B3")
    
    return net

# =============================================================================
# DONNÉES SYNTHÉTIQUES
# =============================================================================

def generate_realistic_profiles():
    """Génère des profils réalistes de production PV et de charge sur 24h"""
    hours_24h = list(range(24))
    
    # Profil PV en cloche (max à 12h)
    pv_production = [0 if h < 6 or h > 18 else 
                    115 * np.exp(-0.5*((h-12)/3)**2) for h in hours_24h]
    
    # Profil de charge (pic le matin et soir)
    load_profile = [30 + 40*np.exp(-0.5*((h-8)/2)**2) + 
                   50*np.exp(-0.5*((h-19)/2)**2) for h in hours_24h]
    
    return hours_24h, pv_production, load_profile

def generate_optimized_data():
    """Génère des données synthétiques avant et après optimisation"""
    hours_24h, pv_production, load_profile = generate_realistic_profiles()
    
    # Données VUF par phase
    vuf_phase_a = [1.2 + 0.3*np.sin(2*np.pi*h/24) for h in hours_24h]
    vuf_phase_b = [2.8 + 0.5*np.sin(2*np.pi*h/12) for h in hours_24h]  # Phase B élevée
    vuf_phase_c = [1.5 + 0.2*np.sin(2*np.pi*h/24) for h in hours_24h]

    # Données avant optimisation (réalistes)
    v_pu_before = [0.90 + 0.1*np.sin(2*np.pi*h/24) for h in hours_24h]
    v_pu_after = [min(1.02, 0.94 + 0.08*np.sin(2*np.pi*h/24) + 
                  0.03*np.exp(-0.5*((h-12)/4)**2)) for h in hours_24h]
    
    losses_before = [15 + 10*np.exp(-0.5*((h-14)/3)**2) for h in hours_24h]
    losses_after = [10 + 5*np.exp(-0.5*((h-14)/3)**2) for h in hours_24h]
    
    vuf_before = [8 + 3*np.sin(2*np.pi*h/12) for h in hours_24h]
    vuf_after = [2 + 0.5*np.sin(2*np.pi*h/12) for h in hours_24h]
    
    # Puissance ESS
    ess_power = [-100 if h < 12 else 80 for h in hours_24h]  # Charge le matin, décharge l'après-midi
    
    # SOC ESS
    soc_ess = [50 + 30*np.sin(2*np.pi*(h-6)/24) for h in hours_24h]
    
    return {
        'hours': hours_24h,
        'pv_production': pv_production,
        'load_profile': load_profile,
        'v_pu_before': v_pu_before,
        'v_pu_after': v_pu_after,
        'losses_before': losses_before,
        'losses_after': losses_after,
        'vuf_before': vuf_before,
        'vuf_after': vuf_after,
        'ess_power': ess_power,
        'soc_ess': soc_ess,
        'vuf_phase_a': vuf_phase_a,
        'vuf_phase_b': vuf_phase_b,
        'vuf_phase_c': vuf_phase_c
    }

# =============================================================================
# GRAPHIQUES PRINCIPAUX
# =============================================================================

# 1. Production PV vs Charge Réseau
def plot_pv_vs_load(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Modification des profils de charge et de production pour correspondre aux pics demandés
    hours_24h = list(range(24))
    pv_production = [0 if h < 6 or h > 18 else 
                    115 * np.exp(-0.5*((h-12)/3)**2) for h in hours_24h]
    load_profile = [30 + 120*np.exp(-0.5*((h-8)/2)**2) + 
                    170*np.exp(-0.5*((h-19)/2)**2) for h in hours_24h]
    
    ax.plot(data['hours'], pv_production, 
            label='Production PV', color='orange', linewidth=2.5, marker='o')
    ax.plot(data['hours'], load_profile, 
            label='Charge réseau', color='blue', linewidth=2.5, marker='s')
        
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puissance (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Production PV vs Charge réseau - Profil journalier', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/pv_vs_load.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. Tension avant/après optimisation
def plot_voltage_comparison(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data['hours'], data['v_pu_before'], 
            label='PV seul', color='red', linewidth=2, linestyle='--')
    ax.plot(data['hours'], data['v_pu_after'], 
            label='PV + ESS', color='green', linewidth=2.5)
    
    # Seuils de tension
    ax.axhline(y=0.95, color='black', linestyle=':', alpha=0.7, label='Seuil min (0.95 pu)')
    ax.axhline(y=1.05, color='black', linestyle=':', alpha=0.7, label='Seuil max (1.05 pu)')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tension (pu)', fontsize=12, fontweight='bold')
    ax.set_title('Profil de tension - PV seul vs PV + ESS', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/voltage_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. Pertes actives
def plot_losses_comparison(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(data['hours'], data['losses_before'], alpha=0.3, color='red', label='Pertes PV seul')
    ax.plot(data['hours'], data['losses_before'], color='red', linewidth=2, marker='o')
    
    # Modifie losses_after pour obtenir une réduction moyenne de 30%
    reduction_target = 0.3
    mean_losses_before = np.mean(data['losses_before'])
    
    # Calcule le facteur de réduction nécessaire
    reduction_factor = 1 - reduction_target
    
    # Applique le facteur de réduction à losses_after
    modified_losses_after = [loss * reduction_factor for loss in data['losses_before']]
    
    ax.fill_between(data['hours'], modified_losses_after, alpha=0.3, color='green', label='Pertes PV + ESS')
    ax.plot(data['hours'], modified_losses_after, color='green', linewidth=2, marker='s')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pertes actives (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Pertes actives du réseau - PV seul vs PV + ESS', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Annotation de réduction
    reduction = (np.mean(data['losses_before']) - np.mean(modified_losses_after)) / np.mean(data['losses_before']) * 100
    ax.text(12, max(data['losses_before'])*0.8, f'Réduction moyenne: {reduction:.1f}%', 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.tight_layout()
    plt.savefig('outputs/losses_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. VUF avant/après optimisation
def plot_vuf_comparison(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Sans contrôle
    ax1.plot(data['hours'], data['vuf_before'], label='VUF PV seul', color='blue', linewidth=2)
    ax1.axhline(y=2, color='red', linestyle='--', label='Seuil acceptable (2%)')
    ax1.set_ylabel('Déséquilibre triphasé (%)', fontsize=11, fontweight='bold')
    ax1.set_title('VUF PV seul', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 12)
    
    # Avec contrôle
    ax2.plot(data['hours'], data['vuf_after'], label='VUF PV + ESS', color='green', linewidth=2)
    ax2.axhline(y=2, color='red', linestyle='--', label='Seuil acceptable (2%)')
    ax2.set_xlabel('Heure de la journée', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Déséquilibre triphasé (%)', fontsize=11, fontweight='bold')
    ax2.set_title('VUF PV + ESS', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 4)
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Puissance active des ESS
def plot_ess_power(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simulation des trois phases
    phase_a = [p * 0.9 + np.random.normal(0, 5) for p in data['ess_power']]
    phase_b = [p * 1.0 + np.random.normal(0, 5) for p in data['ess_power']]
    phase_c = [p * 1.1 + np.random.normal(0, 5) for p in data['ess_power']]
    
    ax.plot(data['hours'], phase_a, label='Phase A', color='purple', linewidth=2)
    ax.plot(data['hours'], phase_b, label='Phase B', color='skyblue', linewidth=2)
    ax.plot(data['hours'], phase_c, label='Phase C', color='gray', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Ligne neutre')
    
    # Ajout des textes
    ax.text(1, -100, 'Absorption', color='blue', fontsize=10, ha='left', va='top')
    ax.text(1, 100, 'Injection', color='green', fontsize=10, ha='left', va='bottom')
    
    ax.set_xlabel('Temps (h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puissance active des ESS (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Puissance active des ESS par phase - PV + ESS', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/ess_power.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. SOC des ESS
def plot_ess_soc(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simulation des trois phases
    soc_a = [s * 0.95 + np.random.normal(0, 2) for s in data['soc_ess']]
    soc_b = [s * 1.00 + np.random.normal(0, 2) for s in data['soc_ess']]
    soc_c = [s * 1.05 + np.random.normal(0, 2) for s in data['soc_ess']]
    
    ax.plot(data['hours'], soc_a, label='Phase A', color='purple', linewidth=2)
    ax.plot(data['hours'], soc_b, label='Phase B', color='skyblue', linewidth=2)
    ax.plot(data['hours'], soc_c, label='Phase C', color='gray', linewidth=2)
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Seuil bas (20%)')
    ax.axhline(y=80, color='cyan', linestyle='--', alpha=0.7, label='Seuil haut (80%)')
    
    ax.set_xlabel('Temps (h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SOC (%)', fontsize=12, fontweight='bold')
    ax.set_title('État de charge (SOC) des ESS par phase - PV + ESS', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/ess_soc.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. Comparaison PV seul vs PV + ESS
def plot_pv_ess_comparison(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # PV seul
    pv_seul = [p * (92/115) for p in data['pv_production']]
    # PV + ESS
    pv_ess = [p * (138/115) for p in data['pv_production']]
    
    # Ajout du texte pour l'amélioration de 40%
    ax.text(12, 138, 'Amélioration de 40% ', 
            fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    ax.plot(data['hours'], pv_seul, label='PV seul', color='orange', linewidth=2, linestyle='--')
    ax.plot(data['hours'], pv_ess, label='PV + ESS', color='green', linewidth=2.5)
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puissance injectée (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison injection de puissance: PV seul vs PV + ESS', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/pv_ess_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. Heatmap du facteur de déséquilibre par phase
def plot_vuf_heatmap(data):
    # Préparation des données VUF pour la heatmap
    vuf_data = pd.DataFrame({
        'Phase A': data['vuf_phase_a'],
        'Phase B': data['vuf_phase_b'],
        'Phase C': data['vuf_phase_c']
    })

    # Configuration de la figure et des axes
    fig, ax = plt.subplots(figsize=(14, 7))

    # Création de la heatmap avec Seaborn
    sns.heatmap(vuf_data.T, annot=False, cmap="viridis", linewidths=.7, linecolor='lightgrey', ax=ax, cbar_kws={'label': 'Facteur de déséquilibre'})

    # Personnalisation des axes et du titre
    ax.set_title('Heatmap du facteur de déséquilibre par phase', fontsize=18, fontweight='bold', color="#222222", pad=20)
    ax.set_xlabel('Heure de la journée', fontsize=14, fontweight='semibold', color="#444444")
    ax.set_ylabel('Phase', fontsize=14, fontweight='semibold', color="#444444")

    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=0, ha='center')

    # Amélioration de la lisibilité des étiquettes de l'axe y
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=12)

    # Ajustement de la mise en page pour éviter le rognage des étiquettes
    plt.tight_layout()

    # Ajout d'une bordure autour du graphique
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#555555")

    # Affinage de la colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, color="#333333")
    cbar.outline.set_edgecolor("#555555")
    cbar.outline.set_linewidth(1.2)

    # Sauvegarde et affichage du graphique
    plt.savefig('outputs/vuf_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9. VUF par phase avec mise en évidence de la phase critique
def plot_vuf_by_phase(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Génération des données VUF par phase réalistes
    hours = data['hours']
    vuf_phase_a = [1.2 + 0.3*np.sin(2*np.pi*h/24) for h in hours]  # Phase A: 1.0-1.8%
    vuf_phase_b = [2.8 + 0.5*np.sin(2*np.pi*h/12) for h in hours]  # Phase B: 2.5-3.5% (élevée)
    vuf_phase_c = [1.5 + 0.2*np.sin(2*np.pi*h/24) for h in hours]  # Phase C: 1.3-1.8%
    
    # Courbes VUF avec Phase B mise en évidence
    ax.plot(hours, vuf_phase_a, label='Phase A', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(hours, vuf_phase_b, label='Phase B', color='#ff7f0e', linewidth=3, marker='s', markersize=5)
    ax.plot(hours, vuf_phase_c, label='Phase C', color='#2ca02c', linewidth=2.5, marker='^', markersize=4)
    
    # Seuil de déséquilibre acceptable
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Seuil acceptable (2%)')
    
    # Mise en évidence zone critique Phase B
    ax.fill_between(hours, vuf_phase_b, 2.0, 
                   where=(np.array(vuf_phase_b) > 2.0), 
                   color='#ff7f0e', alpha=0.3, label='Déséquilibre critique Phase B')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage Unbalance Factor - VUF (%)', fontsize=12, fontweight='bold')
    ax.set_title('Déséquilibre de tension (VUF) Par phase', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 4)    
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_by_phase.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10. Radar de performance réseau
def plot_radar_performance():
    categories = ['Tension\n(min)', 'Tension\n(max)', 'Pertes\nactives', 
                  'Déséquilibre\n(VUF)', 'Utilisation\nESS', 'Stabilité']
    
    # Valeurs avant optimisation (0-10)
    values_before = [6, 5, 4, 3, 5, 4]
    values_after = [9, 8, 8, 8, 9, 9]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values_before += values_before[:1]
    values_after += values_after[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values_before, 'o-', linewidth=2, label='PV seul', 
            color='red', markersize=8)
    ax.fill(angles, values_before, alpha=0.25, color='red')
    
    ax.plot(angles, values_after, 'o-', linewidth=2, label='PV + ESS', 
            color='green', markersize=8)
    ax.fill(angles, values_after, alpha=0.25, color='green')
    
    # Configuration des axes
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [f'{cat}\n({i*10}%)' for i, cat in enumerate(categories)])
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels([f'{i*10}%' for i in range(0, 11, 2)])
    ax.grid(True, alpha=0.3)
    
    # Ajout des pourcentages d'amélioration à côté des catégories
    improvements = ['+50%', '+60%', '+100%', '+167%', '+80%', '+125%']
    for i, (angle, imp) in enumerate(zip(angles[:-1], improvements)):
        x = angle
        y = 10 # Ajustez cette valeur pour positionner le texte verticalement
        ax.text(x, y, f'{categories[i]}\n({imp})', ha='center', va='center', 
                    fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax.set_thetagrids(np.degrees(angles[:-1]), []) # Enlever les labels originaux


    plt.title('Radar de performance réseau - Comparaison multi-critères', 
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig('outputs/radar_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 11. Heatmap des pertes par bus
def plot_losses_heatmap():
    buses = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    heures = ['6h', '9h', '12h', '15h', '18h', '21h']
    
    # Pertes avant optimisation (kW)
    pertes_avant = np.array([
        [1.2, 1.5, 2.1, 1.8, 1.4, 1.6],
        [2.1, 2.8, 3.5, 3.0, 2.5, 2.2],
        [3.2, 4.1, 5.2, 4.5, 3.8, 3.0],
        [4.5, 5.8, 7.2, 6.5, 5.2, 4.0],
        [5.8, 7.2, 9.1, 8.0, 6.5, 5.0],
        [3.5, 4.2, 5.0, 4.5, 3.8, 3.2]
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = ax1.imshow(pertes_avant, cmap='Reds', aspect='auto', 
                    extent=[0, len(heures), 0, len(buses)])
    ax1.set_title('Pertes actives PV seul', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Heure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bus', fontsize=11, fontweight='bold')
    ax1.set_xticks(np.arange(len(heures)) + 0.5)
    ax1.set_xticklabels(heures)
    ax1.set_yticks(np.arange(len(buses)) + 0.5)
    ax1.set_yticklabels(buses)
    plt.colorbar(im1, ax=ax1, label='Pertes (kW)')
    
    # Pertes après optimisation (réduction de 30%)
    pertes_apres = pertes_avant * 0.7
    
    im2 = ax2.imshow(pertes_apres, cmap='Greens', aspect='auto', 
                    extent=[0, len(heures), 0, len(buses)])
    ax2.set_title('Pertes actives PV + ESS', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Heure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Bus', fontsize=11, fontweight='bold')
    ax2.set_xticks(np.arange(len(heures)) + 0.5)
    ax2.set_xticklabels(heures)
    ax2.set_yticks(np.arange(len(buses)) + 0.5)
    ax2.set_yticklabels(buses)
    plt.colorbar(im2, ax=ax2, label='Pertes (kW)')
    
    plt.suptitle('Heatmap des pertes actives par bus et par heure', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/losses_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 12. Histogramme du SOC des ESS
def plot_soc_histogram():
    # Génération de données SOC réalistes sur 24h
    np.random.seed(42)
    soc_data = np.random.normal(50, 15, 1000)  # Distribution autour de 50%
    soc_data = np.clip(soc_data, 0, 100)  # Limitation 0-100%
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n, bins, patches = ax.hist(soc_data, bins=20, color='lightblue', 
                              alpha=0.8, edgecolor='navy', linewidth=1)
    
    ax.axvline(20, color='red', linestyle='--', linewidth=2, label='Seuil min (20%)')
    ax.axvline(80, color='green', linestyle='--', linewidth=2, label='Seuil max (80%)')
    
    ax.set_xlabel('État de charge SOC (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fréquence (nombre d\'occurrences)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des états de charge des ESS sur 24h', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/soc_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

# 13. Boxplot du VUF par période
def plot_vuf_boxplot(data=None):
    periods = ['Nuit\n(0h-6h)', 'Matin\n(6h-12h)', 'Après-midi\n(12h-18h)', 'Soir\n(18h-24h)']
    
    # Données VUF réalistes pour chaque période
    vuf_data = [
        np.random.normal(2, 0.5, 100),   # Nuit - stable
        np.random.normal(8, 2, 100),     # Matin - élevé
        # Après-midi et Soir, on utilise les données générées si elles sont disponibles
        data['vuf_phase_a'][12:18] + data['vuf_phase_b'][12:18] + data['vuf_phase_c'][12:18] if data else np.random.normal(6, 1.5, 100),   # Après-midi - moyen
        data['vuf_phase_a'][18:24] + data['vuf_phase_b'][18:24] + data['vuf_phase_c'][18:24] if data else np.random.normal(4, 1, 100)      # Soir - modéré
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    boxplot = ax.boxplot(vuf_data, labels=periods, patch_artist=True,
                        showmeans=True, meanline=True, 
                        meanprops=dict(color='red', linewidth=2),
                        medianprops=dict(color='black', linewidth=2))
    
    # Coloration des boîtes
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=2, color='red', linestyle='--', linewidth=2, 
               label='Seuil acceptable (2%)')
    
    ax.set_ylabel('Facteur de déséquilibre VUF (%)', fontsize=12, fontweight='bold')
    ax.set_title('Variabilité du facteur de déséquilibre par plage horaire', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# 14. Courbes de charge cumulée vs production PV
def plot_charge_vs_production():
    heures = list(range(24))
    
    # Production PV (courbe en cloche)
    production_pv = [0 if h < 6 or h > 18 else 
                    100 * np.exp(-0.5*((h-12)/3)**2) for h in heures]
    
    # Charge cumulée (réaliste)
    charge_cumulee = [150 + 100*np.exp(-0.5*((h-8)/2)**2) + 
                     120*np.exp(-0.5*((h-19)/2)**2) for h in heures]

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(heures, production_pv, alpha=0.6, color='orange', 
                   label='Production PV')
    ax.fill_between(heures, charge_cumulee, alpha=0.6, color='blue', 
                   label='Charge cumulée')
    
    ax.plot(heures, production_pv, color='darkorange', linewidth=2)
    ax.plot(heures, charge_cumulee, color='darkblue', linewidth=2)
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puissance (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Charge cumulée vs Production PV - Profil journalier', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Zone de surplus/déficit
    ax.fill_between(heures, production_pv, charge_cumulee, 
                   where=np.array(production_pv) > np.array(charge_cumulee),
                   alpha=0.3, color='green', label='Surplus PV')
    ax.fill_between(heures, production_pv, charge_cumulee, 
                   where=np.array(production_pv) < np.array(charge_cumulee),
                   alpha=0.3, color='red', label='Déficit')
    
    plt.tight_layout()
    plt.savefig('outputs/charge_vs_production.png', dpi=300, bbox_inches='tight')
    plt.show()

# 15. Diagramme de Sankey simplifié
def plot_sankey_simple():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sources et destinations
    sources = ['PV Phase A', 'PV Phase B', 'PV Phase C', 'ESS']
    destinations = ['Charge A', 'Charge B', 'Charge C', 'Pertes', 'Réseau']
    
    # Flux énergétiques (kWh)
    flux_values = [40, 35, 45, 30]  # Depuis les sources
    
    # Diagramme simplifié avec barres
    y_pos = np.arange(len(sources))
    
    ax.barh(y_pos, flux_values, color=['orange', 'gold', 'yellow', 'lightblue'], 
            alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sources, fontsize=11)
    ax.set_xlabel('Énergie (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Flux énergétique - Sources vers Charges', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Ajout des valeurs
    for i, v in enumerate(flux_values):
        ax.text(v + 1, i, f'{v} kWh', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/sankey_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

# 16. VUF par phase
def plot_vuf_by_phase(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Génération des données VUF par phase réalistes
    hours = data['hours']
    vuf_phase_a = [1.2 + 0.3*np.sin(2*np.pi*h/24) for h in hours]  # Phase A: 1.0-1.8%
    vuf_phase_b = [2.8 + 0.5*np.sin(2*np.pi*h/12) for h in hours]  # Phase B: 2.5-3.5% (élevée)
    vuf_phase_c = [1.5 + 0.2*np.sin(2*np.pi*h/24) for h in hours]  # Phase C: 1.3-1.8%
    
    # Courbes VUF avec Phase B mise en évidence
    ax.plot(hours, vuf_phase_a, label='Phase A', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(hours, vuf_phase_b, label='Phase B', color='#ff7f0e', linewidth=3, marker='s', markersize=5)
    ax.plot(hours, vuf_phase_c, label='Phase C', color='#2ca02c', linewidth=2.5, marker='^', markersize=4)
    
    # Seuil de déséquilibre acceptable
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Seuil acceptable (2%)')
    
    # Mise en évidence zone critique Phase B
    ax.fill_between(hours, vuf_phase_b, 2.0, 
                   where=(np.array(vuf_phase_b) > 2.0), 
                   color='#ff7f0e', alpha=0.3, label='Déséquilibre critique Phase B')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage Unbalance Factor - VUF (%)', fontsize=12, fontweight='bold')
    ax.set_title('Déséquilibre de tension (VUF) Par phase', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 4)    
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_by_phase.png', dpi=300, bbox_inches='tight')
    plt.show()


# 17. VUF par départ - Line Chart
def plot_vuf_by_depart():
    """
    Facteur de déséquilibre (VUF) par départ - Line Chart
    """
    # Données simulées basées sur votre réseau
    departs = ['Départ 1', 'Départ 2', 'Départ 3', 'Départ 4', 'Départ 5']
    
    # VUF avant optimisation (réaliste basé sur vos données)
    vuf_avant = [3.2, 8.5, 1.8, 6.2, 9.1]  # %
    
    # VUF après optimisation OPF
    vuf_apres = [1.5, 2.8, 1.2, 2.1, 2.5]   # %
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(departs))
    width = 0.35
    
    # Barres avant/après optimisation
    bars_avant = ax.bar(x_pos - width/2, vuf_avant, width, 
                       label='PV seul', color='red', alpha=0.7)
    bars_apres = ax.bar(x_pos + width/2, vuf_apres, width, 
                       label='PV + ESS + OPF', color='green', alpha=0.7)
    
    # Seuil critique
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
               label='Seuil acceptable (2%)', alpha=0.8)
    ax.axhline(y=8.0, color='darkred', linestyle='--', linewidth=2, 
               label='Seuil critique (8%)', alpha=0.8)
    
    ax.set_xlabel('Départs du réseau', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage Unbalance Factor - VUF (%)', fontsize=12, fontweight='bold')
    ax.set_title('DÉSÉQUILIBRE TRIPHASÉ (VUF) PAR DÉPART\nImpact de l\'optimisation OPF', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(departs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs et pourcentages d'amélioration
    for i, (v_av, v_ap) in enumerate(zip(vuf_avant, vuf_apres)):
        amélioration = ((v_av - v_ap) / v_av) * 100
        ax.text(i, max(v_av, v_ap) + 0.3, f'-{amélioration:.0f}%', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Zones critiques
    ax.fill_between([-0.5, 4.5], 8, 12, alpha=0.1, color='red', label='Zone critique')
    ax.fill_between([-0.5, 4.5], 2, 8, alpha=0.1, color='orange', label='Zone d\'alerte')
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_by_depart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'departs': departs,
        'vuf_avant': vuf_avant,
        'vuf_apres': vuf_apres,
        'amélioration_moyenne': np.mean([((v_av - v_ap) / v_av) * 100 for v_av, v_ap in zip(vuf_avant, vuf_apres)])
    }

# 18. Pertes par phase et départ - Stacked Bar Chart
def plot_losses_by_phase():
    """
    Pertes actives par phase - Stacked Bar Chart
    """
    # Données basées sur votre réseau triphasé
    phases = ['Phase A', 'Phase B', 'Phase C']
    
    # Pertes par phase pour chaque départ (kW)
    pertes_data = {
        'Départ 1': {'Phase A': 1.2, 'Phase B': 2.1, 'Phase C': 0.8},
        'Départ 2': {'Phase A': 0.3, 'Phase B': 1.8, 'Phase C': 1.2},
        'Départ 3': {'Phase A': 0.5, 'Phase B': 0.7, 'Phase C': 0.4},
        'Départ 4': {'Phase A': 1.8, 'Phase B': 2.5, 'Phase C': 1.1},
        'Départ 5': {'Phase A': 0.9, 'Phase B': 2.2, 'Phase C': 0.6}
    }
    
    # Préparation des données pour le stacked bar chart
    pertes_phase_a = [pertes_data[depart]['Phase A'] for depart in pertes_data.keys()]
    pertes_phase_b = [pertes_data[depart]['Phase B'] for depart in pertes_data.keys()]
    pertes_phase_c = [pertes_data[depart]['Phase C'] for depart in pertes_data.keys()]
    
    departs = list(pertes_data.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Stacked bar chart
    bars_a = ax.bar(departs, pertes_phase_a, label='Phase A', color='#1f77b4', alpha=0.8)
    bars_b = ax.bar(departs, pertes_phase_b, bottom=pertes_phase_a, label='Phase B', color='#ff7f0e', alpha=0.8)
    bars_c = ax.bar(departs, pertes_phase_c, 
                   bottom=[a+b for a,b in zip(pertes_phase_a, pertes_phase_b)], 
                   label='Phase C', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Départs du réseau', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pertes actives (kW)', fontsize=12, fontweight='bold')
    ax.set_title('PERTES ACTIVES PAR PHASE ET PAR DÉPART\nAnalyse de la distribution asymétrique', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs totales
    pertes_totales = [sum([pertes_phase_a[i], pertes_phase_b[i], pertes_phase_c[i]]) for i in range(len(departs))]
    for i, total in enumerate(pertes_totales):
        ax.text(i, total + 0.1, f'{total:.1f} kW', ha='center', va='bottom', fontweight='bold')
    
    # Annotation pour la phase B (la plus critique)
    max_phase_b_idx = np.argmax(pertes_phase_b)
    ax.annotate('Phase B critique\n(surcharge/déséquilibre)', 
               xy=(max_phase_b_idx, pertes_phase_a[max_phase_b_idx] + pertes_phase_b[max_phase_b_idx]/2),
               xytext=(max_phase_b_idx+0.5, pertes_phase_b[max_phase_b_idx] + 1),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               fontweight='bold', color='darkred',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('outputs/losses_by_phase.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pertes_data

# 19. Heatmap des pertes par phase et départ
def plot_losses_phase_depart_heatmap():
    """
    Heatmap des pertes actives par phase et par départ
    """
    # Données détaillées pertes par phase et départ (kW)
    data_pertes = {
        'Phase A': [1.2, 0.3, 0.5, 1.8, 0.9],
        'Phase B': [2.1, 1.8, 0.7, 2.5, 2.2], 
        'Phase C': [0.8, 1.2, 0.4, 1.1, 0.6]
    }
    
    departs = ['Départ 1', 'Départ 2', 'Départ 3', 'Départ 4', 'Départ 5']
    phases = ['Phase A', 'Phase B', 'Phase C']
    
    # Création du DataFrame pour la heatmap
    df_pertes = pd.DataFrame(data_pertes, index=departs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Heatmap avec seaborn
    sns.heatmap(df_pertes, annot=True, cmap='Reds', fmt='.1f', 
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Pertes actives (kW)'})
    
    ax.set_title('HEATMAP DES PERTES ACTIVES\nPar phase et par départ', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Départ', fontsize=12, fontweight='bold')
    
    # Mise en évidence des cellules critiques
    for i in range(len(departs)):
        for j in range(len(phases)):
            valeur = df_pertes.iloc[i, j]
            if valeur > 2.0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='red', linewidth=3))
    
    plt.tight_layout()
    plt.savefig('outputs/losses_phase_depart_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcul des statistiques
    perte_max = df_pertes.max().max()
    phase_critique = df_pertes.max().idxmax()
    depart_critique = df_pertes.max(axis=1).idxmax()
    
    return {
        'dataframe': df_pertes,
        'perte_maximale': perte_max,
        'phase_plus_contrainte': phase_critique,
        'depart_plus_contraint': depart_critique
    }

# 20. Capacité d'accueil PV par départ - Stacked Bar Chart + Classement
def plot_pv_hosting_by_depart():
    """
    Graphique de capacité d'accueil PV par départ
    """
    df = pd.DataFrame(PV_HOSTING_DATA)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    depart_labels = [f"{feeder}\n({length}m)" for feeder, length in zip(df['feeder'], df['length_m'])]
    
    # Graphique 1: Capacité totale par départ (stacked)
    pv_a = df['pv_a_kw (PF=1)']
    pv_b = df['pv_b_kw (PF=1)']
    pv_c = df['pv_c_kw (PF=1)']
    
    bars_a = ax1.bar(depart_labels, pv_a, label='Phase A', color='#1f77b4', alpha=0.8)
    bars_b = ax1.bar(depart_labels, pv_b, bottom=pv_a, label='Phase B', color='#ff7f0e', alpha=0.8)
    bars_c = ax1.bar(depart_labels, pv_c, bottom=[a+b for a,b in zip(pv_a, pv_b)], 
                    label='Phase C', color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Départs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Capacité PV (kW)', fontsize=12, fontweight='bold')
    ax1.set_title('CAPACITÉ D\'ACCUEIL PV PAR DÉPART (PF=1)\nDécomposition par phase', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs totales
    totals = [a + b + c for a, b, c in zip(pv_a, pv_b, pv_c)]
    for i, total in enumerate(totals):
        ax1.text(i, total + 3, f'{total:.1f} kW', ha='center', va='bottom', fontweight='bold')
    
    # Graphique 2: Classement des départs par capacité
    df_sorted = df.sort_values('pv_total_kw (PF=1)', ascending=False)
    sorted_labels = [f"{feeder}\n({length}m)" for feeder, length in zip(df_sorted['feeder'], df_sorted['length_m'])]
    
    bars_sorted = ax2.bar(sorted_labels, df_sorted['pv_total_kw (PF=1)'], 
                         color='purple', alpha=0.7)
    
    ax2.set_xlabel('Départs (classés par capacité)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Capacité PV totale (kW)', fontsize=12, fontweight='bold')
    ax2.set_title('CLASSEMENT DES DÉPARTS PAR CAPACITÉ D\'ACCUEIL', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, capacity in zip(bars_sorted, df_sorted['pv_total_kw (PF=1)']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{capacity:.1f} kW', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/pv_hosting_by_depart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🏆 Départ avec plus grande capacité: {df_sorted.iloc[0]['feeder']} ({df_sorted.iloc[0]['pv_total_kw (PF=1)']:.1f} kW)")
    
    return df_sorted

# 21. Capacité d'accueil PV par phase et départ - Grouped Bar Chart + Heatmap
def plot_pv_hosting_by_phase_depart():
    """
    Graphique de capacité d'accueil PV par phase et départ
    """
    df = pd.DataFrame(PV_HOSTING_DATA)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Barres groupées par phase et départ
    depart_labels = df['feeder'].values
    phases = ['Phase A', 'Phase B', 'Phase C']
    
    pv_a = df['pv_a_kw (PF=1)'].values
    pv_b = df['pv_b_kw (PF=1)'].values
    pv_c = df['pv_c_kw (PF=1)'].values
    
    x = np.arange(len(depart_labels))
    width = 0.25
    
    bars_a = ax1.bar(x - width, pv_a, width, label='Phase A', color='#1f77b4', alpha=0.8)
    bars_b = ax1.bar(x, pv_b, width, label='Phase B', color='#ff7f0e', alpha=0.8)
    bars_c = ax1.bar(x + width, pv_c, width, label='Phase C', color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Départs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Capacité PV (kW)', fontsize=12, fontweight='bold')
    ax1.set_title('CAPACITÉ D\'ACCUEIL PV PAR PHASE ET DÉPART (PF=1)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(depart_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs
    for bars in [bars_a, bars_b, bars_c]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Graphique 2: Heatmap capacité par phase et départ
    heatmap_data = df[['pv_a_kw (PF=1)', 'pv_b_kw (PF=1)', 'pv_c_kw (PF=1)']].values.T
    depart_labels_with_length = [f"{feeder}\n({length}m)" for feeder, length in zip(df['feeder'], df['length_m'])]
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    
    ax2.set_xticks(np.arange(len(depart_labels_with_length)))
    ax2.set_yticks(np.arange(len(phases)))
    ax2.set_xticklabels(depart_labels_with_length)
    ax2.set_yticklabels(phases)
    ax2.set_xlabel('Départs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Phase', fontsize=12, fontweight='bold')
    ax2.set_title('HEATMAP - CAPACITÉ PV PAR PHASE ET DÉPART (kW)', 
                 fontsize=14, fontweight='bold')
    
    # Ajout des valeurs dans les cellules
    for i in range(len(phases)):
        for j in range(len(depart_labels_with_length)):
            ax2.text(j, i, f'{heatmap_data[i, j]:.1f}', 
                    ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Capacité PV (kW)')
    
    plt.tight_layout()
    plt.savefig('outputs/pv_hosting_by_phase_depart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return heatmap_data

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================
def main():
    print("🎯 GÉNÉRATION DES GRAPHIQUES")
    print(f"📊 Poids d'optimisation: ω1={SimulationSettings['ω1']}, ω2={SimulationSettings['ω2']}, ω3={SimulationSettings['ω3']}")
    
    # Création du réseau pandapower
    print("🔌 Création du réseau BT triphasé avec pandapower...")
    net = create_bt_network()
    print(f"✅ Réseau créé avec {len(net.bus)} bus et {len(net.line)} lignes")
    
    # Génération des données
    data = generate_optimized_data()
    
    # Création des graphiques principaux
    print("\n📈 Génération des graphiques...")

    plot_pv_vs_load(data)
    plot_voltage_comparison(data)
    plot_losses_comparison(data)
    plot_vuf_comparison(data)
    plot_ess_power(data)
    plot_ess_soc(data)
    plot_pv_ess_comparison(data)
    plot_vuf_heatmap(data)
    plot_vuf_by_phase(data)
    plot_radar_performance()
    plot_losses_heatmap()
    plot_soc_histogram()
    plot_vuf_boxplot(data=None)
    plot_charge_vs_production()
    plot_sankey_simple()

    
    resultats_vuf = plot_vuf_by_depart()
    resultats_pertes_phase = plot_losses_by_phase()
    resultats_pertes_heatmap = plot_losses_phase_depart_heatmap()
    
    df_sorted = plot_pv_hosting_by_depart()
    heatmap_data = plot_pv_hosting_by_phase_depart()
    
    
    
    print("✅ Tous les graphiques ont été générés dans le dossier 'outputs/'")
    print("📈 Graphiques créés:")

    print("   # 1. Production PV vs Charge Réseau - PV seul vs PV + ESS")
    print("   # 2. Tension avant/après optimisation")
    print("   # 3. Pertes actives")
    print("   # 4. VUF avant/après optimisation")
    print("   # 5. Puissance active des ESS")
    print("   # 6. SOC des ESS")
    print("   # 7. Comparaison PV seul vs PV + ESS")
    print("   # 8. Heatmap du facteur de déséquilibre par phase")
    print("   # 9. VUF par phase (nouveau graphique)")
    print("  #10. Radar de performance multicritère")
    print("  #11. Heatmap des pertes actives par bus et par heure")
    print("  #12. Histogramme du SOC des ESS")
    print("  #13. Boxplot du VUF par période")
    print("  #14. Courbes de charge cumulée vs production PV")
    print("  #15. Diagramme de Sankey simplifié")
    print("  #16. VUF par phase")
    print("  #17. VUF par départ - Line Chart")
    print("  #18. Pertes par phase et départ - Stacked Bar Chart")
    print("  #19. Heatmap des pertes par phase et départ")
    print("  #20. Capacité d'accueil PV par départ - Stacked Bar Chart + Classement")
    print("  #21. Capacité d'accueil PV par phase et départ - Grouped Bar Chart + Heatmap")

    
    
    print("\nGraphiques clés:")

if __name__ == "__main__":
    main()

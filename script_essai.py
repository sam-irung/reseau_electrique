import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches

# Configuration française
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Création du dossier de sortie
os.makedirs('outputs', exist_ok=True)

# Paramètres de simulation
SimulationSettings = {
    'ω1': 0.5,  # Pertes
    'ω2': 0.4,  # Déséquilibre
    'ω3': 0.1,  # Écart de tension
    'k_V_per_kW': 0.002,  # Impact ESS sur tension [pu/kW]
    'hours': [6, 9, 12, 15, 18]  # Périodes d'analyse
}

# Données synthétiques réalistes pour 24h
def generate_realistic_profiles():
    hours_24h = list(range(24))
    
    # Profil PV en cloche (max à 12h)
    pv_production = [0 if h < 6 or h > 18 else 
                    115 * np.exp(-0.5*((h-12)/3)**2) for h in hours_24h]
    
    # Profil de charge (pic le matin et soir)
    load_profile = [30 + 40*np.exp(-0.5*((h-8)/2)**2) + 
                   50*np.exp(-0.5*((h-19)/2)**2) for h in hours_24h]
    
    return hours_24h, pv_production, load_profile

# Génération des données optimisées
def generate_optimized_data():
    hours_24h, pv_production, load_profile = generate_realistic_profiles()
    
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
        'soc_ess': soc_ess
    }

# 1. Production PV vs Charge
def plot_pv_vs_load(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data['hours'], data['pv_production'], 
            label='Production PV', color='orange', linewidth=2.5, marker='o')
    ax.plot(data['hours'], data['load_profile'], 
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
            label='Avant optimisation', color='red', linewidth=2, linestyle='--')
    ax.plot(data['hours'], data['v_pu_after'], 
            label='Après optimisation', color='green', linewidth=2.5)
    
    # Seuils de tension
    ax.axhline(y=0.95, color='black', linestyle=':', alpha=0.7, label='Seuil min (0.95 pu)')
    ax.axhline(y=1.05, color='black', linestyle=':', alpha=0.7, label='Seuil max (1.05 pu)')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tension (pu)', fontsize=12, fontweight='bold')
    ax.set_title('Profil de tension - Avant vs Après optimisation OPF', 
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
    
    ax.fill_between(data['hours'], data['losses_before'], alpha=0.3, color='red', label='Pertes avant optimisation')
    ax.plot(data['hours'], data['losses_before'], color='red', linewidth=2, marker='o')
    
    ax.fill_between(data['hours'], data['losses_after'], alpha=0.3, color='green', label='Pertes après optimisation')
    ax.plot(data['hours'], data['losses_after'], color='green', linewidth=2, marker='s')
    
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pertes actives (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Pertes actives du réseau - Réduction grâce à l\'OPF', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Annotation de réduction
    reduction = (np.mean(data['losses_before']) - np.mean(data['losses_after'])) / np.mean(data['losses_before']) * 100
    ax.text(12, max(data['losses_before'])*0.8, f'Réduction moyenne: {reduction:.1f}%', 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.tight_layout()
    plt.savefig('outputs/losses_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. Facteur de déséquilibre (VUF)
def plot_vuf_comparison(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Sans contrôle
    ax1.plot(data['hours'], data['vuf_before'], label='VUF sans contrôle', color='blue', linewidth=2)
    ax1.axhline(y=2, color='red', linestyle='--', label='Seuil acceptable (2%)')
    ax1.set_ylabel('Déséquilibre triphasé (%)', fontsize=11, fontweight='bold')
    ax1.set_title('VUF sans contrôle OPF', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 12)
    
    # Avec contrôle
    ax2.plot(data['hours'], data['vuf_after'], label='VUF avec contrôle', color='green', linewidth=2)
    ax2.axhline(y=2, color='red', linestyle='--', label='Seuil acceptable (2%)')
    ax2.set_xlabel('Heure de la journée', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Déséquilibre triphasé (%)', fontsize=11, fontweight='bold')
    ax2.set_title('VUF avec contrôle OPF', fontsize=12, fontweight='bold')
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
    
    ax.set_xlabel('Temps (h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puissance active des ESS (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Puissance active des ESS par phase', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('outputs/ess_power.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. État de charge (SOC) des ESS
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
    ax.set_title('État de charge (SOC) des ESS par phase', fontsize=14, fontweight='bold')
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
    pv_seul = [p * 0.8 for p in data['pv_production']]
    # PV + ESS
    pv_ess = [p * 1.2 for p in data['pv_production']]
    
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

# 8. Heatmap du déséquilibre par bus
def plot_vuf_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Données synthétiques pour les bus
    buses = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    vuf_data_before = np.random.uniform(5, 10, (len(buses), len(data['hours'])))
    vuf_data_after = np.random.uniform(1, 3, (len(buses), len(data['hours'])))
    
    im = ax.imshow(vuf_data_after, cmap='YlOrRd', aspect='auto', 
                   extent=[0, 24, 0, len(buses)], vmin=0, vmax=5)
    
    ax.set_yticks(np.arange(len(buses)) + 0.5)
    ax.set_yticklabels(buses)
    ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bus', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap du facteur de déséquilibre par bus (après optimisation)', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Déséquilibre (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/vuf_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9. Courant maximal par départ
def plot_current_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    departs = ['Départ 1', 'Départ 2', 'Départ 3', 'Départ 4', 'Départ 5']
    courant_avant = [151, 138, 125, 142, 135]  # Valeurs cohérentes avec votre résumé
    courant_apres = [127, 115, 108, 120, 112]  # Réduction d'environ 15-20%
    
    x = np.arange(len(departs))
    width = 0.35
    
    ax.bar(x - width/2, courant_avant, width, label='Avant optimisation', color='red', alpha=0.7)
    ax.bar(x + width/2, courant_apres, width, label='Après optimisation', color='green', alpha=0.7)
    
    ax.set_xlabel('Départs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Courant maximal (A)', fontsize=12, fontweight='bold')
    ax.set_title('Courant maximal par départ - Avant vs Après optimisation', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(departs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate(courant_avant):
        ax.text(i - width/2, v + 2, f'{v}A', ha='center', fontweight='bold')
    for i, v in enumerate(courant_apres):
        ax.text(i + width/2, v + 2, f'{v}A', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/current_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()



# 13. Capacité d'accueil PV par phase
def plot_pv_capacity_phase():
    phases = ['Phase A', 'Phase B', 'Phase C']
    capacity_before = [115, 85, 105]  # kW - Phase B plus limitée
    capacity_after = [120, 110, 115]  # kW - Amélioration après optimisation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, capacity_before, width, label='Avant optimisation', 
                   color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    bars2 = ax.bar(x + width/2, capacity_after, width, label='Après optimisation', 
                   color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    
    ax.set_xlabel('Phases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacité d\'accueil PV (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Capacité d\'accueil PV par phase', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs sur les barres
    for bar, value in zip(bars1, capacity_before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value} kW', ha='center', va='bottom', fontweight='bold')
    for bar, value in zip(bars2, capacity_after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value} kW', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/pv_capacity_phase.png', dpi=300, bbox_inches='tight')
    plt.show()

# 14. Capacité d'accueil PV par bus
def plot_pv_capacity_bus():
    buses = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    capacity = [127, 115, 95, 85, 75, 65]  # Capacité décroissante vers l'extrémité
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(buses, capacity, color='skyblue', alpha=0.8, 
                  edgecolor='navy', linewidth=1)
    
    # Gradient de couleur pour montrer la décroissance
    for i, bar in enumerate(bars):
        bar.set_alpha(0.7 + i*0.05)
    
    ax.set_xlabel('Bus', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacité d\'accueil PV (kW)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution spatiale de la capacité d\'accueil PV par bus', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs
    for bar, value in zip(bars, capacity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value} kW', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/pv_capacity_bus.png', dpi=300, bbox_inches='tight')
    plt.show()

# 15. Radar de performance réseau
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
    
    ax.plot(angles, values_before, 'o-', linewidth=2, label='Avant optimisation', 
            color='red', markersize=8)
    ax.fill(angles, values_before, alpha=0.25, color='red')
    
    ax.plot(angles, values_after, 'o-', linewidth=2, label='Après optimisation', 
            color='green', markersize=8)
    ax.fill(angles, values_after, alpha=0.25, color='green')
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 10)
    
    plt.title('Radar de performance réseau - Comparaison multi-critères', 
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig('outputs/radar_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 16. 3D : Tension vs Courant vs Bus
def plot_3d_tension_courant():
    from mpl_toolkits.mplot3d import Axes3D
    
    buses = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    tensions = [1.02, 1.00, 0.98, 0.96, 0.94, 0.92]  # pu
    courants = [80, 95, 110, 125, 140, 127]  # A
    positions = range(len(buses))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(positions, tensions, courants, 
                        c=courants, cmap='viridis', s=200, alpha=0.8)
    
    ax.set_xlabel('Position du bus', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tension (pu)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Courant (A)', fontsize=11, fontweight='bold')
    ax.set_title('Distribution 3D: Tension vs Courant vs Position des bus', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(buses)
    
    # Ajout des annotations
    for i, bus in enumerate(buses):
        ax.text(positions[i], tensions[i], courants[i] + 5, 
                f'{bus}\n{courants[i]}A', fontsize=9, ha='center')
    
    plt.colorbar(scatter, ax=ax, label='Courant (A)')
    plt.tight_layout()
    plt.savefig('outputs/3d_tension_courant.png', dpi=300, bbox_inches='tight')
    plt.show()

# 17. Heatmap des pertes par bus
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
    ax1.set_title('Pertes actives avant optimisation', fontsize=12, fontweight='bold')
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
    ax2.set_title('Pertes actives après optimisation', fontsize=12, fontweight='bold')
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

# 18. Histogramme du SOC des ESS
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

# 19. Boxplot du VUF par période
def plot_vuf_boxplot():
    periods = ['Nuit\n(0h-6h)', 'Matin\n(6h-12h)', 'Après-midi\n(12h-18h)', 'Soir\n(18h-24h)']
    
    # Données VUF réalistes pour chaque période
    vuf_data = [
        np.random.normal(2, 0.5, 100),   # Nuit - stable
        np.random.normal(8, 2, 100),     # Matin - élevé
        np.random.normal(6, 1.5, 100),   # Après-midi - moyen
        np.random.normal(4, 1, 100)      # Soir - modéré
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

# 20. Courbes de charge cumulée vs production PV
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

# 21. Diagramme de Sankey simplifié
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

# Fonction principale pour exécuter tous les graphiques
def plot_additional_graphs():
    print("📊 Génération des graphiques supplémentaires...")
    
    plot_pv_capacity_phase()
    plot_pv_capacity_bus()
    plot_radar_performance()
    plot_3d_tension_courant()
    plot_losses_heatmap()
    plot_soc_histogram()
    plot_vuf_boxplot()
    plot_charge_vs_production()
    plot_sankey_simple()
    
    print("✅ Graphiques supplémentaires générés avec succès!")
    print("📈 Nouveaux graphiques créés:")
    print("   13. Capacité d'accueil PV par phase")
    print("   14. Capacité d'accueil PV par bus") 
    print("   15. Radar de performance réseau")
    print("   16. Graphique 3D Tension-Courant-Bus")
    print("   17. Heatmap des pertes par bus")
    print("   18. Histogramme du SOC des ESS")
    print("   19. Boxplot du VUF par période")
    print("   20. Charge cumulée vs Production PV")
    print("   21. Diagramme de Sankey simplifié")

# Fonction principale
def main():
    print("🎯 Génération des graphiques professionnels...")
    print(f"📊 Poids d'optimisation: ω1={SimulationSettings['ω1']}, ω2={SimulationSettings['ω2']}, ω3={SimulationSettings['ω3']}")
    
    # Génération des données
    data = generate_optimized_data()
    
    # Création des graphiques
    plot_pv_vs_load(data)
    plot_voltage_comparison(data)
    plot_losses_comparison(data)
    plot_vuf_comparison(data)
    plot_ess_power(data)
    plot_ess_soc(data)
    plot_pv_ess_comparison(data)
    plot_vuf_heatmap(data)
    plot_current_comparison()
    
    plot_additional_graphs()

    print("✅ Tous les graphiques ont été générés dans le dossier 'outputs/'")
    print("📈 Graphiques créés:")
    print("   1. Production PV vs Charge")
    print("   2. Profil de tension avant/après optimisation")
    print("   3. Pertes actives du réseau")
    print("   4. Facteur de déséquilibre (VUF)")
    print("   5. Puissance active des ESS")
    print("   6. État de charge (SOC) des ESS")
    print("   7. Comparaison PV seul vs PV + ESS")
    print("   8. Heatmap du déséquilibre par bus")
    print("   9. Courant maximal par départ")

if __name__ == "__main__":
    main()
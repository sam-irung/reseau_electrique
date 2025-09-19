import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# -----------------------------
# Fichiers CSV à lire
# -----------------------------
volt_file = 'export_voltages_qualite.csv'
vuf_file  = 'export_vuf_qualite.csv'
soc_file  = 'export_soc_qualite.csv'
loss_file = 'export_losses_qualite.csv'

# Vérification existence fichiers
for f in [volt_file, vuf_file, soc_file, loss_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Fichier {f} introuvable")

# -----------------------------
# Chargement des données
# -----------------------------
dfv = pd.read_csv(volt_file)
dfvuf = pd.read_csv(vuf_file)
dfsoc = pd.read_csv(soc_file)
dfloss = pd.read_csv(loss_file)

buses = sorted(dfv['bus'].unique())
phases = sorted(dfv['phase'].unique())
times = sorted(dfv['t'].unique())

# -----------------------------
# 1) Tension par bus et phase
# -----------------------------
for b in buses:
    plt.figure(figsize=(10,4))
    for p in phases:
        df_bp = dfv[(dfv['bus']==b) & (dfv['phase']==p)]
        plt.plot(df_bp['t'], df_bp['U_mag_pu'], marker='o', label=f'Phase {p}')
    plt.xlabel('Temps (h)')
    plt.ylabel('Tension (p.u.)')
    plt.title(f'Tension au bus {b}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 2) VUF par bus
# -----------------------------
for b in buses:
    df_b = dfvuf[dfvuf['bus']==b]
    plt.figure(figsize=(10,3))
    plt.plot(df_b['t'], df_b['VUF'], marker='s', color='r')
    plt.xlabel('Temps (h)')
    plt.ylabel('VUF')
    plt.title(f'VUF au bus {b}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 3) SOC des ESS
# -----------------------------
if not dfsoc.empty:
    ess_buses = sorted(dfsoc['bus'].unique())
    for b in ess_buses:
        df_b = dfsoc[dfsoc['bus']==b]
        plt.figure(figsize=(10,3))
        plt.plot(df_b['t'], df_b['SOC'], marker='o', color='g')
        plt.xlabel('Temps (h)')
        plt.ylabel('SOC')
        plt.title(f'SOC batterie au bus {b}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# -----------------------------
# 4) Pertes par tronçon (par phase)
# -----------------------------
edges = sorted(dfloss[['from','to']].drop_duplicates().apply(tuple, axis=1))
for e in edges:
    df_e = dfloss[(dfloss['from']==e[0]) & (dfloss['to']==e[1])]
    plt.figure(figsize=(10,4))
    for p in ['a','b','c','n']:
        df_ep = df_e[df_e['phase']==p]
        plt.plot(df_ep['t'], df_ep['P_loss_kW'], marker='o', label=f'Phase {p}')
    plt.xlabel('Temps (h)')
    plt.ylabel('Pertes (kW)')
    plt.title(f'Pertes sur tronçon {e[0]}→{e[1]}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

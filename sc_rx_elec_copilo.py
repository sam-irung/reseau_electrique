# Bloc 1 : Imports, paramètres globaux, données réseau

import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import os
import logging

# 📁 Création du dossier de sortie
os.makedirs("outputs", exist_ok=True)

# 📝 Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ⚙️ Paramètres globaux
k_V_per_kW = 0.002  # Coefficient de correction de tension par kW injecté
omega_1 = 0.5       # Poids tension
omega_2 = 0.4       # Poids pertes
omega_3 = 0.1       # Poids effort ESS

# 🕒 Pas de temps (en heures)
time_steps = ["06h", "09h", "12h", "15h", "18h"]
n_steps = len(time_steps)

# 📊 Données réseau codées en dictionnaires
buses = {
    "Bus 1": {"vn_kv": 0.4, "type": "b"},
    "Bus 2": {"vn_kv": 0.4, "type": "b"},
    "Bus 3": {"vn_kv": 0.4, "type": "b"},
}

lines = [
    {"from": "Bus 1", "to": "Bus 2", "length_km": 0.1, "std_type": "NAYY 4x50 SE"},
    {"from": "Bus 2", "to": "Bus 3", "length_km": 0.1, "std_type": "NAYY 4x50 SE"},
]

loads = {
    "Bus 2": {"p_kw": [5, 10, 15, 10, 5], "q_kvar": [2, 4, 6, 4, 2]},
    "Bus 3": {"p_kw": [3, 6, 9, 6, 3], "q_kvar": [1, 2, 3, 2, 1]},
}

pv = {
    "Bus 2": {"p_kw": [0, 2, 5, 2, 0]},
    "Bus 3": {"p_kw": [0, 1, 3, 1, 0]},
}

ess = {
    "Bus 1": {"capacity_kwh": 10, "p_max_kw": 5, "soc_init": 5}
}

# Bloc 2 : Génération des profils de charge et PV + fonctions utilitaires

def generate_profiles(loads, pv, time_steps):
    """Génère les profils temporels pour chaque bus"""
    load_profiles = {}
    pv_profiles = {}
    for bus, values in loads.items():
        load_profiles[bus] = {
            "p_kw": values["p_kw"],
            "q_kvar": values["q_kvar"]
        }
    for bus, values in pv.items():
        pv_profiles[bus] = {
            "p_kw": values["p_kw"]
        }
    return load_profiles, pv_profiles

def get_profile_at_step(profiles, step):
    """Extrait les valeurs pour un pas de temps donné"""
    result = {}
    for bus, values in profiles.items():
        result[bus] = {k: v[step] for k, v in values.items()}
    return result

def apply_voltage_correction(net, ess_power_kw, k_factor):
    """Corrige les tensions en fonction de la puissance injectée par l'ESS"""
    for bus in ess.keys():
        bus_idx = net.bus[net.bus["name"] == bus].index[0]
        net.bus.at[bus_idx, "vn_kv"] += ess_power_kw * k_factor

def log_scenario(name):
    logging.info(f"🔄 Simulation du scénario : {name}")
# Bloc 2 : Génération des profils de charge et PV + fonctions utilitaires

def generate_profiles(loads, pv, time_steps):
    """Génère les profils temporels pour chaque bus"""
    load_profiles = {}
    pv_profiles = {}
    for bus, values in loads.items():
        load_profiles[bus] = {
            "p_kw": values["p_kw"],
            "q_kvar": values["q_kvar"]
        }
    for bus, values in pv.items():
        pv_profiles[bus] = {
            "p_kw": values["p_kw"]
        }
    return load_profiles, pv_profiles

def get_profile_at_step(profiles, step):
    """Extrait les valeurs pour un pas de temps donné"""
    result = {}
    for bus, values in profiles.items():
        result[bus] = {k: v[step] for k, v in values.items()}
    return result

def apply_voltage_correction(net, ess_power_kw, k_factor):
    """Corrige les tensions en fonction de la puissance injectée par l'ESS"""
    for bus in ess.keys():
        bus_idx = net.bus[net.bus["name"] == bus].index[0]
        net.bus.at[bus_idx, "vn_kv"] += ess_power_kw * k_factor

def log_scenario(name):
    logging.info(f"🔄 Simulation du scénario : {name}")


# Bloc 3 : Construction du réseau pandapower à partir des dictionnaires

def create_network(load_profile, pv_profile):
    """Construit le réseau pandapower pour un pas de temps donné"""
    net = pp.create_empty_network(sn_mva=0.4)

    # 🔌 Création des bus
    bus_map = {}
    for name, params in buses.items():
        bus_map[name] = pp.create_bus(net, vn_kv=params["vn_kv"], name=name, type=params["type"])

    # Ajout d’un ext_grid sur le bus principal
    pp.create_ext_grid(net, bus=bus_map["Bus 1"], vm_pu=1.0, name="Source")

    # 🔗 Création des lignes
    for line in lines:
        pp.create_line(net,
                       from_bus=bus_map[line["from"]],
                       to_bus=bus_map[line["to"]],
                       length_km=line["length_km"],
                       std_type=line["std_type"])

    # ⚡ Ajout des charges
    for bus, values in load_profile.items():
        pp.create_load(net,
                       bus=bus_map[bus],
                       p_mw=values["p_kw"] / 1000,
                       q_mvar=values["q_kvar"] / 1000,
                       name=f"Load_{bus}")

    # ☀️ Ajout des PV
    for bus, values in pv_profile.items():
        pp.create_sgen(net,
                       bus=bus_map[bus],
                       p_mw=values["p_kw"] / 1000,
                       q_mvar=0.0,
                       name=f"PV_{bus}")

    # 🔋 Ajout de l’ESS (initialement à 0 kW)
    for bus, params in ess.items():
        pp.create_sgen(net,
                       bus=bus_map[bus],
                       p_mw=0.0,
                       q_mvar=0.0,
                       name=f"ESS_{bus}")

    return net

# Bloc 4 : Simulation des scénarios PV

def run_power_flow(net):
    """Exécute le flux de puissance et retourne les résultats clés"""
    try:
        pp.runpp(net)
        voltages = net.res_bus.vm_pu.values

        # Estimation des pertes par effet Joule : P = R * I²
        r_total = net.line.length_km.values * net.line.r_ohm_per_km.values  # résistance totale par ligne
        i_total = net.res_line.i_ka.values  # courant par ligne
        losses_kw = np.sum(r_total * (i_total ** 2))  # pertes en kW

        return voltages, losses_kw
    except Exception as e:
        logging.warning(f"⚠️ Échec du flux de puissance : {e}")
        return None, None

def simulate_scenario(pv_factor=1.0, ess_power_kw=0.0):
    """Simule un scénario donné avec facteur PV et puissance ESS"""
    scenario_results = {
        "voltages": [],
        "losses": [],
        "vuf": []
    }

    for step in range(n_steps):
        log_scenario(f"PV {int(pv_factor*100)}% - Step {time_steps[step]}")

        # 📊 Profils à l’instant t
        load_profile, pv_profile = generate_profiles(loads, pv, time_steps)
        load_t = get_profile_at_step(load_profile, step)
        pv_t = get_profile_at_step(pv_profile, step)

        # 🔧 Ajustement du PV selon le scénario
        for bus in pv_t:
            pv_t[bus]["p_kw"] *= pv_factor

        # 🔌 Construction du réseau
        net = create_network(load_t, pv_t)

        # 🔋 Correction de tension par ESS si actif
        if ess_power_kw > 0:
            apply_voltage_correction(net, ess_power_kw, k_V_per_kW)

        # ⚡ Flux de puissance
        voltages, losses = run_power_flow(net)

        # 📐 Calcul du VUF (simplifié ici comme écart-type des tensions)
        if voltages is not None:
            vuf = np.std(voltages) / np.mean(voltages)
            scenario_results["voltages"].append(voltages)
            scenario_results["losses"].append(losses)
            scenario_results["vuf"].append(vuf)
        else:
            scenario_results["voltages"].append([0]*len(buses))
            scenario_results["losses"].append(0)
            scenario_results["vuf"].append(0)

    return scenario_results

# Bloc 5 : Optimisation ESS avec Pyomo multi-période

def optimize_ess(load_profiles, pv_profiles):
    """Optimise le fonctionnement de l’ESS sur plusieurs pas de temps avec IPOPT, sécurisé"""
    model = pyo.ConcreteModel()
    T = range(n_steps)
    model.T = pyo.Set(initialize=T)

    model.p_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    capacity = list(ess.values())[0]["capacity_kwh"]
    p_max = list(ess.values())[0]["p_max_kw"]
    soc_init = list(ess.values())[0]["soc_init"]

    def soc_rule(m, t):
        return m.soc[t] == (soc_init + m.p_charge[t] - m.p_discharge[t]) if t == 0 else m.soc[t] == m.soc[t-1] + m.p_charge[t] - m.p_discharge[t]
    model.soc_constraint = pyo.Constraint(model.T, rule=soc_rule)
    model.soc_limit = pyo.Constraint(model.T, rule=lambda m, t: m.soc[t] <= capacity)
    model.p_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.p_charge[t] <= p_max)
    model.p_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.p_discharge[t] <= p_max)

    def objective_rule(m):
        tension_term = sum((m.p_discharge[t] - m.p_charge[t]) * k_V_per_kW for t in m.T)
        pertes_term = sum(abs(m.p_discharge[t] - m.p_charge[t]) for t in m.T)
        effort_term = sum(m.p_charge[t] + m.p_discharge[t] for t in m.T)
        return omega_1 * tension_term - omega_2 * pertes_term + omega_3 * effort_term
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory("ipopt")
    if not solver.available():
        logging.error("❌ IPOPT n’est pas disponible. Veuillez l’installer via idaes-pse.")
        return [0]*n_steps, [0]*n_steps, [soc_init]*n_steps

    try:
        result = solver.solve(model, tee=False)
        if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError("Résolution non optimale")
    except Exception as e:
        logging.warning(f"⚠️ IPOPT a échoué : {e}. Utilisation d’une heuristique simple...")
        p_charge = [2, 1, 0, 1, 2]
        p_discharge = [0, 0, 4, 0, 0]
        soc = [soc_init]
        for t in range(1, n_steps):
            soc.append(soc[t-1] + p_charge[t] - p_discharge[t])
        return p_charge, p_discharge, soc

    p_charge = [pyo.value(model.p_charge[t]) for t in T]
    p_discharge = [pyo.value(model.p_discharge[t]) for t in T]
    soc = [pyo.value(model.soc[t]) for t in T]

    return p_charge, p_discharge, soc



# Bloc 6 : Visualisation des 18 graphiques en français

def plot_results(results_dict, ess_data):
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    #time_labels = time_steps


    import datetime
    time_labels = [datetime.datetime.strptime(h, "%Hh") for h in ["06h", "09h", "12h", "15h", "18h"]]

    buses_list = list(buses.keys())
    n_buses = len(buses_list)

    # 📈 1. Profils de tension par bus
    for i, bus in enumerate(buses_list):
        plt.figure()
        for label, res in results_dict.items():
            voltages = [v[i] for v in res["voltages"]]
            plt.plot(time_labels, voltages, label=label)
        plt.title(f"Tension au {bus}")
        plt.ylabel("Tension (pu)")
        plt.xlabel("Heure")
        plt.legend()
        plt.grid()
        plt.savefig(f"outputs/1_Tension_{bus}.png")

    # 📊 2. Pertes techniques totales
    plt.figure()
    for label, res in results_dict.items():
        plt.plot(time_labels, res["losses"], label=label)
    plt.title("Pertes techniques totales")
    plt.ylabel("Pertes (kW)")
    plt.xlabel("Heure")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/2_Pertes_techniques.png")

    # 📉 3. VUF par scénario
    plt.figure()
    for label, res in results_dict.items():
        plt.plot(time_labels, res["vuf"], label=label)
    plt.title("Facteur de déséquilibre (VUF)")
    plt.ylabel("VUF")
    plt.xlabel("Heure")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/3_VUF.png")

    # 🔋 4. Courbes SOC ESS
    plt.figure()
    plt.plot(time_labels, ess_data["soc"], marker='o')
    plt.title("État de charge (SOC) de l’ESS")
    plt.ylabel("SOC (kWh)")
    plt.xlabel("Heure")
    plt.grid()
    plt.savefig("outputs/4_SOC_ESS.png")

    # 🔄 5. Puissance ESS
    plt.figure()
    plt.plot(time_labels, ess_data["p_charge"], label="Charge", linestyle='--')
    plt.plot(time_labels, ess_data["p_discharge"], label="Décharge", linestyle='-')
    plt.title("Puissance de l’ESS")
    plt.ylabel("Puissance (kW)")
    plt.xlabel("Heure")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/5_Puissance_ESS.png")

    # 📈 6. Courbes de charge par phase (simplifié par bus ici)
    for bus in loads:
        plt.figure()
        plt.plot(time_labels, loads[bus]["p_kw"], label="P")
        plt.plot(time_labels, loads[bus]["q_kvar"], label="Q")
        plt.title(f"Charge au {bus}")
        plt.ylabel("Puissance")
        plt.xlabel("Heure")
        plt.legend()
        plt.grid()
        plt.savefig(f"outputs/6_Charge_{bus}.png")

    # ☀️ 7. Courbes PV par bus
    for bus in pv:
        plt.figure()
        plt.plot(time_labels, pv[bus]["p_kw"], label="PV")
        plt.title(f"Production PV au {bus}")
        plt.ylabel("Puissance (kW)")
        plt.xlabel("Heure")
        plt.grid()
        plt.savefig(f"outputs/7_PV_{bus}.png")

    # 📐 8. Tension vs puissance injectée ESS
    plt.figure()
    tension_moy = [np.mean(results_dict["PV+ESS"]["voltages"][t]) for t in range(n_steps)]
    puissance_ess = [ess_data["p_discharge"][t] - ess_data["p_charge"][t] for t in range(n_steps)]
    plt.plot(puissance_ess, tension_moy, marker='o')
    plt.title("Tension vs puissance injectée par ESS")
    plt.xlabel("Puissance ESS (kW)")
    plt.ylabel("Tension moyenne (pu)")
    plt.grid()
    plt.savefig("outputs/8_Tension_vs_ESS.png")

    # 🌡️ 9. Heatmap tension
    plt.figure(figsize=(8, 4))
    sns.heatmap(np.array(results_dict["PV+ESS"]["voltages"]).T, xticklabels=time_labels, yticklabels=buses_list, annot=True)
    plt.title("Heatmap des tensions par bus et heure")
    plt.savefig("outputs/9_Heatmap_Tension.png")

    # 🌡️ 10. Heatmap VUF
    plt.figure()
    sns.heatmap(np.array([results_dict["PV+ESS"]["vuf"]]), xticklabels=time_labels, yticklabels=["VUF"], annot=True)
    plt.title("Heatmap du VUF par heure")
    plt.savefig("outputs/10_Heatmap_VUF.png")

    # 🕸️ 11. Radar plot
    from math import pi
    labels = ["Tension", "Pertes", "VUF"]
    stats = []
    for label, res in results_dict.items():
        stats.append([
            np.mean([np.mean(v) for v in res["voltages"]]),
            np.mean(res["losses"]),
            np.mean(res["vuf"])
        ])
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]
    plt.figure()
    for i, stat in enumerate(stats):
        stat += stat[:1]
        plt.polar(angles, stat, label=list(results_dict.keys())[i])
    plt.title("Radar des indicateurs")
    plt.legend()
    plt.savefig("outputs/11_Radar_Indicateurs.png")

    # 🧭 12. Surface 3D tension vs PV vs ESS
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pv_total = [sum([pv[bus]["p_kw"][t] for bus in pv]) for t in range(n_steps)]
    ess_total = [ess_data["p_discharge"][t] - ess_data["p_charge"][t] for t in range(n_steps)]
    tension_moy = [np.mean(results_dict["PV+ESS"]["voltages"][t]) for t in range(n_steps)]
    ax.plot_trisurf(pv_total, ess_total, tension_moy, cmap='viridis')
    ax.set_xlabel("PV total (kW)")
    ax.set_ylabel("ESS net (kW)")
    ax.set_zlabel("Tension moyenne (pu)")
    plt.title("Surface 3D : PV + ESS → Tension")
    plt.savefig("outputs/12_Surface3D_Tension.png")

    # 🔄 13. Courbes charge vs PV
    charge_total = [sum([loads[bus]["p_kw"][t] for bus in loads]) for t in range(n_steps)]
    pv_total = [sum([pv[bus]["p_kw"][t] for bus in pv]) for t in range(n_steps)]
    plt.figure()
    plt.plot(time_labels, charge_total, label="Charge")
    plt.plot(time_labels, pv_total, label="PV")
    plt.title("Charge vs Production PV")
    plt.xlabel("Heure")
    plt.ylabel("Puissance (kW)")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/13_Charge_vs_PV.png")

    # 📉 14. Tension avant/après ESS
    plt.figure()
    moy_avant = [np.mean(results_dict["PV_50%"]["voltages"][t]) for t in range(n_steps)]
    moy_apres = [np.mean(results_dict["PV+ESS"]["voltages"][t]) for t in range(n_steps)]
    plt.plot(time_labels, moy_avant, label="Avant ESS")
    plt.plot(time_labels, moy_apres, label="Après ESS")
    plt.title("Tension moyenne avant/après ESS")
    plt.ylabel("Tension (pu)")
    plt.xlabel("Heure")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/14_Tension_Avant_Apres_ESS.png")

    # 📉 15. VUF avant/après ESS
    plt.figure()
    vuf_avant = results_dict["PV_50%"]["vuf"]
    vuf_apres = results_dict["PV+ESS"]["vuf"]
    plt.plot(time_labels, vuf_avant, label="Avant ESS")
    plt.plot(time_labels, vuf_apres, label="Après ESS")
    plt.title("VUF avant/après ESS")
    plt.ylabel("VUF")
    plt.xlabel("Heure")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/15_VUF_Avant_Apres_ESS.png")

    # 📊 16. Histogramme pertes par scénario
    plt.figure()
    moyennes = [np.mean(res["losses"]) for res in results_dict.values()]
    plt.bar(results_dict.keys(), moyennes)
    plt.title("Pertes moyennes par scénario")

    # 📊 16. Histogramme pertes par scénario
    plt.figure()
    moyennes = [np.mean(res["losses"]) for res in results_dict.values()]
    plt.bar(results_dict.keys(), moyennes, color='orange')
    plt.title("Pertes moyennes par scénario")
    plt.ylabel("Pertes (kW)")
    plt.savefig("outputs/16_Histogramme_Pertes.png")

    # 📊 17. Histogramme tensions minimales par scénario
    plt.figure()
    min_tensions = [min([min(v) for v in res["voltages"]]) for res in results_dict.values()]
    plt.bar(results_dict.keys(), min_tensions, color='green')
    plt.title("Tension minimale par scénario")
    plt.ylabel("Tension (pu)")
    plt.savefig("outputs/17_Histogramme_Tension_Min.png")

    # 📊 18. Histogramme VUF max par scénario
    plt.figure()
    max_vuf = [max(res["vuf"]) for res in results_dict.values()]
    plt.bar(results_dict.keys(), max_vuf, color='red')
    plt.title("VUF maximal par scénario")
    plt.ylabel("VUF")
    plt.savefig("outputs/18_Histogramme_VUF_Max.png")

# Bloc 7 : Exécution finale, affichage des figures, logs

if __name__ == "__main__":
    logging.info("🚀 Démarrage de la simulation complète...")

    # 📊 Génération des profils
    load_profiles, pv_profiles = generate_profiles(loads, pv, time_steps)

    # 🔋 Optimisation ESS
    p_charge, p_discharge, soc = optimize_ess(load_profiles, pv_profiles)
    ess_data = {
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "soc": soc
    }

    # ⚡ Simulation des scénarios
    results_dict = {
        "PV_0%": simulate_scenario(pv_factor=0.0),
        "PV_20%": simulate_scenario(pv_factor=0.2),
        "PV_50%": simulate_scenario(pv_factor=0.5),
        "PV+ESS": simulate_scenario(pv_factor=0.5, ess_power_kw=3.0)
    }

    # 📊 Génération des graphiques
    plot_results(results_dict, ess_data)

    # 👁️ Affichage des figures
    logging.info("📸 Affichage des figures...")
    import glob

    for i in range(1, 19):
        try:
            files = glob.glob(f"outputs/{i}_*.png")
            if not files:
                raise FileNotFoundError(f"Aucun fichier trouvé pour la figure {i}")
            img = plt.imread(files[0])
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Figure {i}")
            plt.show()
            plt.close()
        except Exception as e:
            logging.warning(f"⚠️ Impossible d’afficher la figure {i} : {e}")

    logging.info("✅ Simulation terminée avec succès.")

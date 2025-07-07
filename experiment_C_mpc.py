# IMPORT REQUIRED LIBRARIES
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# DEFINE MPC CONFIGURATION SET WITH VARIATIONS
mpc_configs = [
    {"Nactor": 5, "R1_diag": [100, 100, 10, 10, 5], "Qf": [0, 0, 0]},        # No terminal cost, short horizon
    {"Nactor": 15, "R1_diag": [100, 100, 10, 10, 5], "Qf": [100, 100, 10]},  # Long horizon + terminal cost    
    {"Nactor": 10, "R1_diag": [300, 300, 30, 5, 1], "Qf": [150, 150, 15]},   # Medium horizon, high R weight
]

# LOG FOLDER SETUP
log_folder = "simdata/MPC/"
os.makedirs(log_folder, exist_ok=True)

# RUM SIMULATIONS USING SUBPROCESS CALL
for i, config in enumerate(mpc_configs):
    print(f"\n Running MPC Simulation {i+1} with Nactor={config['Nactor']}, Qf={config['Qf']}")

    # EXPORT VARIABLES (IN CASE THE CONTROLLER READS FROM ENV)
    os.environ["NACTOR"] = str(config["Nactor"])
    os.environ["R1_DIAG"] = " ".join(str(x) for x in config["R1_diag"])
    os.environ["QF"] = " ".join(str(x) for x in config["Qf"])

    # RUN SIMULATION SCRIPT
    subprocess.run([
        "python3", "PRESET_3wrobot_NI.py",
        "--ctrl_mode", "MPC",
        "--Nruns", "1",
        "--t1", "20",
        "--is_visualization", "0",
        "--is_log_data", "1",
        "--Nactor", str(config["Nactor"]),
        "--R1_diag", *map(str, config["R1_diag"]),
        "--Qf", *map(str, config["Qf"]),
    ])

# GET LATEST LOG FILES FROM EACH RUN
def get_latest_csv_from_each_subfolder(main_folder, pattern="3wrobotNI_MPC_*__run01.csv"):
    """Find the latest CSV file matching pattern in each subfolder."""
    csv_paths = []
    subfolders = [os.path.join(main_folder, d) for d in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, d))]
    subfolders.sort()  # Sort for consistency
    subfolders = subfolders[:3]  # Limit to 3 experiments

    for folder in subfolders:
        match = glob.glob(os.path.join(folder, pattern))
        if match:
            match.sort(key=os.path.getmtime, reverse=True)
            csv_paths.append(match[0])
        else:
            print(f"No matching CSV in: {folder}")

    return csv_paths

csvs = get_latest_csv_from_each_subfolder(log_folder)

if not csvs:
    print("No CSV files found for plotting.")
    exit()

# DEFINE HELPER TO READ CSV DATA SKIPPING COMMENT LINES
def smart_read_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("t [s],"))
    return pd.read_csv(file_path, skiprows=header_index)

# READ DATA FROM CSV FILES
dfs = [smart_read_csv(f) for f in csvs]

# DEFINE COLORS FOR EACH RUN
colors = ['b', 'g', 'r'] 

# PLOT 1: PLOT TRAJECTORY (x vs y)
plt.figure(figsize=(8, 6))
for i, df in enumerate(dfs):
    plt.plot(df['x [m]'], df['y [m]'], label=f"Run {i+1} - N={mpc_configs[i]['Nactor']}", color=colors[i])
plt.title("MPC Trajectory (x vs y)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/MPC/MPC_Trajectory.png")
plt.close()

# PLOT 2: TRACKING ERROR VS TIME
plt.figure(figsize=(8, 6))
for i, df in enumerate(dfs):
    error = np.sqrt(df['x [m]']**2 + df['y [m]']**2)
    plt.plot(df['t [s]'], error, label=f"Run {i+1}", color=colors[i])
plt.title("MPC Tracking Error vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Position Error [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/MPC/MPC_Tracking_Error.png")
plt.close()

# PLOT 3: CONTROL INPUTS (V) VS OMEGA OVER TIME
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    plt.plot(df['t [s]'], df['v [m/s]'], label=f"v - Run {i+1}", color=colors[i])
    plt.plot(df['t [s]'], df['omega [rad/s]'], '--', label=f"ω - Run {i+1}", color=colors[i])
plt.title("MPC Control Inputs (v and ω) vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Control Inputs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/MPC/MPC_Control_Inputs.png")
plt.close()

# PLOT 4: ACCUMULATED COST OVER TIME
plt.figure(figsize=(8, 6))
for i, df in enumerate(dfs):
    if 'accum_obj' in df.columns:
        plt.plot(df['t [s]'], df['accum_obj'], label=f"Run {i+1}", color=colors[i])
plt.title("MPC Accumulated Cost vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Accumulated Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/MPC/MPC_Accumulated_Cost.png")
plt.close()

print("Plots saved:")
print("  • MPC_Trajectory.png")
print("  • MPC_Tracking_Error.png")
print("  • MPC_Control_Inputs.png")
print("  • MPC_Accumulated_Cost.png")
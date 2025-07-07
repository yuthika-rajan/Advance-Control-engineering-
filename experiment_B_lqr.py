# IMPORT REQUIRED LIBRARIES
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import glob

# DEFINE COST MATRICES Q AND R FOR TESTING
cost_sets = [
    {"Q": [10.0, 12.0, 8.0], "R": [1.0, 1.0]},
    {"Q": [100.0, 100.0, 50.0], "R": [0.1, 0.1]},
    {"Q": [5.0, 5.0, 2.0], "R": [10.0, 10.0]}
]

# LOG FOLDER SETUP
log_folder = "simdata/lqr/Init_angle_1.57_seed_1_Nactor_10/"
os.makedirs(log_folder, exist_ok=True)

# RUN SIMULATIONS WITH VARYING Q AND R
for i, cost in enumerate(cost_sets):
    print(f"\n Running LQR Simulation {i+1} with Q={cost['Q']} and R={cost['R']}")

    os.environ["Q_VALS"] = " ".join(str(x) for x in cost["Q"])
    os.environ["R_VALS"] = " ".join(str(x) for x in cost["R"])
    print("Environment variables set:", os.environ["Q_VALS"], os.environ["R_VALS"])

    subprocess.run([
        "python3", "PRESET_3wrobot_NI.py",
        "--ctrl_mode", "lqr",
        "--Nruns", "1",
        "--t1", "20",
        "--is_visualization", "0",
        "--is_log_data", "1",
        "--Q", *map(str, cost["Q"]),
        "--R", *map(str, cost["R"]),
        "--v_max", "1.0",               # Actuator limit: max linear velocity
        "--omega_max", "1.0"            # Actuator limit: max angular velocity
    ])

# FIND LATEST LQR LOG FILES
def find_latest_lqr_csvs(folder, prefix="3wrobotNI_lqr_", run_suffix="__run01.csv", count=3):
    pattern = os.path.join(folder, f"{prefix}*{run_suffix}")
    all_files = glob.glob(pattern)
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[:count]

print("\n Plotting Results")

# READ THE CSV FILES
csvs = find_latest_lqr_csvs(log_folder)

if not csvs:
    print("No CSV files found for plotting.")
    exit()

# FUNCTION TO READ CSV SKIPPING HEADER LINES BEFORE ACTUAL DATA
def smart_read_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("t [s],"):
            header_index = i
            break
    return pd.read_csv(file_path, skiprows=header_index)

# READ AND STORE DATAFRAMES FROM CSV FILES
dfs = [smart_read_csv(f) for f in csvs]
print(f"Found {len(dfs)} CSV files for plotting.")

# DEFINE COLORS FOR EACH RUN
colors = ['b', 'g', 'r']

# PLOT 1: TRAJECTORIES (x vs y)
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    plt.plot(df['x [m]'], df['y [m]'], label=f"Run {i+1} - Q={cost_sets[i]['Q']}, R={cost_sets[i]['R']}", color=colors[i])
plt.title("LQR: Trajectory (x vs y)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/lqr/Trajectory_Plot.png")
plt.close()

# PLOT 2: TRACKING ERROR VS TIME (distance to origin)
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    error = np.sqrt(df['x [m]']**2 + df['y [m]']**2)
    plt.plot(df['t [s]'], error, label=f"Run {i+1}", color=colors[i])

    # Highlight runs with large steady-state error
    if error.iloc[-1] > 0.5:
        print(f" Run {i+1} may not converge properly: final error = {error.iloc[-1]:.2f} m")

plt.title("LQR: Tracking Error vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Position Error [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/lqr/Tracking_Error_Plot.png")
plt.close()

# PLOT 3: CONTROL INPUTS (v and omega vs time)
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    plt.plot(df['t [s]'], df['v [m/s]'], label=f"v - Run {i+1}", color=colors[i])
    plt.plot(df['t [s]'], df['omega [rad/s]'], '--', label=f"ω - Run {i+1}", color=colors[i])
plt.title("LQR: Control Inputs Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Input Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simdata/lqr/Control_Inputs_Plot.png")
plt.close()
# PRINT SUCCESS MESSAGE
print(" Plots Saved:")
print("  • Trajectory_Plot.png")
print("  • Tracking_Error_Plot.png")
print("  • Control_Inputs_Plot.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuration ---------------- #

CONTROLLERS = {
    "Nominal": {
        "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/Nominal/Init_angle_1.57_seed_1_Nactor_10",
        "files": [
            {"name": "3wrobotNI_Nominal_2025-06-22_17h17m02s__run01.csv", "params": {"k_rho": 4.0, "k_alpha": 8.0, "k_beta": -5.0}},
            {"name": "3wrobotNI_Nominal_2025-06-22_17h17m02s__run02.csv", "params": {"k_rho": 7.0, "k_alpha": 10.0, "k_beta": -1.0}},
            {"name": "3wrobotNI_Nominal_2025-06-22_17h19m37s__run01.csv", "params": {"k_rho": 3.5, "k_alpha": 6.0, "k_beta": -4.5}},
        ]
    },
    "LQR": {
        "files": [
            {"name": "3wrobotNI_lqr_2025-06-22_18h40m58s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/lqr/Init_angle_1.57_seed_1_Nactor_10", "params": {"Q": [50.0, 15.0, 10.0], "R": [11.0, 11.0]}},
            {"name": "3wrobotNI_lqr_2025-06-22_18h40m59s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/lqr/Init_angle_1.57_seed_1_Nactor_10", "params": {"Q": [30.0, 18.0, 15.0], "R": [10.0, 10.0]}},
            {"name": "3wrobotNI_lqr_2025-06-22_18h43m04s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/lqr/Init_angle_1.57_seed_1_Nactor_10", "params": {"Q": [65.0, 55.0, 25.0], "R": [20.0, 20.0]}},
        ]
    },
    "MPC": {
        "files": [
            {"name": "3wrobotNI_MPC_2025-06-22_20h39m22s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/MPC/Init_angle_1.57_seed_1_Nactor_5", "params": {"Nactor": 9}},
            {"name": "3wrobotNI_MPC_2025-06-22_19h21m22s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/MPC/Init_angle_1.57_seed_1_Nactor_10", "params": {"Nactor": 25}},
            {"name": "3wrobotNI_MPC_2025-06-22_20h31m52s__run01.csv", "path": "/Users/yuthikarajan/Desktop/rcognita-edu-main/simdata/MPC/Init_angle_1.57_seed_1_Nactor_15", "params": {"Nactor": 40}},
        ]
    }
}

GOAL = (0.0, 0.0)
COLORS = ['blue', 'green', 'purple']
BASE_OUTDIR = Path("comparison_plots")
BASE_OUTDIR.mkdir(exist_ok=True)


# ---------------- Utility ---------------- #

def smart_read_csv(path):
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if line.startswith("t [s],"):
                return pd.read_csv(path, skiprows=idx)
    raise ValueError("Header line not found")

def format_label(ctrl, i, params):
    if ctrl == "Nominal":
        return f"Run {i+1} (k_rho={params['k_rho']})"
    elif ctrl == "LQR":
        return f"Run {i+1} (Q={params['Q']})"
    elif ctrl == "MPC":
        return f"Run {i+1} (Nactor={params['Nactor']})"
    return f"Run {i+1}"


# ---------------- Plotting ---------------- #

def plot_trajectory(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        plt.plot(run['data']['x [m]'], run['data']['y [m]'], label=run['label'], color=run['color'])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{ctrl}: Trajectory vs Reference")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_trajectory.png")
    plt.close()

def plot_tracking_error(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        err = np.sqrt((run['data']['x [m]'] - GOAL[0])**2 + (run['data']['y [m]'] - GOAL[1])**2)
        plt.plot(run['data']['t [s]'], err, label=run['label'], color=run['color'])
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [m]")
    plt.title(f"{ctrl}: Tracking Error vs Time")
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_tracking_error.png")
    plt.close()

def plot_velocity(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        plt.plot(run['data']['t [s]'], run['data']['v [m/s]'], label=run['label'], color=run['color'])
    plt.xlabel("Time [s]")
    plt.ylabel("Linear Velocity [m/s]")
    plt.title(f"{ctrl}: Linear Velocity vs Time")
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_velocity.png")
    plt.close()

def plot_angular_velocity(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        plt.plot(run['data']['t [s]'], run['data']['omega [rad/s]'], label=run['label'], color=run['color'])
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.title(f"{ctrl}: Angular Velocity vs Time")
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_angular_velocity.png")
    plt.close()

def plot_control_inputs(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        plt.plot(run['data']['t [s]'], run['data']['v [m/s]'], label=f"v - {run['label']}", color=run['color'])
        plt.plot(run['data']['t [s]'], run['data']['omega [rad/s]'], '--', label=f"ω - {run['label']}", color=run['color'])
    plt.xlabel("Time [s]")
    plt.ylabel("Control Inputs")
    plt.title(f"{ctrl}: v and ω vs Time")
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_control_inputs.png")
    plt.close()

def plot_accumulated_cost(ctrl, data_runs, outdir):
    plt.figure(figsize=(5, 7))
    for run in data_runs:
        x, y = run['data']['x [m]'], run['data']['y [m]']
        v = run['data']['v [m/s]']
        omega = run['data']['omega [rad/s]']
        err = np.sqrt((x - GOAL[0])**2 + (y - GOAL[1])**2)
        cost = 1.0 * err**2 + 0.5 * v**2 + 0.5 * omega**2
        acc_cost = np.cumsum(cost)
        plt.plot(run['data']['t [s]'], acc_cost, label=run['label'], color=run['color'])
    plt.xlabel("Time [s]")
    plt.ylabel("Accumulated Cost")
    plt.title(f"{ctrl}: Accumulated Cost (Synthetic)")
    plt.grid()
    plt.legend()
    plt.savefig(outdir / f"{ctrl}_accumulated_cost.png")
    plt.close()


# ---------------- Main ---------------- #

def main():
    for ctrl, info in CONTROLLERS.items():
        print(f"\n📊 Processing {ctrl}")
        outdir = BASE_OUTDIR / ctrl
        outdir.mkdir(parents=True, exist_ok=True)

        data_runs = []
        for i, file_info in enumerate(info["files"]):
            file_path = Path(file_info["path"] if "path" in file_info else info["path"]) / file_info["name"]
            df = smart_read_csv(file_path)
            label = format_label(ctrl, i, file_info["params"])
            data_runs.append({"data": df, "label": label, "color": COLORS[i]})

        if ctrl == "Nominal":
            plot_trajectory(ctrl, data_runs, outdir)
            plot_tracking_error(ctrl, data_runs, outdir)
            plot_velocity(ctrl, data_runs, outdir)
            plot_angular_velocity(ctrl, data_runs, outdir)
        elif ctrl == "LQR":
            plot_control_inputs(ctrl, data_runs, outdir)
            plot_tracking_error(ctrl, data_runs, outdir)
            plot_trajectory(ctrl, data_runs, outdir)
        elif ctrl == "MPC":
            plot_accumulated_cost(ctrl, data_runs, outdir)
            plot_control_inputs(ctrl, data_runs, outdir)
            plot_tracking_error(ctrl, data_runs, outdir)
            plot_trajectory(ctrl, data_runs, outdir)

    print("\n✅ All plots saved in 'comparison_plots/<Controller>' folders.")


if __name__ == "__main__":
    main()
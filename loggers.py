"""
Logger Module for Simulation Data
==================================

This file provides logging functionality for simulations, including both console output and CSV file logging.
Supports a generic base Logger class and a concrete implementation for a 3-wheel robot (Sys3WRobotNI).
"""

import csv
from tabulate import tabulate

class Logger:
    """
    Abstract base class for data loggers.

    Subclasses must override:
        - print_sim_step: prints simulation step data to console.
        - log_data_row: writes simulation step data to CSV.
    """

    def print_sim_step(self, *args, **kwargs):
        """Prints one step of simulation data to the console (must override)."""
        raise NotImplementedError("Subclasses must implement print_sim_step")

    def log_data_row(self, *args, **kwargs):
        """Logs one row of data to a CSV file (must override)."""
        raise NotImplementedError("Subclasses must implement log_data_row")


class Logger3WRobotNI(Logger):
    """
    Logger for a 3-wheel robot (non-holonomic integrator).

    Fields logged:
        - Time (t [s])
        - x position (x [m])
        - y position (y [m])
        - Orientation angle (alpha [rad])
        - Instantaneous cost (run_obj)
        - Accumulated cost (accum_obj)
        - Linear velocity (v [m/s])
        - Angular velocity (omega [rad/s])
    """

    def print_sim_step(self, t, xCoord, yCoord, alpha, run_obj, accum_obj, action):
        """
        Prints one simulation step as a formatted table to the console.
        """
        row_header = [
            't [s]', 'x [m]', 'y [m]', 'alpha [rad]',
            'run_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]'
        ]

        row_data = [t, xCoord, yCoord, alpha, run_obj, accum_obj, action[0], action[1]]

        row_format = (
            '8.3f', '8.3f', '8.3f', '8.3f',
            '8.1f', '8.1f', '8.3f', '8.3f'
        )

        table = tabulate(
            [row_header, row_data],
            floatfmt=row_format,
            headers='firstrow',
            tablefmt='grid'
        )

        print(table)

    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, run_obj, accum_obj, action):
        """
        Appends a row of simulation data to the specified CSV file.

        Parameters:
            datafile (str): Path to CSV file.
            t (float): Time.
            xCoord (float): X-position.
            yCoord (float): Y-position.
            alpha (float): Orientation.
            run_obj (float): Instantaneous cost.
            accum_obj (float): Accumulated cost.
            action (list): Control input [v, omega].
        """
        with open(datafile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, xCoord, yCoord, alpha, run_obj, accum_obj, action[0], action[1]])

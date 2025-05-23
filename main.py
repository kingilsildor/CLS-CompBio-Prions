import time

import numpy as np
from tqdm import tqdm
from itertools import product

from config import (
    GRID_SIZE,
    GRID_SPACING,
    HEALTH_NEURON,
    SAVE_INTERVAL,
    TIME,
    TIME_SPACING,
    k_A,
    k_B,
    k_c,
    D_A,
    D_B,
)
from scripts.create_gif import create_gif
from scripts.create_plots import (
    plot_average_concentration,
    plot_concentrations,
    plot_neuron_deaths,
    plot_neuron_deaths_over_time,
    plot_prion_cell_death,
    plot_total_concentration,
    plot_param_search,
)
from scripts.data_manipulation import delete_npy, read_all_grids, read_grids_at_timestep
from source.cells import create_neuron_dict, neuron_secrete
from source.diffusion import (
    init_diffusion_eq,
    pre_diffusion,
    run_diffusion,
)
from source.grid import initialize_grid, initialize_value_grid


def initialize_simulation(
    k_A=k_A,
    k_B=k_B,
    k_c=k_c,
    D_A=D_A,
    D_B=D_B,
) -> tuple:
    """
    Initialize all simulation grids, neuron/protein/prion states, and diffusion equations.

    Params
    -------
    - k_A (float): Protein decay rate.
    - k_B (float): Prion decay rate.
    - k_c (float): Conversion rate.
    - D_A (float): Protein diffusion coefficient.
    - D_B (float): Prion diffusion coefficient.

    Returns
    --------
    - A: Initial protein concentration grid (np.ndarray)
    - B: Initial prion concentration grid (np.ndarray)
    - eqA: Diffusion equation object for protein
    - eqB: Diffusion equation object for prion
    - neuron_grid: Grid representing neuron health states (np.ndarray)
    - neuron_dict: Dictionary mapping neuron positions to their states (dict)
    - protein_grid: Grid of secreted protein concentrations (np.ndarray)
    - prion_grid: Grid of prion concentrations (np.ndarray)
    """
    dx, nx = GRID_SPACING, GRID_SIZE
    mesh, N = initialize_grid(dx=dx, nx=nx)

    neuron_grid = initialize_value_grid(N, num_items=100, value=HEALTH_NEURON)
    neuron_dict = create_neuron_dict(neuron_grid)
    protein_grid = neuron_secrete(neuron_grid, TIME_SPACING)
    protein_grid = pre_diffusion(protein_grid)
    prion_grid = initialize_value_grid(N, num_items=1, value=1)

    # Initialize diffusion equations
    A, B, eqA, eqB = init_diffusion_eq(
        mesh,
        protein_grid,
        prion_grid,
        k_A=k_A,
        k_B=k_B,
        k_c=k_c,
        D_A=D_A,
        D_B=D_B,
    )

    return A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid


def run_simulation(
    A: np.ndarray,
    B: np.ndarray,
    eqA,
    eqB,
    neuron_grid: np.ndarray,
    neuron_dict: dict,
    protein_grid: np.ndarray,
    prion_grid: np.ndarray,
) -> list:
    """
    Run the main simulation loop for protein and prion diffusion and neuron state updates.

    Params
    -------
    - A (np.ndarray): Initial protein concentration grid
    - B (np.ndarray): Initial prion concentration grid
    - eqA (DiffusionEquation): Diffusion equation object for protein
    - eqB (DiffusionEquation): Diffusion equation object for prion
    - neuron_grid (np.ndarray): Grid representing neuron health states
    - neuron_dict (dict): Dictionary mapping neuron positions to their states
    - protein_grid (np.ndarray): Grid of secreted protein concentrations
    - prion_grid (np.ndarray): Grid of prion concentrations

    Returns
    -------
    - cell_death_counter (list): Number of neurons that died at each time step
    """
    cell_death_counter = run_diffusion(
        A,
        B,
        eqA,
        eqB,
        time=TIME,
        dt=TIME_SPACING,
        neuron_grid=neuron_grid,
        neuron_dict=neuron_dict,
        protein_grid=protein_grid,
        prion_grid=prion_grid,
        save_img=True,
        save_interval=SAVE_INTERVAL,
    )
    return cell_death_counter


def plot_simulation_results(
    neuron_dict,
    cell_death_counter,
    cell_death_results,
    param_combinations,
) -> None:
    """
    Generate and save plots, GIFs, and summary statistics from simulation results.

    Params
    -------
    - neuron_dict (dict): Dictionary mapping neuron positions to their states
    - cell_death_counter (list): Number of neurons that died at each time step
    - cell_death_results (list): List of cell death counts for each parameter combination
    - param_combinations (list): List of parameter combinations used in the simulations
    """
    timepoints = list(range(0, TIME + 1, SAVE_INTERVAL))

    for timepoint in tqdm(timepoints, desc="Plotting", unit=" timepoint"):
        neuron_grid, prion_grid, protein_grid = read_grids_at_timestep(timepoint)
        plot_concentrations(protein_grid, neuron_grid, prion_grid, timepoint, TIME)
        plot_prion_cell_death(prion_grid, neuron_dict, timepoint)

    for grid_name in ["protein", "prion"]:
        if grid_name == "prion":
            set_title = True
        else:
            set_title = False
        grids = read_all_grids(grid_name)
        plot_average_concentration(grids, grid_name, timepoints, set_title)
        plot_total_concentration(grids, grid_name, timepoints, prion=set_title)

    delete_npy()

    create_gif(
        file_path="results",
        timepoints=timepoints,
    )
    create_gif(
        file_path="results",
        timepoints=timepoints,
        file_name="prion_cell_death",
    )

    # Plot neuron deaths summary
    plot_neuron_deaths(neuron_dict)
    plot_neuron_deaths_over_time(cell_death_counter, TIME)

    plot_param_search(cell_death_results, param_combinations, TIME)


def param_search(
    param_amount=12,
    step_size=0.1,
    k_A=k_A,
    k_B=k_B,
    k_c=k_c,
    D_A=D_A,
    D_B=D_B,
) -> tuple:
    """
    Perform parameter search for diffusion coefficients and decay rates.

    Params
    -------
    - param_amount (int): Number of parameter values to test. Default is 12.
    - step_size (float): Step size for parameter variation. Default is 0.1.
    - k_A (float): Protein decay rate. Default is k_A.
    - k_B (float): Prion decay rate. Default is k_B.
    - k_c (float): Conversion rate. Default is k_c.
    - D_A (float): Protein diffusion coefficient. Default is D_A.
    - D_B (float): Prion diffusion coefficient. Default is D_B.

    Returns
    -------
    - cell_death_results (list): List of cell death counts for each parameter combination.
    - param_combinations (list): List of parameter combinations used in the simulations.
    """
    kA_vals = np.linspace(k_A, k_A + step_size * (param_amount - 1), param_amount)
    kB_vals = np.linspace(k_B, k_B + step_size * (param_amount - 1), param_amount)
    kc_vals = np.linspace(k_c, k_c + step_size * (param_amount - 1), param_amount)
    DA_vals = np.linspace(D_A, D_A + step_size * (param_amount - 1), param_amount)
    DB_vals = np.linspace(D_B, D_B + step_size * (param_amount - 1), param_amount)

    kA_vals[-1] = 0.02
    kB_vals[-1] = 0.05
    kc_vals[-1] = 0.01
    DA_vals[-1] = 0.1
    DB_vals[-1] = 0.05

    param_combinations = list(zip(kA_vals, kB_vals, kc_vals, DA_vals, DB_vals))
    cell_death_results = [None] * len(param_combinations)

    for idx, (kA, kB, kc, DA, DB) in tqdm(
        enumerate(param_combinations),
        total=len(param_combinations),
        desc="Running simulations",
        unit="simulation",
    ):
        (
            A,
            B,
            eqA,
            eqB,
            neuron_grid,
            neuron_dict,
            protein_grid,
            prion_grid,
        ) = initialize_simulation(
            k_A=kA,
            k_B=kB,
            k_c=kc,
            D_A=DA,
            D_B=DB,
        )

        cell_death_counter = run_simulation(
            A,
            B,
            eqA,
            eqB,
            neuron_grid,
            neuron_dict,
            protein_grid,
            prion_grid,
        )

        cell_death_results[idx] = cell_death_counter

    return cell_death_results, param_combinations


def main():
    start_time = time.time()
    cell_death_results, param_combinations = param_search(
        param_amount=12,
        step_size=0.5,
        k_A=k_A,
        k_B=k_B,
        k_c=k_c,
        D_A=D_A,
        D_B=D_B,
    )

    A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid = (
        initialize_simulation()
    )

    cell_death_counter = run_simulation(
        A,
        B,
        eqA,
        eqB,
        neuron_grid,
        neuron_dict,
        protein_grid,
        prion_grid,
    )
    plot_simulation_results(
        neuron_dict, cell_death_counter, cell_death_results, param_combinations
    )
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

import time

import numpy as np
from tqdm import tqdm

from config import (
    GRID_SIZE,
    GRID_SPACING,
    HEALTH_NEURON,
    SAVE_INTERVAL,
    TIME,
    TIME_SPACING,
)
from scripts.create_gif import create_gif
from scripts.create_plots import (
    plot_average_concentration,
    plot_concentrations,
    plot_neuron_deaths,
    plot_prion_cell_death,
    plot_neuron_deaths_over_time,
)
from scripts.data_manipulation import delete_npy, read_all_grids, read_grids_at_timestep
from source.cells import create_neuron_dict, neuron_secrete
from source.diffusion import (
    init_diffusion_eq,
    pre_diffusion,
    run_diffusion,
)
from source.grid import initialize_grid, initialize_value_grid


def initialize_simulation() -> tuple:
    """
    Initialize all simulation grids, neuron/protein/prion states, and diffusion equations.

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
        k_A=0.02,
        k_B=0.05,
        k_c=0.01,
        D_A=0.1,
        D_B=0.05,
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


def plot_simulation_results(neuron_dict, cell_death_counter) -> None:
    """
    Generate and save plots, GIFs, and summary statistics from simulation results.

    Params
    -------
    - neuron_dict (dict): Dictionary mapping neuron positions to their states
    """
    timepoints = list(range(0, TIME + 1, SAVE_INTERVAL))

    # Plot concentrations and prion cell death at each timepoint
    for timepoint in tqdm(timepoints, desc="Plotting", unit=" timepoint"):
        neuron_grid, prion_grid, protein_grid = read_grids_at_timestep(timepoint)
        plot_concentrations(protein_grid, neuron_grid, prion_grid, timepoint, TIME)
        plot_prion_cell_death(prion_grid, neuron_dict, timepoint)

    # Plot average concentrations over time
    for grid_name in ["protein", "prion"]:
        grids = read_all_grids(grid_name)
        plot_average_concentration(grids, grid_name, timepoints)

    # Clean up intermediate files
    delete_npy()

    # Create GIFs from results
    create_gif(
        file_path="results",
        timepoints=timepoints,
    )
    create_gif(
        file_path="results",
        timepoints=timepoints,
        file_name="prion_cell_death",
        delete_img=False,
    )

    # Plot neuron deaths summary
    plot_neuron_deaths(neuron_dict)
    plot_neuron_deaths_over_time(cell_death_counter, TIME)


def main():
    start_time = time.time()
    A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid = (
        initialize_simulation()
    )

    cell_death_counter = run_simulation(
        A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid
    )
    plot_simulation_results(neuron_dict, cell_death_counter)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

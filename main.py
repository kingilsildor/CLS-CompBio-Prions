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

    Returns:
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
    A,
    B,
    eqA,
    eqB,
    neuron_grid,
    neuron_dict,
    protein_grid,
    prion_grid,
):
    """
    Run the main simulation loop for protein and prion diffusion and neuron state updates.

    Params:
    -------
    - A: Protein concentration grid (np.ndarray)
    - B: Prion concentration grid (np.ndarray)
    - eqA: Diffusion equation object for protein
    - eqB: Diffusion equation object for prion
    - neuron_grid: Grid representing neuron health states (np.ndarray)
    - neuron_dict: Dictionary mapping neuron positions to their states (dict)
    - protein_grid: Grid of secreted protein concentrations (np.ndarray)
    - prion_grid: Grid of prion concentrations (np.ndarray)
    """
    run_diffusion(
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


def plot_simulation_results(neuron_dict):
    """
    Generate and save plots, GIFs, and summary statistics from simulation results.

    Params:
    -------
    - neuron_dict (dict): Dictionary mapping neuron positions to their states
    """
    timepoints = list(range(0, TIME + 1, SAVE_INTERVAL))

    # Plot concentrations and prion cell death at each timepoint
    for timepoint in tqdm(timepoints, desc="Plotting", unit="timepoint"):
        neuron_grid, prion_grid, protein_grid = read_grids_at_timestep(timepoint)
        plot_concentrations(protein_grid, neuron_grid, prion_grid, timepoint, TIME)
        plot_prion_cell_death(protein_grid, neuron_dict, timepoint)

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
    )

    # Plot neuron deaths summary
    plot_neuron_deaths(neuron_dict)


def main():
    A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid = (
        initialize_simulation()
    )

    run_simulation(A, B, eqA, eqB, neuron_grid, neuron_dict, protein_grid, prion_grid)
    plot_simulation_results(neuron_dict)


if __name__ == "__main__":
    main()

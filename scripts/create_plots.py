import matplotlib.pyplot as plt
import numpy as np

from config import *

plt.style.use("bmh")


def normalize_diffusion(grid) -> np.ndarray:
    """
    Normalize a diffusion grid to the range [0, 1].

    Params
    -------
    - grid (np.ndarray): Input grid to normalize.

    Returns
    -------
    - grid (np.ndarray): Normalized grid.
    """
    grid -= grid.min()
    if grid.max() != 0:
        grid /= grid.max()
    return grid


def plot_concentrations(
    protein_grid, neuron_grid, prion_grid, step, total_steps
) -> None:
    """
    Plot and save the concentrations of protein, neurons, and prions at a simulation step.

    Params
    -------
    - protein_grid (np.ndarray): Protein concentration grid.
    - neuron_grid (np.ndarray): Neuron health state grid.
    - prion_grid (np.ndarray): Prion concentration grid.
    - step (int): Current simulation step.
    - total_steps (int): Total number of simulation steps.

    Returns
    -------
    - Plot of protein, neuron, and prion concentrations.
    """
    protein_grid = normalize_diffusion(protein_grid)
    prion_grid = normalize_diffusion(prion_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(protein_grid.T, origin="lower", cmap="viridis")
    axes[0].set_title("Protein concentration")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        neuron_grid.T, origin="lower", cmap="gray", vmin=DEATH_NEURON, vmax=MAX_AGE
    )
    axes[1].set_title(f"Amount of neurons: {np.sum(neuron_grid >= HEALTH_NEURON)}")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(prion_grid.T, origin="lower", cmap="plasma")
    axes[2].set_title("Prion concentration")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Simulation Step {step}/{total_steps}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"results/step_{step:03d}.png", dpi=FIG_DPI)
    plt.close()


def plot_average_concentration(grids, grid_name, timepoints) -> None:
    """
    Plot and save the total concentration of a grid over time.

    Params
    -------
    - grids (list of np.ndarray): List of grids at each timepoint.
    - grid_name (str): Name of the grid (for file naming).
    - timepoints (list): List of timepoints.

    Returns
    -------
    - Plot of total concentration over time.
    """
    total_concentration = [np.sum(grid) for grid in grids]

    plt.plot(timepoints, total_concentration, marker="o")
    plt.title("Total Concentration Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Total Concentration")

    plt.tight_layout()
    plt.savefig(f"results/total_concentration_{grid_name}.png", dpi=FIG_DPI)
    plt.close()


def plot_neuron_deaths(neuron_dict) -> None:
    """
    Plot and save the locations and causes of neuron deaths.

    Params
    -------
    - neuron_dict (dict): Dictionary of Neuron objects.

    Returns
    -------
    - Plot of neuron locations and causes of death.
    """
    death_colors = {
        None: "green",  # Alive
        "age": "blue",
        "prion": "red",
        "apoptosis": "orange",
    }
    markers = {None: "o", "age": "s", "prion": "x", "apoptosis": "D"}

    plt.figure(figsize=(8, 8))
    for neuron in neuron_dict.values():
        x, y = neuron.get_coordinates()
        cause = neuron.died
        plt.scatter(
            x,
            y,
            color=death_colors.get(cause, "gray"),
            marker=markers.get(cause, "o"),
            label=cause if cause is not None else "alive",
            s=80,
            edgecolor="k",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Death Cause")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Neuron Locations and Cause of Death")
    plt.savefig("results/neuron_deaths.png", dpi=FIG_DPI)
    plt.close()


def plot_prion_cell_death(prion_grid, neuron_dict, step) -> None:
    """
    Plot and save neuron locations and prion concentration, highlighting prion-induced deaths.

    Params
    -------
    - prion_grid (np.ndarray): Prion concentration grid.
    - neuron_dict (dict): Dictionary of Neuron objects.
    - step (int): Current simulation step.

    Returns
    -------
    - Plot of neuron locations and prion concentration.
    """
    prion_grid = normalize_diffusion(prion_grid)

    plt.figure(figsize=(8, 8))
    for neuron in neuron_dict.values():
        x, y = neuron.get_coordinates()
        if neuron.died == "prion":
            plt.scatter(x, y, color="red", marker="x", s=80, edgecolor="k")
        else:
            plt.scatter(x, y, color="green", marker="o", s=80, edgecolor="k")

    plt.imshow(prion_grid.T, origin="lower", cmap="plasma")
    plt.colorbar(label="Prion Concentration")
    plt.legend(
        ["Neuron (alive)", "Neuron (prion-induced death)"],
        loc="upper right",
        title="Neuron Status",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Neuron Locations and Prion Concentration")
    plt.savefig(f"results/prion_cell_death_{step:03d}.png", dpi=FIG_DPI)
    plt.close()


def plot_neuron_deaths_over_time(cell_death_counter, time) -> None:
    """
    Plot and save the number of neuron deaths over time.

    Params
    -------
    - cell_death_counter (list): List of neuron deaths at each time step.
    - timepoints (list): List of timepoints.

    Returns
    -------
    - Plot of neuron deaths over time.
    """
    timepoints = np.arange(0, time + 1)

    plt.hist(
        timepoints,
        bins=10,
        weights=cell_death_counter,
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.title("Neuron Deaths Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Number of Neuron Deaths")

    plt.tight_layout()
    plt.savefig("results/neuron_deaths_over_time.png", dpi=FIG_DPI)
    plt.close()

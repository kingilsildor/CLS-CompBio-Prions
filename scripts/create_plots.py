import matplotlib.pyplot as plt
import numpy as np

from config import *

plt.style.use("bmh")


def normalize_diffusion(grid):
    grid -= grid.min()
    if grid.max() != 0:
        grid /= grid.max()

    return grid


def plot_concentrations(protein_grid, neuron_grid, prion_grid, step, total_steps):
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


def plot_average_concentration(grids, grid_name, timepoints):
    avg_concentration = [np.mean(grid) for grid in grids]

    plt.plot(timepoints, avg_concentration, marker="o")
    plt.title("Average Concentration Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Average Concentration")

    plt.tight_layout()
    plt.savefig(f"results/average_concentration_{grid_name}.png", dpi=FIG_DPI)
    plt.close()


def plot_neuron_deaths(neuron_dict):
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


def plot_prion_cell_death(prion_grid, neuron_dict, step):
    plt.figure(figsize=(8, 8))
    for neuron in neuron_dict.values():
        x, y = neuron.get_coordinates()
        if neuron.died == "prion":
            plt.scatter(x, y, color="red", marker="x", s=80, edgecolor="k")
        else:
            plt.scatter(x, y, color="green", marker="o", s=80, edgecolor="k")

    plt.imshow(prion_grid.T, origin="lower", cmap="plasma")
    plt.colorbar(label="Prion Concentration")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Neuron Locations and Prion Concentration")
    plt.savefig(f"results/prion_cell_death_{step:03d}.png", dpi=FIG_DPI)
    plt.close()

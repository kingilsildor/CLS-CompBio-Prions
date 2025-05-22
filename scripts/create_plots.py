import matplotlib.pyplot as plt
import numpy as np

from config import *


def plot_diffusion(grid):
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Concentration")
    plt.title("Diffusion Simulation")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def plot_diffusion_overlay(grid1, grid2):
    plt.imshow(grid1, cmap="viridis", interpolation="nearest", alpha=0.5)

    masked_grid2 = np.ma.masked_less(grid2, HEALTH_NEURON)
    plt.imshow(masked_grid2, cmap="plasma", interpolation="nearest", alpha=0.9)

    plt.colorbar(label="Concentration")
    plt.title(f"Overlay: Only grid2 Values ≥ {HEALTH_NEURON}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

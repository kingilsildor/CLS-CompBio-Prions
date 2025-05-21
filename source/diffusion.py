import matplotlib.pyplot as plt
import numpy as np
from fipy import CellVariable, DiffusionTerm, TransientTerm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from config import *
from source.cells import neuron_secrete, prion_cell_death


def pre_diffusion(init_grid):
    mask_main = init_grid == SECRETED_VALUE
    mask_diagonal = init_grid == SECRETED_VALUE / 2

    dist_main = distance_transform_edt(~mask_main)
    dist_diagonal = distance_transform_edt(~mask_diagonal)

    power = 1 / 5

    grad_main = np.where(mask_main, 1.0, 1 / np.power((1 + dist_main), power))
    grad_diagonal = np.where(
        mask_diagonal, 1.0, 1 / np.power((1 + dist_diagonal), power)
    )
    combined_gradient = grad_main + grad_diagonal
    combined_gradient /= np.max(combined_gradient)

    return combined_gradient


def set_boundary_conditions(mesh, A, B):
    for faceGroup in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
        A.faceGrad.constrain([0.0], faceGroup)
        B.faceGrad.constrain([0.0], faceGroup)


def set_equations(A, B, k_A, k_B, k_c, D_A, D_B):
    eqA = TransientTerm(var=A) == -(k_A * A) - (k_c * A * B) + DiffusionTerm(
        coeff=D_A, var=A
    )
    eqB = TransientTerm(var=B) == (k_c * A * B) - (k_B * B) + DiffusionTerm(
        coeff=D_B, var=B
    )

    return eqA, eqB


def init_diffusion_eq(mesh, protein_grid, prion_grid, k_A, k_B, k_c, D_A, D_B, dx):
    D = max(D_A, D_B)
    delta_t = 0.5 * (1 / (D * (1 / dx**2 + 1 / dx**2)))

    A = CellVariable(name="A", mesh=mesh, value=protein_grid.flatten(), hasOld=True)
    B = CellVariable(name="B", mesh=mesh, value=prion_grid.flatten(), hasOld=True)

    set_boundary_conditions(mesh, A, B)
    eqA, eqB = set_equations(A, B, k_A, k_B, k_c, D_A, D_B)

    return A, B, eqA, eqB, delta_t


def save_image(protein_grid, neuron_grid, prion_grid, step, total_steps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(protein_grid.T, origin="lower", cmap="viridis", vmin=0)
    axes[0].set_title("Protein concentration")
    fig.colorbar(im0, ax=axes[0])

    red_pos = (DEATH_NEURON - DEATH_MIN) / (MAX_AGE - DEATH_MIN)
    white_pos = (0 - DEATH_MIN) / (MAX_AGE - DEATH_MIN)

    colors = [
        (0.0, "darkred"),
        (red_pos * 0.75, "red"),
        (white_pos, "white"),
        (1.0, "blue"),
    ]

    custom_cmap = LinearSegmentedColormap.from_list("red_white_blue_stretched", colors)
    norm = Normalize(vmin=DEATH_MIN, vmax=MAX_AGE)
    im1 = axes[1].imshow(neuron_grid.T, origin="lower", cmap=custom_cmap, norm=norm)
    axes[1].set_title("Neuron distribution")

    cbar = fig.colorbar(im1, ax=axes[1])
    cbar.set_label("Neuron age")
    cbar.set_ticks([-1, 50, 100, 150, 200])

    im2 = axes[2].imshow(prion_grid.T, origin="lower", cmap="plasma", vmin=0)
    axes[2].set_title("Prion concentration")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Simulation Step {step}/{total_steps - 1}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"results/step_{step:03d}.png", dpi=FIG_DPI)
    plt.close()


def run_diffusion(
    A,
    B,
    eqA,
    eqB,
    steps,
    dt,
    nx,
    neuron_grid,
    neuron_dict,
    protein_grid,
    prion_grid,
    save_img=True,
    save_interval=10,
):
    for step in tqdm(range(steps)):
        if save_img and step % save_interval == 0:
            save_image(
                protein_grid,
                neuron_grid,
                prion_grid,
                step,
                steps,
            )

        A.updateOld()
        B.updateOld()

        eqA.solve(var=A, dt=dt)
        eqB.solve(var=B, dt=dt)

        A.value += neuron_secrete(neuron_grid).flatten()
        protein_grid = A.value.reshape((nx, nx))
        prion_grid = B.value.reshape((nx, nx))

        for neuron in neuron_dict.values():
            if neuron.alive:
                neuron.age_cell()
            coords = neuron.get_coordinates()
            neuron_grid[int(coords[0]), int(coords[1])] = neuron.get_age()

        neuron_grid = prion_cell_death(prion_grid, neuron_grid, neuron_dict)

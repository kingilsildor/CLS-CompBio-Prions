import matplotlib.pyplot as plt
import numpy as np
from fipy import CellVariable, DiffusionTerm, TransientTerm
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from tqdm import tqdm

from config import *
from source.cells import neuron_secrete, prion_cell_death


def make_diffusion_kernel(size: int, sigma_scale: float = 0.3) -> np.ndarray:
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, center] = 1.0

    sigma = size * sigma_scale
    kernel = gaussian_filter(kernel, sigma=sigma)

    kernel /= kernel.sum()

    return kernel


def make_diffusion_gradient(init_grid, diffusion_power, scaling_factor):
    mask_main = init_grid == SECRETED_VALUE
    mask_diagonal = init_grid == SECRETED_VALUE / 2

    dist_main = distance_transform_edt(~mask_main)
    dist_diagonal = distance_transform_edt(~mask_diagonal)

    grad_main = np.where(mask_main, 1.0, 1 / np.power((1 + dist_main), diffusion_power))
    grad_diagonal = np.where(
        mask_diagonal, 1.0, 1 / np.power((1 + dist_diagonal), diffusion_power)
    )
    combined_gradient = grad_main + grad_diagonal

    combined_gradient *= scaling_factor
    combined_gradient /= np.max(combined_gradient)

    return combined_gradient


def pre_diffusion(init_grid, diffusion_power=1 / 5, scaling_factor=2, kernel_size=9):
    kernel = make_diffusion_kernel(kernel_size)
    gradient = make_diffusion_gradient(init_grid, diffusion_power, scaling_factor)
    weighted_grid = init_grid * gradient
    diffused = convolve(weighted_grid, kernel, mode="reflect")

    return diffused


def normalize_diffusion(grid):
    grid -= grid.min()
    if grid.max() != 0:
        grid /= grid.max()

    return grid


def set_boundary_conditions(mesh, A, B):
    for faceGroup in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
        A.faceGrad.constrain([0.0], faceGroup)
        B.faceGrad.constrain([0.0], faceGroup)


def set_equations(A, B, k_A, k_B, k_c, D_A, D_B, chi=0.5):
    eqA = TransientTerm(var=A) == -(k_A * A) - (k_c * A * B) + DiffusionTerm(
        coeff=D_A, var=A
    )
    eqB = TransientTerm(var=B) == (k_c * A * B) - (k_B * B) + DiffusionTerm(
        coeff=D_B, var=B
    ) - DiffusionTerm(coeff=chi * B, var=A)

    return eqA, eqB


def init_diffusion_eq(mesh, protein_grid, prion_grid, k_A, k_B, k_c, D_A, D_B):
    A = CellVariable(name="A", mesh=mesh, value=protein_grid.flatten(), hasOld=True)
    B = CellVariable(name="B", mesh=mesh, value=prion_grid.flatten(), hasOld=True)

    set_boundary_conditions(mesh, A, B)

    eqA, eqB = set_equations(A, B, k_A, k_B, k_c, D_A, D_B)

    return A, B, eqA, eqB


def save_image(protein_grid, neuron_grid, prion_grid, step, total_steps):
    # protein_grid = normalize_diffusion(protein_grid)
    # prion_grid = normalize_diffusion(prion_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(protein_grid.T, origin="lower", cmap="viridis")
    axes[0].set_title("Protein concentration")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(neuron_grid.T, origin="lower", cmap="gray")
    axes[1].set_title("Amount of neurons")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(prion_grid.T, origin="lower", cmap="plasma")
    axes[2].set_title("Prion concentration")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Simulation Step {step}/{total_steps - 1}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"results/step_{step:03d}.png", dpi=300)
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
    prion_grid,
    save_img=True,
    save_interval=10,
):
    for step in tqdm(range(steps)):
        A.updateOld()
        B.updateOld()

        eqA.solve(var=A, dt=dt)
        eqB.solve(var=B, dt=dt)

        A.value += neuron_secrete(neuron_grid, dt).flatten()
        protein_grid = A.value.reshape((nx, nx))

        prion_grid = B.value.reshape((nx, nx))
        prion_cell_death(prion_grid, neuron_grid, neuron_dict)

        if save_img and step % save_interval == 0:
            save_image(protein_grid, neuron_grid, prion_grid, step, steps)

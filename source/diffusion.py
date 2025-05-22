import numpy as np
from fipy import CellVariable, DiffusionTerm, TransientTerm
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from tqdm import tqdm

from config import *
from scripts.data_manipulation import write_grid
from source.cells import neuron_secrete


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


def set_boundary_conditions(mesh, A, B):
    for faceGroup in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
        A.faceGrad.constrain([0.0], faceGroup)
        B.faceGrad.constrain([0.0], faceGroup)


def set_equations(A, B, k_A, k_B, k_c, D_A, D_B, chi):  # Increased default chi
    eqA = TransientTerm(var=A) == -(k_A * A) - (k_c * A * B) + DiffusionTerm(
        coeff=D_A, var=A
    )
    eqB = TransientTerm(var=B) == (k_c * A * B) - (k_B * B) + DiffusionTerm(
        coeff=D_B, var=B
    ) - DiffusionTerm(coeff=chi * B, var=A)

    return eqA, eqB


def init_diffusion_eq(mesh, protein_grid, prion_grid, k_A, k_B, k_c, D_A, D_B, chi=10):
    A = CellVariable(name="A", mesh=mesh, value=protein_grid.flatten(), hasOld=True)
    B = CellVariable(name="B", mesh=mesh, value=prion_grid.flatten(), hasOld=True)

    set_boundary_conditions(mesh, A, B)

    eqA, eqB = set_equations(A, B, k_A, k_B, k_c, D_A, D_B, chi=chi)

    return A, B, eqA, eqB


def run_diffusion(
    A,
    B,
    eqA,
    eqB,
    time,
    dt,
    neuron_grid,
    neuron_dict,
    protein_grid,
    prion_grid,
    save_img=True,
    save_interval=10,
):
    time = time + 1
    for step in tqdm(range(time), desc="Running Diffusion", unit="step"):
        if save_img and step % save_interval == 0:
            write_grid(protein_grid, "protein", step)
            write_grid(prion_grid, "prion", step)
            write_grid(neuron_grid, "neuron", step)

        A.updateOld()
        B.updateOld()

        eqA.solve(var=A, dt=dt)
        eqB.solve(var=B, dt=dt)

        A.value += neuron_secrete(neuron_grid, dt).flatten()
        protein_grid = A.value.reshape((GRID_SIZE, GRID_SIZE))
        prion_grid = B.value.reshape((GRID_SIZE, GRID_SIZE))

        for neuron in neuron_dict.values():
            if neuron.alive:
                neuron.age_cell()
                neuron.prion_cell_death(prion_grid)
            coords = neuron.get_coordinates()
            neuron_grid[int(coords[0]), int(coords[1])] = neuron.get_age()

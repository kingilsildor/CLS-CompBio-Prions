import numpy as np
from fipy import CellVariable, DiffusionTerm, TransientTerm
from fipy.meshes.uniformGrid2D import UniformGrid2D
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from tqdm import tqdm

from config import *
from scripts.data_manipulation import write_grid
from source.cells import neuron_secrete


def make_diffusion_kernel(
    size: int,
    sigma_scale: float = 0.3,
) -> np.ndarray:
    """
    Create a normalized 2D Gaussian kernel for diffusion.

    Params
    -------
    - size (int): Size of the kernel (will be made odd if even).
    - sigma_scale (float, optional): Scaling factor for Gaussian sigma. Defaults to 0.3.

    Returns
    -------
    - kernel (np.ndarray): Normalized 2D Gaussian kernel.
    """
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, center] = 1.0

    sigma = size * sigma_scale
    kernel = gaussian_filter(kernel, sigma=sigma)

    kernel /= kernel.sum()

    return kernel


def make_diffusion_gradient(
    init_grid: np.ndarray,
    diffusion_power: float,
    scaling_factor: float,
) -> np.ndarray:
    """
    Create a diffusion gradient based on initial grid and parameters.

    Params
    -------
    - init_grid (np.ndarray): Initial grid for gradient calculation.
    - diffusion_power (float): Power for distance-based decay.
    - scaling_factor (float): Scaling factor for gradient.

    Returns
    -------
    - combined_gradient (np.ndarray): Combined diffusion gradient.
    """
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


def pre_diffusion(
    init_grid: np.ndarray,
    diffusion_power: float = 1 / 5,
    scaling_factor: float = 2,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Apply pre-diffusion to an initial grid using a Gaussian kernel and gradient.

    Params
    -------
    - init_grid (np.ndarray): Initial grid to diffuse.
    - diffusion_power (float, optional): Power for gradient decay. Defaults to 1/5.
    - scaling_factor (float, optional): Gradient scaling. Defaults to 2.
    - kernel_size (int, optional): Size of Gaussian kernel. Defaults to 9.

    Returns
    -------
    - diffused (np.ndarray): Diffused grid.
    """
    kernel = make_diffusion_kernel(kernel_size)
    gradient = make_diffusion_gradient(init_grid, diffusion_power, scaling_factor)
    weighted_grid = init_grid * gradient
    diffused = convolve(weighted_grid, kernel, mode="reflect")

    return diffused


def set_boundary_conditions(
    mesh: UniformGrid2D,
    A: CellVariable,
    B: CellVariable,
) -> None:
    """
    Set zero-gradient (Neumann) boundary conditions for protein and prion variables.

    Params
    -------
    - mesh (UniformGrid2D): FiPy mesh object.
    - A (CellVariable): Protein concentration variable.
    - B (CellVariable): Prion concentration variable.
    """
    for faceGroup in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
        A.faceGrad.constrain([0.0], faceGroup)
        B.faceGrad.constrain([0.0], faceGroup)


def set_equations(
    A: CellVariable,
    B: CellVariable,
    k_A: float,
    k_B: float,
    k_c: float,
    D_A: float,
    D_B: float,
    chi: float,
) -> tuple:
    """
    Define the coupled diffusion-reaction equations for protein and prion.

    Params
    -------
    - A (CellVariable): Protein concentration variable.
    - B (CellVariable): Prion concentration variable.
    - k_A, k_B, k_c (float): Reaction rate constants.
    - D_A, D_B (float): Diffusion coefficients.
    - chi (float): Cross-diffusion coefficient.

    Returns
    -------
    - eqA, eqB: FiPy equation objects for protein and prion.
    """
    eqA = TransientTerm(var=A) == -(k_A * A) - (k_c * A * B) + DiffusionTerm(
        coeff=D_A, var=A
    )
    eqB = TransientTerm(var=B) == (k_c * A * B) - (k_B * B) + DiffusionTerm(
        coeff=D_B, var=B
    ) - DiffusionTerm(coeff=chi * B, var=A)

    return eqA, eqB


def init_diffusion_eq(
    mesh: UniformGrid2D,
    protein_grid: np.ndarray,
    prion_grid: np.ndarray,
    k_A: float,
    k_B: float,
    k_c: float,
    D_A: float,
    D_B: float,
    chi: float = 100.0,
):
    """
    Initialize FiPy CellVariables and equations for protein and prion diffusion.

    Params
    -------
    - mesh (UniformGrid2D): FiPy mesh object.
    - protein_grid (np.ndarray): Initial protein concentration grid.
    - prion_grid (np.ndarray): Initial prion concentration grid.
    - k_A, k_B, k_c: Reaction rate constants.
    - D_A, D_B: Diffusion coefficients.
    - chi (float, optional): Cross-diffusion coefficient. Defaults to 10.

    Returns
    -------
    - A, B: FiPy CellVariables for protein and prion.
    - eqA, eqB: FiPy equation objects for protein and prion.
    """
    A = CellVariable(name="A", mesh=mesh, value=protein_grid.flatten(), hasOld=True)
    B = CellVariable(name="B", mesh=mesh, value=prion_grid.flatten(), hasOld=True)

    set_boundary_conditions(mesh, A, B)

    eqA, eqB = set_equations(A, B, k_A, k_B, k_c, D_A, D_B, chi=chi)

    return A, B, eqA, eqB


def run_diffusion(
    A: CellVariable,
    B: CellVariable,
    eqA,
    eqB,
    time: int,
    dt: float,
    neuron_grid: np.ndarray,
    neuron_dict: dict,
    protein_grid: np.ndarray,
    prion_grid: np.ndarray,
    sim_amount: int = 5,
    save_img: bool = True,
    save_interval: int = 10,
):
    """
    Run the main diffusion simulation loop for protein and prion concentrations.

    Params
    -------
    - A (CellVariable): Protein concentration variable.
    - B (CellVariable): Prion concentration variable.
    - eqA (BinaryTerm): Diffusion equation for protein.
    - eqB (BinaryTerm): Diffusion equation for prion.
    - time (int): Number of simulation steps.
    - dt (float): Time step size.
    - neuron_grid (np.ndarray): Grid representing neuron health states.
    - neuron_dict (dict): Dictionary mapping neuron positions to their states.
    - protein_grid (np.ndarray): Protein concentration grid.
    - prion_grid (np.ndarray): Prion concentration grid.
    - sim_amount (int, optional): Number of simulation repetitions per step. Defaults to 5.
    - save_img (bool, optional): Whether to save grid images. Defaults to True.
    - save_interval (int, optional): Interval for saving images. Defaults to 10.

    """
    time = time + 1
    cell_death_counter = np.zeros(time, dtype=int)

    for step in tqdm(range(time), desc="Running Diffusion", unit=" step"):
        protein_accumulator = np.zeros((GRID_SIZE, GRID_SIZE))
        prion_accumulator = np.zeros((GRID_SIZE, GRID_SIZE))
        neuron_accumulator = np.zeros((GRID_SIZE, GRID_SIZE))
        cell_death_sum = 0

        for _ in range(sim_amount):
            A_temp = A.copy()
            B_temp = B.copy()

            A_temp.updateOld()
            B_temp.updateOld()

            eqA.solve(var=A_temp, dt=dt)
            eqB.solve(var=B_temp, dt=dt)

            A_temp.value += neuron_secrete(neuron_grid, dt).flatten()
            protein_grid = A_temp.value.reshape((GRID_SIZE, GRID_SIZE))
            prion_grid = B_temp.value.reshape((GRID_SIZE, GRID_SIZE))

            copy_neuron_dict = {k: v.copy() for k, v in neuron_dict.items()}
            temp_neuron_grid = np.copy(neuron_grid)

            for neuron in copy_neuron_dict.values():
                if neuron.alive:
                    cell_death_sum += neuron.age_cell(
                        temp_neuron_grid, copy_neuron_dict
                    )
                    cell_death_sum += neuron.prion_cell_death(
                        prion_grid, temp_neuron_grid, copy_neuron_dict
                    )
                coords = neuron.get_coordinates()
                temp_neuron_grid[int(coords[0]), int(coords[1])] = neuron.get_age()

            protein_accumulator += protein_grid
            prion_accumulator += prion_grid
            neuron_accumulator += temp_neuron_grid

        protein_grid = protein_accumulator / sim_amount
        prion_grid = prion_accumulator / sim_amount
        neuron_grid = neuron_accumulator / sim_amount
        cell_death_counter[step] = cell_death_sum / sim_amount

        if save_img and step % save_interval == 0:
            write_grid(protein_grid, "protein", step)
            write_grid(prion_grid, "prion", step)
            write_grid(neuron_grid, "neuron", step)

        neuron_dict.update(copy_neuron_dict)
    return cell_death_counter

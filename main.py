import numpy as np

from config import *
from source.cells import create_neuron_dict, neuron_secrete
from source.diffusion import init_diffusion_eq, pre_diffusion, run_diffusion
from source.grid import initialize_grid, initialize_value_grid

dx, nx = 1.0, 100
steps = 200
mesh, N = initialize_grid(dx=dx, nx=nx)
print(f"Mesh: {mesh.shape}")

neuron_grid = initialize_value_grid(N, num_items=100, value=HEALTH_NEURON)
print(f"Neuron grid: {neuron_grid.shape}")
neuron_dict = create_neuron_dict(neuron_grid)
print(f"Neuron dict: {len(neuron_dict)} neurons")
protein_grid = neuron_secrete(neuron_grid)
print(f"Protein grid: {protein_grid.shape} Before diffusion")
protein_grid = pre_diffusion(mesh, protein_grid, D_A=5, steps=int(steps / 2), dt=0.2)
print(f"Protein grid: {protein_grid.shape} After diffusion")

prion_grid = initialize_value_grid(N, num_items=1, value=1)
print(f"Prion grid: {prion_grid.shape}, with {np.sum(prion_grid == 1)} prion")

A, B, eqA, eqB, delta_t = init_diffusion_eq(
    mesh,
    protein_grid,
    prion_grid,
    k_A=0.02,
    k_B=0.05,
    k_c=0.01,
    D_A=0.1,
    D_B=0.05,
    dx=dx,
)
print("Diffusion equations initialized")
print(f"A: {A.shape}, B: {B.shape}")

run_diffusion(
    A,
    B,
    eqA,
    eqB,
    steps=steps,
    dt=delta_t,
    nx=nx,
    neuron_grid=neuron_grid,
    neuron_dict=neuron_dict,
    prion_grid=prion_grid,
    save_img=True,
    save_interval=10,
)

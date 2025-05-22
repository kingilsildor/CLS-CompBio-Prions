from config import *
from scripts.create_gif import create_gif
from source.cells import create_neuron_dict, neuron_secrete
from source.diffusion import (
    init_diffusion_eq,
    normalize_diffusion,
    pre_diffusion,
    run_diffusion,
)
from source.grid import initialize_grid, initialize_value_grid

mesh, N = initialize_grid(dx=GRID_SPACING, nx=GRID_SIZE)
print(f"Mesh: {mesh.shape}")

neuron_grid = initialize_value_grid(N, num_items=NUMBER_OF_NEURONS, value=HEALTH_NEURON)
print(f"Neuron grid: {neuron_grid.shape}")
neuron_dict = create_neuron_dict(neuron_grid)
print(f"Neuron dict: {len(neuron_dict)} neurons")

protein_grid = neuron_secrete(neuron_grid)
protein_grid = pre_diffusion(protein_grid)
protein_grid = normalize_diffusion(protein_grid)
print(f"Protein grid initialized: {protein_grid.shape}")


prion_grid = initialize_value_grid(N, num_items=1, value=1)
print(f"Prion grid initialized: {prion_grid.shape}")

A, B, eqA, eqB = init_diffusion_eq(
    mesh,
    protein_grid,
    prion_grid,
    k_A=k_A,
    k_B=k_B,
    k_c=k_c,
    D_A=D_A,
    D_B=D_B,
)
print("Diffusion equations initialized")

run_diffusion(
    A,
    B,
    eqA,
    eqB,
    steps=int(TIME / TIME_SPACING) + 1,
    dt=TIME_SPACING,
    neuron_grid=neuron_grid,
    neuron_dict=neuron_dict,
    protein_grid=protein_grid,
    prion_grid=prion_grid,
    save_img=True,
    save_interval=SAVE_INTERVAL,
)

create_gif(
    file_path="results",
    timepoints=list(range(0, int(TIME / TIME_SPACING) + 1, SAVE_INTERVAL)),
    delete_img=False,
    duration=GIF_DURATION,
)

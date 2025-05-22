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

dx, nx = TIME_SPACING, GRID_SIZE
steps = TIME
mesh, N = initialize_grid(dx=dx, nx=nx)

neuron_grid = initialize_value_grid(N, num_items=100, value=HEALTH_NEURON)
neuron_dict = create_neuron_dict(neuron_grid)
protein_grid = neuron_secrete(neuron_grid)
protein_grid = pre_diffusion(protein_grid)
protein_grid = normalize_diffusion(protein_grid)

prion_grid = initialize_value_grid(N, num_items=1, value=1)

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

run_diffusion(
    A,
    B,
    eqA,
    eqB,
    steps=TIME + 1,
    dt=TIME_SPACING,
    nx=nx,
    neuron_grid=neuron_grid,
    neuron_dict=neuron_dict,
    prion_grid=prion_grid,
    save_img=True,
    save_interval=10,
)

create_gif(
    file_path="results",
    timepoints=list(range(0, TIME + 1, SAVE_INTERVAL)),
    delete_img=False,
)

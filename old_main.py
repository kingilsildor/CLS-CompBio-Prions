import matplotlib.pyplot as plt
import numpy as np
from cells import Neuron, neuron_secrete, prion_cell_death
from fipy import CellVariable, DiffusionTerm, TransientTerm
from grid import initialize_grid, initialize_value_grid

from config import *

# Grid parameters
dx, nx = 1.0, 100
mesh, N = initialize_grid(dx=dx, nx=nx)
neuron_grid = initialize_value_grid(N, num_items=100, value=HEALTH_NEURON)
prion_grid = initialize_value_grid(N, num_items=100, value=1)
prion_grid = np.full(N, 10.0)

neuron_grid = neuron_grid.reshape((nx, nx))
neuron_coords = np.argwhere(neuron_grid == 1)

ages = np.random.normal(
    loc=MAX_AGE * MEAN_AGE_FACTOR,
    scale=MAX_AGE * STD_AGE_FACTOR,
    size=len(neuron_coords),
)
ages = [min(MAX_AGE, max(0, int(round(age)))) for age in ages]

for coords in neuron_coords:
    x, y = coords
    neuron = Neuron(x, y, age=ages.pop(0))
init_grid = neuron_secrete(neuron_grid)

plt.imshow(
    init_grid,
    origin="lower",
    cmap="viridis",
)

# Kill the neurons if the concentration is above the threshold
neuron_dict = None
death_coords, neuron_grid = prion_cell_death(
    prion_grid.reshape((nx, nx)), neuron_grid.reshape((nx, nx)), neuron_dict
)
results = [neuron_dict[tuple(coord)]().die() for coord in coords]


# Parameters
k_A = 0.02
k_B = 0.05
k_c = 0.01
D_A = 0.1
D_B = 0.05

beta = 0.1
N_cells = 10

# Finite volume solver parameters
D = max(D_A, D_B)
delta_t = 0.5 * (1 / (D * (1 / dx**2 + 1 / dx**2)))

# Initialize cell values
A = CellVariable(name="A", mesh=mesh, value=0, hasOld=True)
B = CellVariable(name="B", mesh=mesh, value=prion_grid, hasOld=True)

# Set boundary conditions
for faceGroup in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
    A.faceGrad.constrain([0.0], faceGroup)
    B.faceGrad.constrain([0.0], faceGroup)


eqA = TransientTerm(var=A) == (beta * N_cells) - (k_A * A) - (
    k_c * A * B
) + DiffusionTerm(coeff=D_A, var=A)
eqB = TransientTerm(var=B) == (k_c * A * B) - (k_B * B) + DiffusionTerm(
    coeff=D_B, var=B
)

dt = 0.1
steps = 200

# A_init = CellVariable(name="A_init", mesh=mesh, value=init_grid.flatten(), hasOld=True)
# A_diff_only = TransientTerm(var=A_init) == DiffusionTerm(coeff=D_A, var=A_init)

# for _ in range(steps):
#     A_init.updateOld()
#     A_diff_only.solve(var=A_init, dt=dt)

# plt.imshow(
#     np.array(A_init.value).reshape((nx, nx)).T,
#     origin="lower",
#     cmap="viridis",
# )
# plt.show()


# for step in range(steps):
#     A.updateOld()
#     B.updateOld()

#     eqA.solve(var=A, dt=dt)
#     eqB.solve(var=B, dt=dt)

#     if step % 10 == 0:
#         plt.figure(figsize=(10, 4))
#         plt.imshow(
#             np.array(A.value).reshape((nx, nx)).T,
#             origin="lower",
#             cmap="viridis",
#             # vmin=0,
#             # vmax=2,
#         )
#         plt.title(f"Step {step}")
#         plt.colorbar()
#         plt.savefig(f"reaction_diffusion_images/step_{step}.png")

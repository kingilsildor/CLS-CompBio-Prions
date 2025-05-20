import matplotlib.pyplot as plt
import numpy as np
from fipy import CellVariable, DiffusionTerm, TransientTerm
from grid import create_value_gradient, initialize_grid, initialize_value_grid

# Grid parameters
dx, nx = 1.0, 100
mesh, N = initialize_grid(dx=dx, nx=nx)
neuron_grid = initialize_value_grid(N, num_items=10, value=1)
prion_grid = initialize_value_grid(N, num_items=1, value=1)

gradient = create_value_gradient(neuron_grid, value=1, decay_rate=0.5)

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
A = CellVariable(name="A", mesh=mesh, value=10.0, hasOld=True)
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
steps = 100

for step in range(steps):
    A.updateOld()
    B.updateOld()

    eqA.solve(var=A, dt=dt)
    eqB.solve(var=B, dt=dt)

    if step % 10 == 0:
        plt.figure(figsize=(10, 4))
        plt.imshow(
            np.array(A.value).reshape((nx, nx)).T,
            origin="lower",
            cmap="viridis",
            # vmin=0,
            # vmax=2,
        )
        plt.title(f"Step {step}")
        plt.colorbar()
        plt.savefig(f"reaction_diffusion_images/step_{step}.png")

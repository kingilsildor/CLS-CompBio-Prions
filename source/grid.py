import numpy as np
from fipy import (
    Grid2D,
)

from config import *


def initialize_grid(dx=1.0, nx=100):
    dy = dx
    ny = nx
    N = nx * ny

    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    return mesh, N


def initialize_value_grid(N, num_items=1, value=1.0):
    random_index = np.random.randint(0, N, num_items)
    grid = np.zeros(N)
    grid[random_index] = value
    return grid

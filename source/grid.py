import numpy as np
from fipy import Grid2D

from config import *


def initialize_grid(
    dx: float = 1.0,
    nx: int = 100,
):
    """
    Initialize a 2D square grid mesh using FiPy's Grid2D.

    Params
    -------
    - dx (float, optional): Grid spacing in both x and y directions. Defaults to 1.0.
    - nx (int, optional): Number of grid points in x (and y) direction. Defaults to 100.

    Returns
    -------
    - mesh (fipy.meshes.uniformGrid2D.Grid2D): The generated 2D grid mesh.
    - N (int): Total number of grid points (nx * ny).
    """
    dy = dx
    ny = nx
    N = nx * ny

    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    return mesh, N


def initialize_value_grid(
    N: int,
    num_items: int = 1,
    value: float = 1.0,
):
    """
    Initialize a 2D grid of zeros and set a specified number of random positions to a given value.

    Params
    -------
    - N (int): Total number of grid points (should be a perfect square).
    - num_items (int, optional): Number of grid points to set to 'value'. Defaults to 1.
    - value (float, optional): Value to assign to the selected grid points. Defaults to 1.0.

    Returns
    -------
    - grid (np.ndarray): 2D numpy array of shape (sqrt(N), sqrt(N)) with 'num_items' entries set to 'value'.
    """
    random_index = np.random.randint(0, N, num_items)
    grid = np.zeros(N)
    grid[random_index] = value
    grid = grid.reshape((int(np.sqrt(N)), int(np.sqrt(N))))
    return grid

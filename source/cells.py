import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import convolve2d

from config import *


class Neuron:
    def __init__(self, x, y, age=0):
        self.x = x
        self.y = y
        self.alive = True
        self.age = age  # in timesteps

    def die(self):
        self.alive = False
        return 0

    def age_cell(self) -> int:
        self.age += 1

        P_apop = 1 / (1 + np.exp((-9.19 / MAX_AGE) * (self.age - 0.5 * MAX_AGE)))
        if 0 < self.age < MAX_AGE and np.random.rand() > P_apop:
            self.die()
            return 0
        if self.age >= MAX_AGE:
            self.die()
            return 0
        return 1

    def get_index(self, nx):
        i = int(self.x)
        j = int(self.y)
        return i + j * nx

    def get_coordinates(self):
        return self.x, self.y

    def get_age(self):
        return self.age


def neuron_secrete(grid):
    mask = grid == HEALTH_NEURON

    full_structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)

    cardinal_structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    full_dilation = binary_dilation(mask, structure=full_structure)
    cardinal_dilation = binary_dilation(mask, structure=cardinal_structure)

    diagonal_mask = full_dilation & ~cardinal_dilation & ~mask
    cardinal_mask = cardinal_dilation & ~mask

    diagonal_mask &= grid == 0
    cardinal_mask &= grid == 0

    updated_grid = np.zeros_like(grid)
    updated_grid[cardinal_mask] = SECRETED_VALUE
    updated_grid[diagonal_mask] = SECRETED_VALUE / 2  # Half value on diagonals

    return updated_grid


def prion_cell_death(prion_grid, cell_grid, neuron_dict):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    neighbor_prion_sum = convolve2d(
        prion_grid, kernel, mode="same", boundary="fill", fillvalue=0
    )
    print(neighbor_prion_sum)

    death_mask = (neighbor_prion_sum > PRION_THRESHOLD) & (cell_grid != 0)
    death_coords = np.argwhere(death_mask)
    return death_coords

import numpy as np
from scipy.ndimage import binary_dilation

from config import *


class Neuron:
    def __init__(self, x, y, age=0):
        self.x = x
        self.y = y
        self.alive = True
        self.age = age  # in timesteps

    def die(self):
        self.alive = False
        self.age = DEATH_NEURON
        return 0

    def age_cell(self):
        self.age += 1

        P_apop = P_apop = GAMMA * np.exp(DELTA * self.age)

        if MIN_AGE < self.age < MAX_AGE and np.random.rand() < P_apop:
            print(
                f"Neuron at ({self.x}, {self.y}) died due to apoptosis. Age: {self.age}, P_apop: {P_apop}"
            )
            self.die()
        if self.age >= MAX_AGE:
            print(f"Neuron at ({self.x}, {self.y}) died due to age.")
            self.die()

    def get_index(self, nx):
        i = int(self.x)
        j = int(self.y)
        return i + j * nx

    def get_coordinates(self):
        return self.x, self.y

    def get_age(self):
        return self.age


def neuron_secrete(grid):
    mask = grid >= HEALTH_NEURON

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


def get_prion_neighbors(prion_grid):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    padded_prion = np.pad(prion_grid, pad_width=1, mode="constant", constant_values=0)

    return kernel, padded_prion


def prion_cell_death(neuron, prion_grid, neuron_grid):
    x, y = neuron.get_coordinates()
    rows, cols = prion_grid.shape

    x_min = max(0, x - 1)
    x_max = min(rows, x + 1)
    y_min = max(0, y - 1)
    y_max = min(cols, y + 1)

    neighborhood = prion_grid[x_min:x_max, y_min:y_max].copy()

    if (
        0 <= x - x_min < neighborhood.shape[0]
        and 0 <= y - y_min < neighborhood.shape[1]
    ):
        neighborhood[x - x_min, y - y_min] = 0
        prion_sum = np.sum(neighborhood)

        if prion_sum > PRION_THRESHOLD:
            print(
                f"Neuron at ({x}, {y}) died due to prion accumulation. Prion concentration: {prion_sum}"
            )
            neuron.die()


def neuron_age(N_neurons):
    ages = np.random.exponential(scale=EXP_SCALE, size=N_neurons)
    ages = [min(MAX_AGE, max(1, int(round(age)))) for age in ages]
    return ages


def create_neuron_dict(neuron_grid):
    neuron_dict = {}
    neuron_coords = np.argwhere(neuron_grid == 1)
    ages = neuron_age(len(neuron_coords))

    for coords in neuron_coords:
        x, y = coords
        neuron = Neuron(x, y, age=ages.pop(0))
        neuron_grid[x, y] = neuron.get_age()
        neuron_dict[tuple(coords)] = neuron
    return neuron_dict

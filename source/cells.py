import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from config import *


class Neuron:
    def __init__(self, x, y, age=0):
        self.x = x
        self.y = y
        self.alive = True
        self.age = age  # in timesteps
        self.died = None

    def die(self, cause):
        self.alive = False
        self.age = DEATH_NEURON
        self.died = cause
        return 0

    def age_cell(self):
        self.age += 1

        AGE_HALF = MAX_AGE / 2
        P_apop = GAMMA / (1 + np.exp(-DELTA * (self.age - AGE_HALF)))

        if MIN_AGE < self.age < MAX_AGE and np.random.rand() < P_apop:
            self.die("apoptosis")
        elif self.age >= MAX_AGE:
            self.die("age")

    def prion_cell_death(self, prion_grid):
        neighborhood = prion_grid[self.x - 1 : self.x + 2, self.y - 1 : self.y + 2]
        neighbor_sum = np.sum(neighborhood) - prion_grid[self.x, self.y]
        if neighbor_sum > PRION_THRESHOLD:
            self.die("prion")

    def get_index(self, nx):
        i = int(self.x)
        j = int(self.y)
        return i + j * nx

    def get_coordinates(self):
        return self.x, self.y

    def get_age(self):
        return self.age


def neuron_secrete(neuron_grid, dt):
    mask = neuron_grid == HEALTH_NEURON

    full_structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)

    cardinal_structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    full_dilation = binary_dilation(mask, structure=full_structure)
    cardinal_dilation = binary_dilation(mask, structure=cardinal_structure)

    diagonal_mask = full_dilation & ~cardinal_dilation & ~mask
    cardinal_mask = cardinal_dilation & ~mask

    diagonal_mask &= neuron_grid == 0
    cardinal_mask &= neuron_grid == 0

    updated_grid = np.zeros_like(neuron_grid)
    updated_grid[cardinal_mask] = SECRETED_VALUE * dt
    updated_grid[diagonal_mask] = (SECRETED_VALUE / 2) * dt  # Half value on diagonals
    updated_grid = gaussian_filter(updated_grid, sigma=1)
    return updated_grid


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

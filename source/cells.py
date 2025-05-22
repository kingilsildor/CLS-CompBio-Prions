import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from config import *


class Neuron:
    """
    Represents a neuron in the simulation grid.

    Attributes
    ----------
    x (int): X-coordinate of the neuron.
    y (int): Y-coordinate of the neuron.
    alive (bool): Whether the neuron is alive.
    age (int): Age of the neuron (in timesteps).
    died (str or None): Cause of neuron death.
    """

    def __init__(self, x, y, age=0):
        """
        Initialize a neuron at position (x, y) with a given age.

        Params
        -------
        - x (int): X-coordinate.
        - y (int): Y-coordinate.
        - age (int, optional): Initial age. Defaults to 0.
        """
        self.x = x
        self.y = y
        self.alive = True
        self.age = age  # in timesteps
        self.died = None

    def die(self, cause) -> int:
        """
        Mark the neuron as dead and record the cause.

        Params
        -------
        - cause (str): Cause of death.

        Returns
        -------
        - int: 0 (for compatibility).
        """
        self.alive = False
        self.age = DEATH_NEURON
        self.died = cause
        return 0

    def age_cell(self) -> None:
        """
        Increment the neuron's age and check for apoptosis or age-related death.
        """
        self.age += 1

        AGE_HALF = MAX_AGE / 2
        P_apop = GAMMA / (1 + np.exp(-DELTA * (self.age - AGE_HALF)))

        if MIN_AGE < self.age and np.random.rand() < P_apop:
            self.die("apoptosis")
        # elif self.age >= MAX_AGE:
        #     self.die("age")

    def prion_cell_death(self, prion_grid) -> None:
        """
        Check for prion-induced neuron death based on prion concentration in the neighborhood.

        Params
        -------
        - prion_grid (np.ndarray): Grid of prion concentrations.
        """
        neighborhood = prion_grid[self.x - 1 : self.x + 2, self.y - 1 : self.y + 2]
        neighbor_sum = np.sum(neighborhood) - prion_grid[self.x, self.y]
        if neighbor_sum > PRION_THRESHOLD:
            self.die("prion")

    def get_index(self, nx) -> int:
        """
        Get the flattened index of the neuron in a grid.

        Params
        -------
        - nx (int): Number of grid points in x-direction.

        Returns
        -------
        - int: Flattened index.
        """
        i = int(self.x)
        j = int(self.y)
        return i + j * nx

    def get_coordinates(self) -> tuple:
        """
        Get the (x, y) coordinates of the neuron.

        Returns
        -------
        - tuple: (x, y)
        """
        return self.x, self.y

    def get_age(self) -> int:
        """
        Get the current age of the neuron.

        Returns
        -------
        - int: Age of the neuron.
        """
        return self.age


def neuron_secrete(neuron_grid, dt) -> np.ndarray:
    """
    Simulate protein secretion from healthy neurons to neighboring cells.

    Params
    -------
    - neuron_grid (np.ndarray): Grid representing neuron health states.
    - dt (float): Time step size.

    Returns
    -------
    - updated_grid (np.ndarray): Grid of secreted protein values.
    """
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


def neuron_age(N_neurons) -> list:
    """
    Generate random ages for a given number of neurons using an exponential distribution.

    Params
    -------
    - N_neurons (int): Number of neurons.

    Returns
    -------
    - ages (list): List of neuron ages.
    """
    ages = np.random.exponential(scale=EXP_SCALE, size=N_neurons)
    ages = [min(MAX_AGE, max(1, int(round(age)))) for age in ages]
    return ages


def create_neuron_dict(neuron_grid) -> dict:
    """
    Create a dictionary of Neuron objects from a neuron grid.

    Params
    -------
    - neuron_grid (np.ndarray): Grid representing neuron health states.

    Returns
    -------
    - neuron_dict (dict): Dictionary mapping coordinates to Neuron objects.
    """
    neuron_dict = {}
    neuron_coords = np.argwhere(neuron_grid == 1)
    ages = neuron_age(len(neuron_coords))

    for coords in neuron_coords:
        x, y = coords
        neuron = Neuron(x, y, age=ages.pop(0))
        neuron_grid[x, y] = neuron.get_age()
        neuron_dict[tuple(coords)] = neuron
    return neuron_dict

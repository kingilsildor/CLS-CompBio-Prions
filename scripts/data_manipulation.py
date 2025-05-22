import glob
import os

import numpy as np


def write_grid(
    grid: np.ndarray, grid_name: str, timestep: int, dir: str = "data"
) -> None:
    """
    Save a grid as a .npy file for a given timestep.

    Params
    -------
    - grid (np.ndarray): Grid to save.
    - grid_name (str): Name of the grid (e.g., "neuron", "prion", "protein").
    - timestep (int): Simulation timestep.
    - dir (str, optional): Directory to save the file. Defaults to "data".

    """
    os.makedirs(dir, exist_ok=True)
    filename = f"{grid_name}_t{timestep:03d}.npy"
    filepath = os.path.join(dir, filename)
    np.save(filepath, grid)


def read_all_grids(grid: str, dir: str = "data") -> list:
    """
    Read all .npy files for a specific grid type from a directory.

    Params
    -------
    - grid (str): Grid name to search for in filenames.
    - dir (str, optional): Directory to search. Defaults to "data".

    Returns
    -------
    - grids (list of np.ndarray): List of loaded grids.
    """
    grids = []
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".npy") and grid in file:
            path = os.path.join(dir, file)
            grids.append(np.load(path))
    return grids


def read_grids_at_timestep(timestep: int, dir: str = "data") -> list:
    """
    Read neuron, prion, and protein grids at a specific timestep.

    Params
    -------
    - timestep (int): Simulation timestep.
    - dir (str, optional): Directory to search. Defaults to "data".

    Returns
    -------
    - grids (list of np.ndarray): List of [neuron, prion, protein] grids at the timestep.
    """
    timestep_tag = f"_t{timestep:03d}"
    prefixes = ["neuron", "prion", "protein"]
    grids = []

    for prefix in prefixes:
        filename = f"{prefix}{timestep_tag}.npy"
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            grids.append(np.load(filepath))
        else:
            raise FileNotFoundError(f"Expected file not found: {filename}")

    return grids


def delete_npy(dir: str = "data") -> None:
    """
    Delete all .npy files in a directory.

    Params
    -------
    - dir (str, optional): Directory to clear. Defaults to "data".

    Returns
    -------
    - None
    """
    images = glob.glob(f"{dir}/*.npy")
    for image in images:
        os.remove(image)
    assert not glob.glob(f"{dir}/*.npy")

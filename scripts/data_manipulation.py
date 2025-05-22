import glob
import os

import numpy as np


def write_grid(grid, gird_name, timestep, dir="data"):
    os.makedirs(dir, exist_ok=True)

    filename = f"{gird_name}_t{timestep:03d}.npy"
    filepath = os.path.join(dir, filename)
    np.save(filepath, grid)


def read_all_grids(grid, dir="data"):
    grids = []
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".npy") and grid in file:
            path = os.path.join(dir, file)
            grids.append(np.load(path))
    return grids


def read_grids_at_timestep(timestep, dir="data"):
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


def delete_npy(dir="data"):
    images = glob.glob(f"{dir}/*.npy")
    for image in images:
        os.remove(image)
    assert not glob.glob(f"{dir}/*.npy")

import glob
import os

import imageio


def create_gif(file_path, timepoints, duration=10, delete_img=True):
    images = [imageio.imread(f"{file_path}/step_{i:03d}.png") for i in timepoints]
    imageio.mimsave(f"{file_path}/{file_path}_sim.gif", images, duration=duration)

    if delete_img:
        images = glob.glob(f"{file_path}/step_*.png")
        for image in images:
            os.remove(image)
        assert not glob.glob(f"{file_path}/step_*.png")

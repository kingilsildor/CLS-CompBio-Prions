import glob
import os

import imageio


def create_gif(file_path, timepoints, file_name="step", duration=10, delete_img=True):
    images = [
        imageio.imread(f"{file_path}/{file_name}_{i:03d}.png") for i in timepoints
    ]
    gif_path = f"{file_path}/{file_name}_sim.gif"
    imageio.mimsave(gif_path, images, duration=duration, loop=0)

    if delete_img:
        images = glob.glob(f"{file_path}/{file_name}_*.png")
        for image in images:
            os.remove(image)
        assert not glob.glob(f"{file_path}/{file_name}_*.png")

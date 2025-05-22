import glob
import os

import imageio


def create_gif(
    file_path: str,
    timepoints: list,
    file_name: str = "step",
    duration: int = 10,
    delete_img: bool = True,
):
    """
    Create a GIF animation from a sequence of PNG images and optionally delete the images.

    Params
    -------
    - file_path (str): Directory containing the images.
    - timepoints (list): List of timepoint indices to include in the GIF.
    - file_name (str, optional): Base name of the image files. Defaults to "step".
    - duration (int, optional): Duration of the GIF animation in seconds. Defaults to 10.
    - delete_img (bool, optional): Whether to delete the PNG images after creating the GIF. Defaults to True.

    Returns
    -------
    - A GIF file saved in the specified directory.
    """
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

import numpy as np

from config import *


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True
        self.age = 0  # in timesteps

    def secrete(self): ...

    def die(self):
        self.alive = False

    def age_cell(self) -> int:
        self.age += 1
        P_death = 1 - np.exp(-((BASE_DEATH_RATE * self.age) ** SHAPE_PARAM))

        if np.random.random() >= P_death:
            self.die()
            return 0

        return 1

    def get_coordinates(self):
        return self.x, self.y

    def get_age(self):
        return self.age

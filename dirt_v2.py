import pygame
import numpy as np

class Dirt:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.radius = 6
        self.color = (139, 69, 19)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), self.radius)



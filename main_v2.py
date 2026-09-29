import pygame
import random
import numpy as np

from dirt_v2 import Dirt
from cleaner_v2 import Cleaner

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Porszívó Szimuláció - Támadó mechanika")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

WHITE = (255, 255, 255)

# Porszívók és koszok inicializálása
cleaner_list = [
    Cleaner(
        random.randint(50, WIDTH - 50),
        random.randint(50, HEIGHT - 50),
        speed=random.uniform(1.5, 3.5) # A 30.5 sebesség túl nagy volt, visszavettem, hogy ne ugráljanak ki a pályáról
    ) for _ in range(5)
]

dirt_list = [
    Dirt(random.randint(30, WIDTH - 30), random.randint(30, HEIGHT - 30))
    for _ in range(20)
]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Logika frissítése (átadjuk a cleaner_list-et is a támadáshoz!)
    for cleaner in cleaner_list:
        cleaner.update(dirt_list, cleaner_list, (WIDTH, HEIGHT))

    # Véletlenszerű új kosz generálása
    if random.random() < 0.003 and len(dirt_list) < 25:
        dirt_list.append(Dirt(random.randint(30, WIDTH - 30), random.randint(30, HEIGHT - 30)))

    # Kirajzolás
    screen.fill(WHITE)
    
    for dirt in dirt_list:
        dirt.draw(screen)
        
    for cleaner in cleaner_list:
        cleaner.draw(screen, font)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
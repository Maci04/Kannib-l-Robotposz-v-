import pygame
import numpy as np

class Cleaner:
    def __init__(self, x, y, speed=2.5):
        self.pos = np.array([x, y], dtype=float)
        
        # Véletlenszerű kezdő irányvektor (egységvektor)
        random_angle = np.random.uniform(0, 2 * np.pi)
        self.dir = np.array([np.cos(random_angle), np.sin(random_angle)])
        
        self.speed = speed
        self.energy = 100.0
        self.radius = 15
        self.sensor_range = 100.0  # Látótávolság (pixel)
        
        self.RED = (200, 0, 0)
        self.BLUE = (0, 102, 204)

    def update(self, dirt_list, bounds):
        if self.energy <= 0:
            self.energy = 0
            return

        self.energy -= self.speed * 0.05

        closest_dirt = None
        min_dist = float('inf')
        
        # Beállítjuk a látószöget (90 fok = +-45 fok a haladási iránytól)
        fov_angle_deg = 90  

        # 1. KOSZ KERESÉSE A LÁTÓTÁVOLSÁGON ÉS A LÁTÓSZÖGÖN BELÜL
        for dirt in dirt_list:
            to_dirt = dirt.pos - self.pos
            dist = np.linalg.norm(to_dirt)
            
            if dist < self.sensor_range and dist > 0:
                to_dirt_dir = to_dirt / dist
                
                # Skaláris szorzat a robot mozgási iránya és a kosz iránya között
                dot_product = np.dot(self.dir, to_dirt_dir)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                
                # Szög kiszámítása fokban
                angle_deg = np.degrees(np.arccos(dot_product))
                
                # Csak akkor veszi észre, ha a látószögén és a látótávolságán belül van
                if angle_deg <= (fov_angle_deg / 2) and dist < min_dist:
                    min_dist = dist
                    closest_dirt = dirt

        # 2. IRÁNY MEGHATÁROZÁSA
        if closest_dirt is not None:
            # Ha lát koszt a látószögében, felé fordul
            target_dir = closest_dirt.pos - self.pos
            self.dir = target_dir / np.linalg.norm(target_dir)

        
        else:
            # Ha nem lát koszt, kis valószínűséggel korrigálja az irányát (bolyongás)
            if np.random.rand() < 0.05:
                angle_change = np.random.uniform(-0.3, 0.3)
                cos_a, sin_a = np.cos(angle_change), np.sin(angle_change)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                self.dir = np.dot(rot_matrix, self.dir)
        
    
        # 3. MOZGÁS ÉS FALAKRÓL VALÓ VISSZAPATTANÁS
        self.pos += self.dir * self.speed

        width, height = bounds
        if self.pos[0] <= self.radius or self.pos[0] >= width - self.radius:
            self.dir[0] *= -1  # Visszapattan a függőleges falról
            self.pos[0] = np.clip(self.pos[0], self.radius, width - self.radius)

        if self.pos[1] <= self.radius or self.pos[1] >= height - self.radius:
            self.dir[1] *= -1  # Visszapattan a vízszintes falról
            self.pos[1] = np.clip(self.pos[1], self.radius, height - self.radius)

        # 4. TAKARÍTÁS
        if closest_dirt is not None and min_dist < self.radius:
            dirt_list.remove(closest_dirt)
            self.energy = min(100.0, self.energy + 20.0)

    def draw(self, surface, font):
        color = self.RED if self.energy <= 0 else self.BLUE
        
        # Robot kirajzolása
        pygame.draw.circle(surface, color, self.pos.astype(int), self.radius)
        
        # --- LÁTÓSZÖG (ZSEBLÁMPA FÉNY) KIRAJZOLÁSA ---
        current_angle = np.arctan2(self.dir[1], self.dir[0])
        fov_rad = np.radians(90)  # 90 fokos látószög
        
        left_angle = current_angle - fov_rad / 2
        right_angle = current_angle + fov_rad / 2
        
        p1 = self.pos
        p2 = self.pos + np.array([np.cos(left_angle), np.sin(left_angle)]) * self.sensor_range
        p3 = self.pos + np.array([np.cos(right_angle), np.sin(right_angle)]) * self.sensor_range
        
        # Látószög vonalainak kirajzolása halványszürkével
        pygame.draw.line(surface, (200, 200, 200), p1.astype(int), p2.astype(int), 1)
        pygame.draw.line(surface, (200, 200, 200), p1.astype(int), p3.astype(int), 1)
        # ---------------------------------------------

        # Energia kiírása
        energy_text = font.render(f"E: {int(self.energy)}", True, (0, 0, 0))
        surface.blit(energy_text, (self.pos[0] - 15, self.pos[1] - 30))
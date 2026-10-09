import pygame
import numpy as np
import random


class Cleaner:
    def __init__(self, x, y, speed=2.5):
        self.pos = np.array([x, y], dtype=float)

        random_angle = np.random.uniform(0, 2 * np.pi)
        self.dir = np.array([np.cos(random_angle), np.sin(random_angle)])

        self.speed = speed
        self.energy = 100.0
        self.radius = 15
        self.sensor_range = 100.0

        self.attack_threshold = 75.0
        self.is_attacking = False

        # Effektek változói
        self.leech_timer = 0
        self.leech_target = None
        self.attacked_timer = 0
        self.floating_texts = []

        # Védekező mechanizmus
        self.danger_range = 70.0         # 360°-os veszélyérzékelő hatótáv
        self.sprint_multiplier = 1.6     # menekülés közbeni sebességszorzó
        self.is_fleeing = False
        self.attacker = None             # aki éppen rátapadt
        self.shock_cooldown = 0
        self.shock_flash = 0             # sokk effekt időzítője
        self.shock_cost = 5.0            # a védő energiába kerül
        self.shock_damage = 10.0         # a támadó ennyit veszít
        self.shock_cooldown_max = 120

        self.RED = (200, 0, 0)
        self.BLUE = (0, 102, 204)
        self.GREY = (100, 100, 100)

    def update(self, dirt_list, cleaner_list, bounds):
        if self.energy <= 0:
            self.energy = 0
            self.leech_timer = 0
            self.attacked_timer = 0
            self.is_fleeing = False
            self.shock_flash = 0
            return

        self.is_attacking = self.energy < self.attack_threshold

        if self.leech_timer > 0:
            self.leech_timer -= 1
        else:
            self.leech_target = None

        if self.attacked_timer > 0:
            self.attacked_timer -= 1
        else:
            self.attacker = None

        if self.shock_cooldown > 0:
            self.shock_cooldown -= 1
        if self.shock_flash > 0:
            self.shock_flash -= 1

        # 0. ELLENTÁMADÁS: ha rátapadtak, elektrosokkal visszaütünk
        if (self.attacked_timer > 0 and self.attacker is not None
                and self.attacker.energy > 0
                and self.shock_cooldown == 0 and self.energy > 10):
            self.energy -= self.shock_cost
            self.attacker.energy -= self.shock_damage
            self.attacker.leech_timer = 0
            self.attacker.leech_target = None

            # Visszarúgás: a támadót eltoljuk és elfordítjuk
            push = self.attacker.pos - self.pos
            push_len = np.linalg.norm(push)
            if push_len > 0:
                push_dir = push / push_len
                self.attacker.pos += push_dir * 25
                self.attacker.dir = push_dir.copy()
                width, height = bounds
                r = self.attacker.radius
                self.attacker.pos[0] = np.clip(self.attacker.pos[0], r, width - r)
                self.attacker.pos[1] = np.clip(self.attacker.pos[1], r, height - r)

            self.shock_cooldown = self.shock_cooldown_max
            self.shock_flash = 12
            self.attacker.floating_texts.append({
                'pos': self.attacker.pos.copy() + np.array([-10, -20]),
                'text': f"-{int(self.shock_damage)}",
                'color': (255, 215, 0),
                'life': 45
            })
            self.floating_texts.append({
                'pos': self.pos.copy() + np.array([10, -20]),
                'text': f"-{int(self.shock_cost)}",
                'color': (255, 215, 0),
                'life': 45
            })
            self.attacker = None

        closest_dirt = None
        min_dirt_dist = float('inf')

        closest_victim = None
        min_victim_dist = float('inf')

        fov_angle_deg = 90

        # 1. KOSZ KERESÉSE
        for dirt in dirt_list:
            to_dirt = dirt.pos - self.pos
            dist = np.linalg.norm(to_dirt)

            if 0 < dist < self.sensor_range:
                to_dirt_dir = to_dirt / dist
                dot_product = np.clip(np.dot(self.dir, to_dirt_dir), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_product))

                if angle_deg <= (fov_angle_deg / 2) and dist < min_dirt_dist:
                    min_dirt_dist = dist
                    closest_dirt = dirt

        # 2. ÁLDOZAT KERESÉSE (Csak élőket keres!)
        if self.is_attacking:
            for other in cleaner_list:
                if other is not self and other.energy > 0:
                    to_other = other.pos - self.pos
                    dist = np.linalg.norm(to_other)

                    if 0 < dist < self.sensor_range:
                        to_other_dir = to_other / dist
                        dot_product = np.clip(np.dot(self.dir, to_other_dir), -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(dot_product))

                        if angle_deg <= (fov_angle_deg / 2) and dist < min_victim_dist:
                            min_victim_dist = dist
                            closest_victim = other

        # 2/b. VESZÉLY ÉSZLELÉSE (360°, a hátunk mögé is lát)
        closest_threat = None
        min_threat_dist = float('inf')
        for other in cleaner_list:
            if other is not self and other.energy > 0 and other.is_attacking:
                to_me = self.pos - other.pos
                dist = np.linalg.norm(to_me)
                if 0 < dist < self.danger_range and dist < min_threat_dist:
                    # Csak akkor veszély, ha felénk néz
                    if np.dot(other.dir, to_me / dist) > 0.5:
                        min_threat_dist = dist
                        closest_threat = other

        # A vadász (támadó üzemmód) nem menekül, csak a nem támadók
        self.is_fleeing = closest_threat is not None and not self.is_attacking

        # 3. IRÁNY MEGHATÁROZÁSA
        if self.is_fleeing:
            away = self.pos - closest_threat.pos
            away_len = np.linalg.norm(away)
            flee_dir = away / away_len if away_len > 0 else -self.dir

            # Ne szoruljon sarokba: faltól közel a pálya közepe felé is terelünk
            width, height = bounds
            margin = 80
            if (self.pos[0] < margin or self.pos[0] > width - margin or
                    self.pos[1] < margin or self.pos[1] > height - margin):
                to_center = np.array([width / 2, height / 2]) - self.pos
                center_len = np.linalg.norm(to_center)
                if center_len > 0:
                    flee_dir = flee_dir + 0.8 * to_center / center_len

            flee_len = np.linalg.norm(flee_dir)
            if flee_len > 0:
                self.dir = flee_dir / flee_len
        else:
            target_pos = None
            if self.is_attacking and closest_victim is not None:
                target_pos = closest_victim.pos
            elif closest_dirt is not None:
                target_pos = closest_dirt.pos

            if target_pos is not None:
                target_dir = target_pos - self.pos
                if np.linalg.norm(target_dir) > 0:
                    self.dir = target_dir / np.linalg.norm(target_dir)
            else:
                if np.random.rand() < 0.05:
                    angle_change = np.random.uniform(-0.3, 0.3)
                    cos_a, sin_a = np.cos(angle_change), np.sin(angle_change)
                    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    self.dir = np.dot(rot_matrix, self.dir)

        # 4. MOZGÁS (sprint közben gyorsabban fogy az energia)
        current_speed = self.speed * (self.sprint_multiplier if self.is_fleeing else 1.0)
        self.energy -= current_speed * 0.05
        self.pos += self.dir * current_speed

        # 5. ROBOT-ROBOT ÜTKÖZÉS ÉS TÁMADÁS (Halott porszívókat ignoráljuk!)
        for other in cleaner_list:
            if other is not self and other.energy > 0:
                dist_vec = self.pos - other.pos
                dist = np.linalg.norm(dist_vec)

                if 0 < dist < (self.radius + other.radius):
                    # 5/a. Fizikai eltolás
                    overlap = (self.radius + other.radius) - dist
                    normal = dist_vec / dist

                    self.pos += normal * (overlap / 2)
                    other.pos -= normal * (overlap / 2)

                    self_to_other = -normal
                    other_to_self = normal

                    self_front = np.dot(self.dir, self_to_other) > 0.5
                    other_front = np.dot(other.dir, other_to_self) > 0.5

                    # 5/b. Két támadó szemből összecsap (Visszapattanás)
                    # Az id() ellenőrzés azért kell, hogy a levonás egy frame-ben csak egyszer fusson le a párosra
                    if self.is_attacking and other.is_attacking and self_front and other_front and id(self) < id(other):
                        collision_dmg = 2.0
                        self.energy -= collision_dmg
                        other.energy -= collision_dmg

                        angle_change = np.random.uniform(-0.5, 0.5)
                        cos_a = np.cos(np.pi + angle_change)
                        sin_a = np.sin(np.pi + angle_change)
                        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                        self.dir = np.dot(rot_matrix, self.dir)
                        other.dir = np.dot(rot_matrix, other.dir)

                        self.floating_texts.append({'pos': self.pos.copy() + np.array([-10, -20]), 'text': f"-{int(collision_dmg)}", 'color': (255, 140, 0), 'life': 45})
                        other.floating_texts.append({'pos': other.pos.copy() + np.array([10, -20]), 'text': f"-{int(collision_dmg)}", 'color': (255, 140, 0), 'life': 45})

                    # 5/c. Rácuppanás (Energiaszívás)
                    elif self.is_attacking:
                        back_of_other = -other.dir
                        back_hit = np.dot(other_to_self, back_of_other) > 0.5

                        if self_front and back_hit:
                            steal_rate = 2.0
                            stolen = min(steal_rate, other.energy)
                            other.energy -= stolen
                            self.energy = min(100.0, self.energy + stolen)

                            self.leech_timer = 3
                            self.leech_target = other
                            other.attacked_timer = 3
                            other.attacker = self  # a védekezéshez kell

                            if random.random() < 0.2:
                                self.floating_texts.append({'pos': self.pos.copy() + np.array([-10, -20]), 'text': f"+{int(stolen)}", 'color': (0, 200, 0), 'life': 45})
                                other.floating_texts.append({'pos': other.pos.copy() + np.array([10, -20]), 'text': f"-{int(stolen)}", 'color': (255, 0, 0), 'life': 45})

        # 6. FALAKRÓL VALÓ VISSZAPATTANÁS
        width, height = bounds
        if self.pos[0] <= self.radius or self.pos[0] >= width - self.radius:
            self.dir[0] *= -1
            self.pos[0] = np.clip(self.pos[0], self.radius, width - self.radius)

        if self.pos[1] <= self.radius or self.pos[1] >= height - self.radius:
            self.dir[1] *= -1
            self.pos[1] = np.clip(self.pos[1], self.radius, height - self.radius)

        # 7. TAKARÍTÁS
        if closest_dirt is not None and min_dirt_dist < self.radius:
            if closest_dirt in dirt_list:
                dirt_list.remove(closest_dirt)
                self.energy = min(100.0, self.energy + 20.0)

    def draw(self, surface, font):
        draw_pos = self.pos.copy()
        if self.attacked_timer > 0 and self.energy > 0:
            jitter = np.array([random.randint(-3, 3), random.randint(-3, 3)])
            draw_pos += jitter

        if self.energy <= 0:
            color = self.GREY
        elif self.leech_timer > 0:
            color = (255, 140, 0) if pygame.time.get_ticks() % 150 < 75 else self.RED
        elif self.is_attacking:
            color = self.RED
        else:
            color = self.BLUE

        pygame.draw.circle(surface, color, draw_pos.astype(int), self.radius)

        # Menekülő porszívó: fehér gyűrű a testen
        if self.is_fleeing and self.energy > 0:
            pygame.draw.circle(surface, (255, 255, 255), draw_pos.astype(int), self.radius - 4, 2)

        # Sokk effekt: táguló sárga gyűrű
        if self.shock_flash > 0:
            pygame.draw.circle(surface, (255, 215, 0), draw_pos.astype(int),
                               self.radius + 20 - self.shock_flash, 3)

        mouth_pos = draw_pos + self.dir * (self.radius - 3)
        pygame.draw.circle(surface, (0, 0, 0), mouth_pos.astype(int), 4)

        # Villám sugár kirajzolása (csak élő célpontra!)
        if self.leech_timer > 0 and self.leech_target is not None and self.leech_target.energy > 0:
            start_pos = mouth_pos
            target_draw_pos = self.leech_target.pos.copy()
            if self.leech_target.attacked_timer > 0:
                target_draw_pos += np.array([random.randint(-3, 3), random.randint(-3, 3)])
            end_pos = target_draw_pos

            steps = 4
            lightning_points = [start_pos]
            for i in range(1, steps):
                fraction = i / steps
                base_point = start_pos + (end_pos - start_pos) * fraction

                line_dir = end_pos - start_pos
                line_norm = np.linalg.norm(line_dir)
                if line_norm > 0:
                    line_dir = line_dir / line_norm
                    perp = np.array([-line_dir[1], line_dir[0]])
                    noise = perp * random.uniform(-6, 6)
                    lightning_points.append(base_point + noise)

            lightning_points.append(end_pos)
            pygame.draw.lines(surface, (0, 255, 255), False, [p.astype(int) for p in lightning_points], 3)

        if self.energy > 0:
            current_angle = np.arctan2(self.dir[1], self.dir[0])
            fov_rad = np.radians(90)

            left_angle = current_angle - fov_rad / 2
            right_angle = current_angle + fov_rad / 2

            p1 = draw_pos
            p2 = draw_pos + np.array([np.cos(left_angle), np.sin(left_angle)]) * self.sensor_range
            p3 = draw_pos + np.array([np.cos(right_angle), np.sin(right_angle)]) * self.sensor_range

            fov_color = (255, 150, 150) if self.is_attacking else (200, 200, 200)
            pygame.draw.line(surface, fov_color, p1.astype(int), p2.astype(int), 1)
            pygame.draw.line(surface, fov_color, p1.astype(int), p3.astype(int), 1)

        energy_text = font.render(f"E: {int(self.energy)}", True, (0, 0, 0))
        surface.blit(energy_text, (draw_pos[0] - 15, draw_pos[1] - 30))

        for ft in self.floating_texts[:]:
            ft['pos'][1] -= 1.0
            ft['life'] -= 1

            if ft['life'] <= 0:
                self.floating_texts.remove(ft)
            else:
                text_surf = font.render(ft['text'], True, ft['color'])
                surface.blit(text_surf, ft['pos'].astype(int))

import os
import sys
import math
import time
from collections import deque
from typing import Tuple, List, Optional, Dict, Any

# Munkakönyvtár rögzítése a main.py mappájára
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from ursina import (
    Ursina,
    Entity,
    Text,
    Button as UButton,
    Texture,
    camera,
    color,
    mouse,
    held_keys,
    time as utime,
    window,
    DirectionalLight,
    AmbientLight,
    Vec2,
    Vec3,
    destroy,
)

try:
    from ursina.shaders import lit_with_shadows_shader
except ImportError:
    lit_with_shadows_shader = None

from agent import RobotAgent, AgentMessage, lerp_angle
from environment import SimulationEnvironment, MAP_CATALOG, MAX_WORLD_SCALE, extract_merged_rectangles
from gui_manager import GuiManager
import camera_rig
from sound_manager import SoundManager
from coordination import ROLE_SHORT, ROLE_LABEL
from camera_rig import OuterCameraState, fpv_pose, rate_limited_angle, follow_profile
from evolution import EvolutionaryAlgorithm


# ==============================================================================
# BIZTONSÁGOS URSINA SZÍNSEGÉDFÜGGVÉNYEK
# ==============================================================================
def rgb(r: int, g: int, b: int):
    """Ursina-szín 0..255 RGB értékekből (0..1 tartományra normalizálva)."""
    return color.Color(r / 255.0, g / 255.0, b / 255.0, 1.0)


def rgba(r: int, g: int, b: int, a: int = 255):
    """Ursina-szín 0..255 RGBA értékekből (0..1 tartományra normalizálva)."""
    return color.Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)


# Központilag módosítható billentyűkiosztás
KEYS: Dict[str, str] = {
    "gui_toggle": "tab",        # a teljes szimulációs GUI elrejtése / visszaállítása
    "fullscreen": "f11",        # teljes képernyő ki/be
    "step": ".",                # 1 tick léptetés
    "settings": "o",            # GUI- és kamera-beállítások panel
    "fit_map": "end",           # teljes pálya belátható nézete
    "follow": "z",              # kijelölt robot követése ki/be
    "camera_home": "home",      # alapértelmezett kameranézet
    "comm": "k",                # kommunikációs elemek (chat log + szerepcímkék) ki/be
    "sound": "b",               # hangeffektek BE/KI
}


CHAR_W_UI = 0.0171


def fit_text_scale(text: str, base_scale: float, width_ui: float, min_factor: float = 0.55) -> float:
    """A gomb feliratának skálája úgy, hogy beleférjen a gomb szélességébe."""
    clean = text.replace("\n", " ")
    need = max(1, len(clean)) * CHAR_W_UI * base_scale
    avail = width_ui * 0.90
    if need <= avail:
        return base_scale
    return max(base_scale * min_factor, base_scale * avail / need)


class UI:
    """Referencia kép szerinti modern, sötétkék-smaragd dashboard színpaletta."""
    BG = rgb(10, 14, 20)
    PANEL = rgba(16, 22, 32, 242)
    PANEL_INNER = rgba(21, 29, 42, 248)
    PANEL_EDGE = rgba(36, 50, 70, 255)
    DIVIDER = rgba(32, 44, 62, 210)

    TEXT = rgb(238, 244, 252)
    TEXT_DIM = rgb(142, 156, 176)
    TEXT_MUTED = rgb(98, 112, 132)

    TITLE = rgb(52, 211, 153)
    PRIMARY = rgb(16, 185, 129)
    PRIMARY_HOVER = rgb(52, 211, 153)

    ACTIVE_BG = rgb(18, 53, 46)
    ACTIVE_EDGE = rgb(34, 118, 96)
    ACTIVE_HOVER = rgb(24, 72, 62)

    BTN = rgb(22, 30, 44)
    BTN_HOVER = rgb(34, 46, 66)
    BTN_PRESSED = rgb(14, 20, 30)

    SELECTED = rgb(18, 53, 46)
    SELECTED_HOVER = rgb(28, 78, 66)
    ACCENT = rgb(18, 53, 46)
    ACCENT_HOVER = rgb(28, 78, 66)

    WARN = rgb(245, 166, 35)
    DANGER = rgb(239, 68, 68)
    DANGER_BG = rgb(40, 20, 30)
    DANGER_EDGE = rgb(120, 38, 58)
    DANGER_HOVER = rgb(165, 48, 72)


class ProgressBarUI:
    """Vízszintes folyamatsáv (progress bar) kártyákhoz, szektorokhoz és utility értékekhez."""

    def __init__(
        self,
        parent,
        left_x: float,
        cy: float,
        width: float,
        height: float = 0.007,
        fg_color=None,
        bg_color=None,
        z: float = -0.012,
    ):
        self.left_x = left_x
        self.cy = cy
        self.width = width
        self.height = height
        self.z = z
        self.track = Entity(
            parent=parent,
            model="quad",
            color=bg_color or rgb(32, 43, 60),
            scale=(width, height),
            position=(left_x + width / 2.0, cy, z),
        )
        self.fill = Entity(
            parent=parent,
            model="quad",
            color=fg_color or UI.PRIMARY,
            scale=(0.001, height),
            position=(left_x, cy, z - 0.001),
        )

    def set_progress(self, ratio: float, col=None) -> None:
        r = max(0.0, min(1.0, float(ratio)))
        if r <= 0.005:
            self.fill.enabled = False
        else:
            self.fill.enabled = True
            fw = max(0.002, self.width * r)
            self.fill.scale_x = fw
            self.fill.x = self.left_x + fw / 2.0
        if col is not None:
            self.fill.color = col

    def set_enabled(self, val: bool) -> None:
        self.track.enabled = val
        self.fill.enabled = val


for _m_data in MAP_CATALOG.values():
    if "difficulty" in _m_data:
        _m_data["difficulty"] = _m_data["difficulty"].replace("★", "*").replace("☆", "-")


# ==============================================================================
# 0. ALKALMAZÁS-IKON
# ==============================================================================
def ensure_app_icon() -> str:
    icon_dir = os.path.join(SCRIPT_DIR, "textures")
    icon_path = os.path.join(icon_dir, "ursina.ico")
    if os.path.exists(icon_path):
        return icon_path
    try:
        os.makedirs(icon_dir, exist_ok=True)
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((8, 8, 248, 248), radius=52, fill=(10, 14, 20, 255), outline=(16, 185, 129, 255), width=10)
        d.ellipse((48, 48, 208, 208), fill=(22, 30, 44, 255), outline=(52, 211, 153, 255), width=8)
        d.ellipse((98, 98, 158, 158), fill=(16, 185, 129, 255))
        d.rectangle((120, 52, 136, 84), fill=(245, 166, 35, 255))
        img.save(icon_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except Exception as exc:
        print(f"[ikon] Nem sikerült létrehozni az alkalmazásikont: {exc}")
    return icon_path


# ==============================================================================
# 1. AUTOMATIKUS NAGY FELBONTÁSÚ TEXTÚRA-GENERÁTOR
# ==============================================================================
def ensure_3d_textures() -> str:
    tex_dir = os.path.join(SCRIPT_DIR, "assets", "textures")
    os.makedirs(tex_dir, exist_ok=True)
    rng = np.random.default_rng(101)

    def save_if_missing(filename: str, builder_fn) -> None:
        path = os.path.join(tex_dir, filename)
        if not os.path.exists(path):
            img = builder_fn()
            img.save(path, "PNG")

    def make_concrete() -> Image.Image:
        arr = rng.integers(52, 66, size=(512, 512, 3), dtype=np.uint8)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 8, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 128):
            draw.line([(i, 0), (i, 512)], fill=(38, 44, 54), width=3)
            draw.line([(0, i), (512, i)], fill=(38, 44, 54), width=3)
            draw.line([(i + 3, 0), (i + 3, 512)], fill=(74, 82, 96), width=1)
        return img

    def make_parquet() -> Image.Image:
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :] = [115, 82, 54]
        noise = rng.integers(-10, 10, size=(512, 512, 1), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for y in range(0, 512, 64):
            draw.line([(0, y), (512, y)], fill=(70, 48, 30), width=3)
            offset = (y // 64) * 128
            for x in range(offset % 256, 512, 256):
                draw.line([(x, y), (x, y + 64)], fill=(70, 48, 30), width=2)
        return img

    def make_ice_floor() -> Image.Image:
        arr = rng.integers(145, 175, size=(512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) - 35, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) - 10, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 128):
            draw.line([(i, 0), (i, 512)], fill=(95, 135, 175), width=2)
            draw.line([(0, i), (512, i)], fill=(95, 135, 175), width=2)
        return img

    def make_industrial_floor() -> Image.Image:
        arr = rng.integers(40, 52, size=(512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 64):
            draw.line([(i, 0), (i, 512)], fill=(28, 32, 38), width=4)
            draw.line([(0, i), (512, i)], fill=(28, 32, 38), width=4)
            draw.rectangle([i + 8, i + 8, i + 56, i + 56], outline=(68, 76, 88), width=2)
        return img

    def make_arena_floor() -> Image.Image:
        arr = rng.integers(32, 42, size=(512, 512, 3), dtype=np.uint8)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 12, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 128):
            draw.line([(i, 0), (i, 512)], fill=(88, 58, 135), width=3)
            draw.line([(0, i), (512, i)], fill=(88, 58, 135), width=3)
        return img

    def make_wall_industrial() -> Image.Image:
        arr = rng.integers(105, 122, size=(512, 512, 3), dtype=np.uint8)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 15, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 511, 511], outline=(60, 72, 88), width=8)
        draw.line([(0, 256), (512, 256)], fill=(70, 82, 100), width=6)
        draw.line([(256, 0), (256, 512)], fill=(70, 82, 100), width=6)
        draw.rectangle([0, 440, 512, 512], fill=(215, 165, 20))
        for x in range(-512, 512, 48):
            draw.polygon([(x, 512), (x + 24, 512), (x + 96, 440), (x + 72, 440)], fill=(30, 34, 40))
        return img

    def make_wall_hazard() -> Image.Image:
        img = make_wall_industrial()
        draw = ImageDraw.Draw(img)
        draw.rectangle([16, 20, 496, 120], fill=(75, 85, 102), outline=(45, 52, 65), width=4)
        return img

    def make_wall_brick() -> Image.Image:
        arr = rng.integers(195, 212, size=(512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 450, 512, 512], fill=(95, 65, 42))
        draw.line([(0, 450), (512, 450)], fill=(65, 42, 24), width=4)
        return img

    def make_rack_metal() -> Image.Image:
        arr = rng.integers(135, 155, size=(512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + 25, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) - 25, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for y in range(0, 512, 128):
            draw.rectangle([0, y, 512, y + 22], fill=(55, 65, 82))
        for x in range(0, 512, 128):
            draw.rectangle([x, 0, x + 18, 512], fill=(45, 55, 70))
        return img

    def make_robot_chassis() -> Image.Image:
        img = Image.new("RGB", (512, 512), (210, 218, 228))
        draw = ImageDraw.Draw(img)
        draw.ellipse([16, 16, 496, 496], fill=(185, 195, 210), outline=(90, 102, 120), width=12)
        draw.ellipse([72, 72, 440, 440], fill=(220, 228, 238), outline=(120, 135, 155), width=6)
        draw.ellipse([180, 180, 332, 332], fill=(140, 152, 170), outline=(80, 90, 105), width=4)
        return img

    save_if_missing("floor_concrete.png", make_concrete)
    save_if_missing("floor_parquet.png", make_parquet)
    save_if_missing("floor_ice.png", make_ice_floor)
    save_if_missing("floor_industrial.png", make_industrial_floor)
    save_if_missing("floor_arena.png", make_arena_floor)
    save_if_missing("wall_industrial.png", make_wall_industrial)
    save_if_missing("wall_hazard.png", make_wall_hazard)
    save_if_missing("wall_brick.png", make_wall_brick)
    save_if_missing("rack_metal.png", make_rack_metal)
    save_if_missing("robot_chassis.png", make_robot_chassis)

    return tex_dir


# ==============================================================================
# 2. MATPLOTLIB EVOLÚCIÓS ELEMZŐ ABLAK
# ==============================================================================
def show_matplotlib_stats(history: dict, mode_name: str = "CLASSIC") -> None:
    if len(history["generation"]) == 0:
        print("Még nincs lezárt generáció a grafikonok kirajzolásához!")
        return

    gens = history["generation"]
    plt.style.use(
        "seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"
    )
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"3D Evolúciós Labor és Statisztikák - Mód: {mode_name}")

    axs[0, 0].plot(gens, history["max_fitness"], label="Max Fitnesz", color="#f43f5e", linewidth=2)
    axs[0, 0].plot(gens, history["avg_fitness"], label="Átlag Fitnesz", color="#38bdf8", linewidth=2)
    axs[0, 0].set_title(f"Fitnesz alakulása ({mode_name})")
    axs[0, 0].set_xlabel("Generáció")
    axs[0, 0].legend()

    axs[0, 1].plot(gens, history["avg_aggressiveness"], label="Agresszivitás", color="#ef4444", linewidth=2)
    axs[0, 1].plot(gens, history["avg_cooperation"], label="Kooperáció", color="#10b981", linewidth=2)
    axs[0, 1].plot(gens, history["avg_risk_tolerance"], label="Kockázatvállalás", color="#eab308", linewidth=2)
    axs[0, 1].set_title("Átlagos Genom Súlyok (Agresszió vs. Kooperáció)")
    axs[0, 1].set_xlabel("Generáció")
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].legend()

    axs[1, 0].plot(gens, history["survival_rate"], label="Túlélési arány (%)", color="#a855f7", linewidth=2)
    axs[1, 0].plot(
        gens,
        history["return_or_goal_rate"],
        label="Célba érés / Bázisra térés (%)",
        color="#3b82f6",
        linestyle="--",
        linewidth=2,
    )
    axs[1, 0].set_title("Túlélési és Küldetés-teljesítési Arány (%)")
    axs[1, 0].set_xlabel("Generáció")
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].legend()

    ax4_twin = axs[1, 1].twinx()
    l1 = axs[1, 1].plot(gens, history["total_dirt"], label="Feltakarított kosz", color="#10b981", linewidth=2)
    l2 = ax4_twin.plot(
        gens,
        history["total_kills"],
        label="Harci kiütések (Kills)",
        color="#f97316",
        linestyle="--",
        linewidth=2,
    )
    l3 = ax4_twin.plot(
        gens,
        history["total_messages"],
        label="Rádióüzenetek (db)",
        color="#06b6d4",
        linestyle=":",
        linewidth=2,
    )
    axs[1, 1].set_title("Takarítás, Kommunikáció és Harc Dinamikája")
    axs[1, 1].set_xlabel("Generáció")
    lines = l1 + l2 + l3
    axs[1, 1].legend(lines, [l.get_label() for l in lines], loc="upper left")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. 3D RÉSZECSKEEFFEKT-MOTOR
# ==============================================================================
class ParticleSystem3D:
    MAX_PARTICLES = 110

    def __init__(self):
        self.particles: List[List[Any]] = []
        self.rng = np.random.default_rng(77)

    def clear_all(self) -> None:
        for p in self.particles:
            destroy(p[0])
        self.particles.clear()

    def _enforce_cap(self, needed: int) -> None:
        while len(self.particles) + needed > self.MAX_PARTICLES and self.particles:
            oldest = self.particles.pop(0)
            destroy(oldest[0])

    def spawn_dust_puff(self, x: float, z: float, count: int = 4) -> None:
        self._enforce_cap(count)
        for _ in range(count):
            ang = float(self.rng.uniform(0, 2 * math.pi))
            spd = float(self.rng.uniform(0.6, 1.9))
            vx = math.cos(ang) * spd
            vz = math.sin(ang) * spd
            vy = float(self.rng.uniform(1.2, 2.6))
            sc = float(self.rng.uniform(0.16, 0.28))

            ent = Entity(
                model="sphere",
                position=(x + self.rng.uniform(-0.2, 0.2), 0.12, z + self.rng.uniform(-0.2, 0.2)),
                scale=(sc, sc, sc),
                color=rgba(165, 235, 205, 210),
                unlit=True,
            )
            self.particles.append([ent, vx, vy, vz, 0.0, 0.42, -1.2, sc])

    def spawn_combat_sparks(self, x: float, z: float, count: int = 8) -> None:
        self._enforce_cap(count)
        spark_colors = [
            rgb(255, 200, 40),
            rgb(255, 110, 30),
            rgb(244, 63, 94),
        ]
        for i in range(count):
            ang = float(self.rng.uniform(0, 2 * math.pi))
            spd = float(self.rng.uniform(2.2, 5.2))
            vx = math.cos(ang) * spd
            vz = math.sin(ang) * spd
            vy = float(self.rng.uniform(3.0, 6.2))
            sc = float(self.rng.uniform(0.10, 0.18))

            ent = Entity(
                model="cube",
                position=(x, 0.42, z),
                scale=(sc, sc, sc),
                rotation=(self.rng.uniform(0, 180), self.rng.uniform(0, 180), 0),
                color=spark_colors[i % len(spark_colors)],
                unlit=True,
            )
            self.particles.append([ent, vx, vy, vz, 0.0, 0.48, 13.5, sc])

    def spawn_energy_orbs(self, x: float, z: float, count: int = 6) -> None:
        self._enforce_cap(count)
        for _ in range(count):
            ang = float(self.rng.uniform(0, 2 * math.pi))
            rad = float(self.rng.uniform(0.3, 0.9))
            vx = -math.sin(ang) * 1.2
            vz = math.cos(ang) * 1.2
            vy = float(self.rng.uniform(2.0, 3.8))
            sc = float(self.rng.uniform(0.14, 0.22))

            ent = Entity(
                model="sphere",
                position=(x + math.cos(ang) * rad, 0.25, z + math.sin(ang) * rad),
                scale=(sc, sc, sc),
                color=rgb(52, 211, 153),
                unlit=True,
            )
            self.particles.append([ent, vx, vy, vz, 0.0, 0.55, 0.0, sc])

    def update(self, dt: float) -> None:
        alive_list: List[List[Any]] = []
        for p in self.particles:
            ent, vx, vy, vz, age, max_age, grav, init_sc = p
            age += dt
            if age >= max_age:
                destroy(ent)
                continue

            vy -= grav * dt
            ent.x += vx * dt
            ent.y = max(0.03, ent.y + vy * dt)
            ent.z += vz * dt

            ratio = 1.0 - (age / max_age)
            cur_sc = max(0.02, init_sc * ratio)
            ent.scale = (cur_sc, cur_sc, cur_sc)

            p[2] = vy
            p[4] = age
            alive_list.append(p)

        self.particles = alive_list


# ==============================================================================
# 4. TÖBBKOMPONENSŰ 3D ROBOTPORSZÍVÓ MODELL (RobotVisual3D)
# ==============================================================================
STATE_URSINA_COLORS = {
    "WAITING_HELP": rgb(244, 114, 182),
    "CLEANING": rgb(56, 189, 248),
    "COOPERATING": rgb(45, 212, 191),
    "HUNTING": rgb(244, 63, 94),
    "FLEEING": rgb(250, 204, 21),
    "RETURNING_BASE": rgb(168, 85, 247),
    "SEEKING_CHARGER": rgb(251, 146, 60),
    "HELPING_ALLY": rgb(34, 197, 94),
    "MOVING_TO_GOAL": rgb(96, 165, 250),
    "REACHED_GOAL": rgb(16, 185, 129),
    "DISABLED": rgb(90, 100, 115),
}

STATE_HU_LABELS = {
    "WAITING_HELP": "Segítségre vár",
    "CLEANING": "Takarít",
    "COOPERATING": "Együttműködik",
    "HUNTING": "Vadászik",
    "FLEEING": "Menekül",
    "RETURNING_BASE": "Hazatér",
    "SEEKING_CHARGER": "Töltőt keres",
    "HELPING_ALLY": "Társat ment",
    "MOVING_TO_GOAL": "Cél felé tart",
    "REACHED_GOAL": "Célba ért",
    "DISABLED": "Kiesett",
}

TEAM_URSINA_COLORS = {
    0: rgb(225, 232, 240),
    1: rgb(59, 130, 246),
    2: rgb(244, 63, 94),
    3: rgb(16, 185, 129),
}

MODE_CATALOG: Dict[str, Dict[str, str]] = {
    "CLASSIC": {
        "short": "Classic Mode",
        "title": "Classic Mode (Szabad 3D Szimuláció)",
        "subtitle": "Autonóm takarítás, lokális szenzorok, rádiókommunikáció és kannibál energiarablás.",
        "default_map": "COMPACT",
    },
    "ANTARCTIC": {
        "short": "Antarctic Expedition",
        "title": "Antarctic Expedition (3D Antarktisz)",
        "subtitle": "Juss el az A-bázisról a B-kutatóállomásra töltőállomás nélkül, 3D jégzónákon át!",
        "default_map": "ANTARCTIC",
    },
    "COOP": {
        "short": "Cooperative Cleanup",
        "title": "Cooperative Cleanup (3D Kooperatív)",
        "subtitle": "6 szektoros területfelosztás, feladatátadás, SOS energiaátadás és hazatérés.",
        "default_map": "INDUSTRIAL",
    },
    "TDM": {
        "short": "Team Deathmatch",
        "title": "Team Deathmatch (3D Csapatharc)",
        "subtitle": "2 vagy 3 csapat taktikai küzdelme: célpontmegjelölés, csapattárs-mentés és harc.",
        "default_map": "ARENA",
    },
}


class RobotVisual3D:
    """Részletes, többrészes 3D-s robotporszívó modell Ursina Engine-ben."""

    def __init__(self, agent: RobotAgent, robot_tex, use_lit_shader: bool, on_select_cb):
        self.agent = agent
        self.on_select_cb = on_select_cb
        shd = lit_with_shadows_shader if (use_lit_shader and lit_with_shadows_shader) else None

        self.root = Entity(position=(agent.pos[0], 0.0, agent.pos[1]))

        self.bumper = Entity(
            parent=self.root,
            model="sphere",
            scale=(1.44, 0.28, 1.44),
            position=(0, 0.14, 0),
            color=rgb(38, 44, 54),
            shader=shd,
            collider="box",
        )
        self.bumper.on_click = lambda: self.on_select_cb(self.agent.id)

        self.click_proxy = Entity(
            parent=self.root,
            model="cube",
            scale=(1.6, 0.9, 1.6),
            position=(0, 0.42, 0),
            color=rgba(0, 0, 0, 0),
            collider="box",
        )
        self.click_proxy.on_click = lambda: self.on_select_cb(self.agent.id)

        body_col = TEAM_URSINA_COLORS.get(agent.team_id, rgb(225, 232, 240))
        self.chassis = Entity(
            parent=self.root,
            model="sphere",
            texture=robot_tex,
            scale=(1.30, 0.34, 1.30),
            position=(0, 0.24, 0),
            color=body_col,
            shader=shd,
        )

        self.led_ring = Entity(
            parent=self.root,
            model="sphere",
            scale=(0.92, 0.12, 0.92),
            position=(0, 0.40, 0),
            color=STATE_URSINA_COLORS.get(agent.state, rgb(56, 189, 248)),
            unlit=True,
        )

        self.lidar_dome = Entity(
            parent=self.root,
            model="sphere",
            scale=(0.46, 0.28, 0.46),
            position=(0, 0.48, 0),
            color=rgb(30, 38, 50),
            shader=shd,
        )

        self.front_visor = Entity(
            parent=self.root,
            model="cube",
            scale=(0.38, 0.14, 0.52),
            position=(0.54, 0.30, 0.0),
            color=rgb(235, 248, 255),
            unlit=True,
        )

        self.side_brush = Entity(
            parent=self.root,
            model="cube",
            scale=(0.55, 0.04, 0.10),
            position=(0.48, 0.08, 0.48),
            color=rgb(16, 185, 129),
            unlit=True,
        )

        # A referencia képen látható fehér kijelölő gyűrű
        self.selection_ring = Entity(
            parent=self.root,
            model="sphere",
            scale=(2.35, 0.05, 2.35),
            position=(0, 0.04, 0),
            color=rgb(245, 250, 255),
            unlit=True,
            enabled=False,
        )

        self.label_3d = Text(
            parent=self.root,
            text=f"{agent.name}",
            position=(0, 1.35, 0),
            origin=(0, 0),
            scale=5.4,
            color=UI.TITLE,
            billboard=True,
        )

    def update_visual(
        self, dt: float, is_selected: bool, show_all_labels: bool, in_fpv: bool = False, show_role: bool = True
    ) -> None:
        ag = self.agent
        self.root.position = Vec3(float(ag.render_pos[0]), 0.0, float(ag.render_pos[1]))
        self.root.rotation_y = -math.degrees(ag.render_angle)

        self.selection_ring.enabled = is_selected and not in_fpv

        if not ag.alive:
            self.led_ring.color = rgb(90, 95, 105)
            self.chassis.color = rgb(75, 80, 90)
            self.front_visor.color = rgb(180, 40, 50)
            self.label_3d.enabled = is_selected and not in_fpv
            if is_selected:
                self.label_3d.text = f"{ag.name} [KIESETT]"
                self.label_3d.color = rgb(244, 63, 94)
            return

        self.led_ring.color = STATE_URSINA_COLORS.get(ag.state, rgb(56, 189, 248))
        self.side_brush.rotation_y += dt * 520.0

        show_lbl = (is_selected or show_all_labels or ag.state in ("HUNTING", "HELPING_ALLY")) and not (
            is_selected and in_fpv
        )
        self.label_3d.enabled = show_lbl
        if show_lbl:
            tag = f" {ROLE_SHORT.get(getattr(ag, 'role_id', ''), '')}" if show_role else ""
            if is_selected and not show_all_labels:
                self.label_3d.text = f"{ag.name}{tag}"
                self.label_3d.color = UI.TITLE
            else:
                self.label_3d.text = f"{ag.name}{tag} [{ag.energy:.0f}%]"
                self.label_3d.color = (
                    UI.TITLE
                    if is_selected
                    else (rgb(52, 211, 153) if ag.energy_ratio > 0.35 else rgb(248, 113, 113))
                )

    def destroy_visual(self) -> None:
        destroy(self.root)


# ==============================================================================
# 5. TELJESÍTMÉNYMÉRŐ (PerformanceMonitor)
# ==============================================================================
class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.frame_times: deque = deque(maxlen=window_size)
        self.tick_timestamps: deque = deque(maxlen=90)
        self.ai_decisions_window: deque = deque(maxlen=90)
        self.current_fps: float = 60.0
        self.avg_fps: float = 60.0
        self.min_fps: float = 60.0
        self.frame_ms: float = 16.6
        self.tps: float = 0.0
        self.decisions_per_sec: float = 0.0

    def record_frame(self, dt: float) -> None:
        dt_c = max(0.0005, dt)
        self.frame_times.append(dt_c)
        self.current_fps = 1.0 / dt_c
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        max_dt = max(self.frame_times)
        self.avg_fps = 1.0 / avg_dt
        self.min_fps = 1.0 / max_dt
        self.frame_ms = avg_dt * 1000.0

    def record_logical_tick(self, ai_decisions: int) -> None:
        now = time.perf_counter()
        self.tick_timestamps.append(now)
        self.ai_decisions_window.append((now, ai_decisions))

    def update_rates(self) -> None:
        now = time.perf_counter()
        while self.tick_timestamps and now - self.tick_timestamps[0] > 1.5:
            self.tick_timestamps.popleft()
        while self.ai_decisions_window and now - self.ai_decisions_window[0][0] > 1.5:
            self.ai_decisions_window.popleft()

        if len(self.tick_timestamps) >= 2:
            span = max(0.1, now - self.tick_timestamps[0])
            self.tps = len(self.tick_timestamps) / span
            self.decisions_per_sec = sum(d for _, d in self.ai_decisions_window) / span
        else:
            self.tps = 0.0
            self.decisions_per_sec = 0.0


# ==============================================================================
# 6. FŐ 3D URSINA SZIMULÁCIÓS ALKALMAZÁS
# ==============================================================================
class UrsinaSimulationController:
    GRID_W = 88
    GRID_H = 60
    SPEED_LEVELS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    BASE_TICKS_PER_SEC = 12.0

    @property
    def max_steps(self) -> int:
        return int(self.cfg_max_steps * max(1, self.cfg_world_scale))

    BASE_W = 88
    BASE_H = 60
    CAM_MIN_DIST = 14.0
    RMB_DRAG_THRESHOLD = 0.012
    CAM_MAX_DIST = 115.0

    def __init__(self):
        tex_dir = ensure_3d_textures()

        self.perf = PerformanceMonitor()
        self.particles = ParticleSystem3D()

        self.selected_mode: str = "COOP"
        self.selected_map: str = "INDUSTRIAL"
        self.cfg_pop_size: int = 20
        self.cfg_dirt_rate: int = 2
        self.cfg_mutation_rate: float = 0.15
        self.cfg_max_steps: int = 600
        self.cfg_world_scale: int = 1
        self.cfg_seed: int = 42
        self.cfg_num_teams: int = 2
        self.cfg_start_energy: float = 100.0
        self.use_lit_shader: bool = True
        self.show_all_3d_labels: bool = False
        self.show_profiler: bool = True

        self.sound = SoundManager(os.path.join(SCRIPT_DIR, "assets", "sounds"), os.path.join(SCRIPT_DIR, "settings.json"))
        self.show_role_tags: bool = True
        self._snd_cleaned: int = 0
        self._low_energy_notified: set = set()

        self.cam_speed: float = 1.0
        self.zoom_speed: float = 1.0
        self.edge_scroll: bool = False
        self._drag: Optional[Dict[str, Any]] = None

        self._layout_key = None
        self._fs_prev: bool = False
        self._pending_fullscreen = None

        self.in_menu: bool = True
        self.paused: bool = False
        self.speed_idx: int = 3  # 1.0x
        self.tick_accumulator: float = 0.0
        self.current_step: int = 0
        self.show_summary_modal: bool = False
        self.last_summary_text: str = ""
        self.status_banner: str = "3D Ursina Engine + FPV Kamera & Részecskemotor kész."

        self.camera_mode: str = "STRATEGIC"
        self.cam_pivot = Vec3(self.GRID_W / 2.0, 0.0, self.GRID_H / 2.0)
        self.cam_distance: float = 68.0
        self.cam_yaw: float = 40.0
        self.cam_pitch: float = 44.0
        self.fpv_smooth_angle: float = 0.0
        self._saved_outer: Optional[OuterCameraState] = None
        self._fpv_agent = None
        self._outer_fov: float = camera_rig.OUTER_FOV_DEFAULT
        self._outer_near: float = camera_rig.OUTER_NEAR_CLIP

        self.selected_agent_id: int = 0
        self.inspector_tab: str = "DECISION"
        self.ev_filter: str = "MIND"
        self.chat_filter: str = "MIND"
        self.bottom_log_mode: str = "CHAT"  # "CHAT" vagy "EVENT"
        self.chat_scroll_offset: int = 0
        self.visible_chat_senders: List[int] = []

        self.map_entities: List[Entity] = []
        self.dirt_entities: Dict[Tuple[int, int], Entity] = {}
        self.robot_visuals: List[RobotVisual3D] = []
        self.beam_entities: List[Tuple[Entity, List[float]]] = []
        self.ring_entities: List[Tuple[Entity, List[float]]] = []
        self.target_marker_3d: Optional[Entity] = None
        self.target_line_3d: Optional[Entity] = None

        self.textures = {
            name: Texture(os.path.join(tex_dir, name))
            for name in [
                "floor_concrete.png",
                "floor_parquet.png",
                "floor_ice.png",
                "floor_industrial.png",
                "floor_arena.png",
                "wall_industrial.png",
                "wall_hazard.png",
                "wall_brick.png",
                "rack_metal.png",
                "robot_chassis.png",
            ]
        }

        self.env: Optional[SimulationEnvironment] = None
        self.ea: Optional[EvolutionaryAlgorithm] = None

        self._setup_lighting()
        self._build_hud_ui()
        self._build_main_menu_ui()
        self._ensure_cursor()
        self._refresh_sound_labels()
        self._poll_window_layout()
        self.start_simulation(keep_ea=False)
        self.open_main_menu()
        self._initialized = True

    @property
    def speed_multiplier(self) -> float:
        return self.SPEED_LEVELS[self.speed_idx]

    def _setup_lighting(self) -> None:
        self.ambient_light = AmbientLight(color=rgba(165, 180, 195, 255))
        self.dir_light = DirectionalLight(y=35, z=20, shadows=False)
        self.dir_light.look_at(Vec3(-0.4, -1.0, -0.35))

    def _apply_theme_lighting(self, theme: str) -> None:
        window.color = UI.BG
        if theme == "ANTARCTIC":
            self.ambient_light.color = rgba(175, 205, 235, 255)
        elif theme == "ARENA":
            self.ambient_light.color = rgba(165, 160, 195, 255)
        else:
            self.ambient_light.color = rgba(165, 180, 195, 255)

    def _clear_3d_scene(self) -> None:
        self.particles.clear_all()

        for e in self.map_entities:
            destroy(e)
        self.map_entities.clear()

        for e in self.dirt_entities.values():
            destroy(e)
        self.dirt_entities.clear()

        for rv in self.robot_visuals:
            rv.destroy_visual()
        self.robot_visuals.clear()

        for e, _ in self.beam_entities:
            destroy(e)
        self.beam_entities.clear()

        for e, _ in self.ring_entities:
            destroy(e)
        self.ring_entities.clear()

        if self.target_marker_3d:
            destroy(self.target_marker_3d)
            self.target_marker_3d = None
        if self.target_line_3d:
            destroy(self.target_line_3d)
            self.target_line_3d = None

    def _build_3d_world(self) -> None:
        self._clear_3d_scene()
        m_cfg = MAP_CATALOG[self.env.map_key]
        self._apply_theme_lighting(m_cfg["theme"])
        shd = lit_with_shadows_shader if (self.use_lit_shader and lit_with_shadows_shader) else None

        w, h = self.env.width, self.env.height
        floor_tex = self.textures[m_cfg["floor_tex"]]

        # Sötét, letisztult alaplap smaragdzöld szegéllyel a referencia kép alapján
        base_border = Entity(
            model="cube",
            position=(w / 2.0, -0.30, h / 2.0),
            scale=(w + 0.8, 0.45, h + 0.8),
            color=rgb(24, 110, 88),
            unlit=True,
        )
        base_floor = Entity(
            model="cube",
            position=(w / 2.0, -0.25, h / 2.0),
            scale=(w, 0.5, h),
            texture=floor_tex,
            texture_scale=(w / 8.0, h / 8.0),
            color=rgb(28, 38, 50),
            shader=shd,
        )
        self.map_entities.extend([base_border, base_floor])

        zone_colors = {
            1: rgba(26, 98, 78, 210),
            2: rgba(34, 88, 138, 210),
            3: rgba(75, 48, 88, 195),
        }
        for z_val, z_col in zone_colors.items():
            for x0, y0, rw, rh in extract_merged_rectangles(self.env.floor_grid, z_val):
                z_ent = Entity(
                    model="cube",
                    position=(x0 + rw / 2.0, 0.015, y0 + rh / 2.0),
                    scale=(rw, 0.03, rh),
                    color=z_col,
                    unlit=True,
                )
                self.map_entities.append(z_ent)

        hz_col = (
            rgba(70, 175, 230, 165)
            if m_cfg["theme"] == "ANTARCTIC"
            else rgba(120, 52, 68, 175)
        )
        for x0, y0, rw, rh in extract_merged_rectangles(self.env.hazard_grid, 1):
            hz_ent = Entity(
                model="sphere",
                position=(x0 + rw / 2.0, 0.025, y0 + rh / 2.0),
                scale=(rw, 0.04, rh),
                color=hz_col,
                unlit=True,
            )
            self.map_entities.append(hz_ent)

        # Belső és külső falak smaragdzöld-türkiz izometrikus stílusban
        for x0, y0, rw, rh in extract_merged_rectangles(self.env.obstacle_grid, 1):
            is_outer = x0 == 0 or y0 == 0 or (x0 + rw >= w) or (y0 + rh >= h)
            if is_outer:
                wall_h = 0.35
                w_ent = Entity(
                    model="cube",
                    position=(x0 + rw / 2.0, wall_h / 2.0, y0 + rh / 2.0),
                    scale=(rw, wall_h, rh),
                    color=rgb(30, 115, 92),
                    unlit=True,
                )
                self.map_entities.append(w_ent)
            else:
                wall_h = 2.6
                w_ent = Entity(
                    model="cube",
                    position=(x0 + rw / 2.0, wall_h / 2.0, y0 + rh / 2.0),
                    scale=(rw, wall_h, rh),
                    color=rgb(38, 166, 128),
                    shader=shd,
                )
                cap_ent = Entity(
                    model="cube",
                    position=(x0 + rw / 2.0, wall_h + 0.05, y0 + rh / 2.0),
                    scale=(rw + 0.04, 0.10, rh + 0.04),
                    color=rgb(72, 219, 172),
                    unlit=True,
                )
                self.map_entities.extend([w_ent, cap_ent])

        for x0, y0, rw, rh in extract_merged_rectangles(self.env.obstacle_grid, 2):
            rack_h = 2.1
            r_ent = Entity(
                model="cube",
                position=(x0 + rw / 2.0, rack_h / 2.0, y0 + rh / 2.0),
                scale=(rw, rack_h, rh),
                color=rgb(34, 150, 116),
                shader=shd,
            )
            rcap = Entity(
                model="cube",
                position=(x0 + rw / 2.0, rack_h + 0.04, y0 + rh / 2.0),
                scale=(rw + 0.04, 0.08, rh + 0.04),
                color=rgb(64, 205, 160),
                unlit=True,
            )
            self.map_entities.extend([r_ent, rcap])

        bx, bz = float(self.env.base_pos[0]), float(self.env.base_pos[1])
        base_pad = Entity(
            model="sphere",
            position=(bx, 0.03, bz),
            scale=(4.0, 0.06, 4.0),
            color=rgba(16, 185, 129, 180),
            unlit=True,
        )
        self.map_entities.append(base_pad)

        if self.env.mode in ("ANTARCTIC", "TDM") or self.env.map_key in ("ANTARCTIC", "WAREHOUSE", "ARENA"):
            gx, gz = float(self.env.goal_pos[0]), float(self.env.goal_pos[1])
            goal_pad = Entity(
                model="sphere",
                position=(gx, 0.03, gz),
                scale=(4.0, 0.06, 4.0),
                color=rgba(56, 189, 248, 180),
                unlit=True,
            )
            self.map_entities.append(goal_pad)

        for c_pos in self.env.chargers:
            cx, cz = float(c_pos[0]), float(c_pos[1])
            dock_base = Entity(
                model="sphere",
                position=(cx, 0.04, cz),
                scale=(2.4, 0.08, 2.4),
                color=rgba(16, 185, 129, 200),
                unlit=True,
            )
            self.map_entities.append(dock_base)

        self.target_marker_3d = Entity(
            model="sphere",
            scale=(1.1, 0.08, 1.1),
            color=rgb(52, 211, 153),
            unlit=True,
            enabled=False,
        )
        self.target_line_3d = Entity(
            model="cube",
            scale=(0.10, 0.04, 0.08),
            color=rgba(52, 211, 153, 180),
            unlit=True,
            enabled=False,
        )

        robot_tex = self.textures["robot_chassis.png"]
        for agent in self.ea.agents:
            rv = RobotVisual3D(agent, robot_tex, self.use_lit_shader, self._on_robot_clicked)
            self.robot_visuals.append(rv)

        self._sync_3d_dirt(spawn_dust=False)

    def _sync_3d_dirt(self, spawn_dust: bool = True) -> None:
        sig = (self.env.total_dirt_spawned, self.env.total_dirt_cleaned, int(self.env.tick_count == 0))
        if sig == getattr(self, "_dirt_signature", None):
            return
        self._dirt_signature = sig
        active_coords = set(map(tuple, np.argwhere(self.env.dirt_grid > 0)))

        to_remove = [coord for coord in self.dirt_entities if coord not in active_coords]
        for coord in to_remove:
            destroy(self.dirt_entities.pop(coord))
            if spawn_dust:
                self.particles.spawn_dust_puff(coord[0] + 0.5, coord[1] + 0.5, count=4)

        for coord in active_coords:
            if coord not in self.dirt_entities:
                x, y = coord
                intensity = int(self.env.dirt_grid[x, y])
                diam = 0.55 + 0.22 * intensity
                h_d = 0.12 + 0.06 * intensity
                d_col = (
                    rgb(245, 158, 11)
                    if intensity == 1
                    else (rgb(168, 85, 247) if intensity == 2 else rgb(52, 211, 153))
                )
                d_ent = Entity(
                    model="sphere",
                    position=(x + 0.5, 0.06, y + 0.5),
                    scale=(diam, h_d, diam),
                    color=d_col,
                    unlit=True,
                )
                self.dirt_entities[coord] = d_ent

    def _spawn_3d_beam(self, p1: np.ndarray, p2: np.ndarray, beam_col, duration: float = 0.32) -> None:
        v1 = Vec3(float(p1[0]), 0.35, float(p1[1]))
        v2 = Vec3(float(p2[0]), 0.35, float(p2[1]))
        mid = (v1 + v2) * 0.5
        dist = max(0.1, float(np.linalg.norm(p2 - p1)))
        ang = math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))

        beam = Entity(
            model="cube",
            position=mid,
            scale=(dist, 0.12, 0.12),
            rotation_y=-ang,
            color=beam_col,
            unlit=True,
        )
        self.beam_entities.append((beam, [0.0, duration]))

    def _spawn_3d_comm_ring(self, pos: np.ndarray, duration: float = 0.45) -> None:
        ring = Entity(
            model="sphere",
            position=(float(pos[0]), 0.08, float(pos[1])),
            scale=(1.2, 0.03, 1.2),
            color=rgba(52, 211, 153, 150),
            unlit=True,
        )
        self.ring_entities.append((ring, [0.0, duration]))

    # ==========================================================================
    # KÁRTYA- ÉS GOMBÉPÍTŐ SEGÉDEK A REFERENCIA KÉP DIZÁJNJÁHOZ
    # ==========================================================================
    @staticmethod
    def _make_card(parent, cx: float, cy: float, w: float, h: float, bg=UI.PANEL, edge=UI.PANEL_EDGE, z: float = 0.0):
        border = Entity(
            parent=parent,
            model="quad",
            color=edge,
            scale=(w + 0.004, h + 0.004),
            position=(cx, cy, z),
        )
        fill = Entity(
            parent=parent,
            model="quad",
            color=bg,
            scale=(w, h),
            position=(cx, cy, z - 0.001),
        )
        return fill, border

    
    def _make_btn(
        self,
        parent,
        lbl: str,
        pos: Tuple[float, float],
        sc: Tuple[float, float],
        cb,
        col=UI.BTN,
        hover=UI.BTN_HOVER,
        edge=UI.PANEL_EDGE,
        text_col=UI.TEXT,
        text_scale: float = 0.62,
        z: float = -0.01,
    ):
        if edge is not None:
            Entity(
                parent=parent,
                model="quad",
                color=edge,
                scale=(sc[0] + 0.003, sc[1] + 0.003),
                position=(pos[0], pos[1], z + 0.001),
            )

        btn = UButton(
            parent=parent,
            text=lbl if lbl else " ",
            position=(pos[0], pos[1], z),
            scale=sc,
            color=col,
            highlight_color=col,  # Saját hover kezelést használunk
            pressed_color=UI.BTN_PRESSED,
        )

        if btn.text_entity is not None:
            btn.text_entity.scale *= fit_text_scale(lbl, text_scale, sc[0])
            btn.text_entity.color = text_col

            original_text_scale = btn.text_entity.scale
            original_color = btn.color

            def on_hover():
                btn.text_entity.scale = original_text_scale
                btn.color = tuple(c * 0.88 for c in original_color)

            def on_unhover():
                btn.text_entity.scale = original_text_scale
                btn.color = original_color

            btn.on_mouse_enter = on_hover
            btn.on_mouse_exit = on_unhover

        btn.on_click = self._clicked(cb)

        return btn


    # ==========================================================================
    # HUD FELÉPÍTÉSE A REFERENCIA KÉP ALAPJÁN
    # ==========================================================================
    def _build_hud_ui(self) -> None:
        self.hud_root = Entity(parent=camera.ui)
        self.gui = GuiManager(on_change=self._on_gui_state_changed)

        def new_panel(name: str, anchor: str, **kw) -> Entity:
            root = Entity(parent=self.hud_root)
            self.gui.add_panel(name, root, anchor, **kw)
            return root

        # ------------------------------------------------------------ 1. BAL FELSŐ SZIGET (topbar)
        top_root = new_panel(
            "topbar",
            "TL",
            title="Küldetés / felső sáv",
            rects=[(-0.62, 0.455, 0.50, 0.054)],
            on_layout=self._layout_topbar,
        )
        self.top_bar_bg, _ = self._make_card(top_root, -0.62, 0.455, 0.50, 0.054)
        self.top_mode_text = Text(
            parent=top_root,
            text="Cooperative Cleanup",
            position=(-0.855, 0.463, -0.01),
            scale=0.58,
            color=UI.TITLE,
        )
        self.top_map_text = Text(
            parent=top_root,
            text="5. Industrial Facility",
            position=(-0.655, 0.463, -0.01),
            scale=0.56,
            color=UI.TEXT_DIM,
        )
        self.top_gen_text = Text(
            parent=top_root,
            text="Generáció #1",
            position=(-0.475, 0.463, -0.01),
            scale=0.56,
            color=UI.TEXT_DIM,
        )
        # Kompatibilitási szövegmező
        self.top_bar_text = Text(parent=top_root, text=" ", position=(-0.85, 0.46, -0.01), scale=0.55, enabled=False)

        # ------------------------------------------------------------ 2. FELSŐ KÖZÉPSŐ SZIGET (speed)
        spd = new_panel(
            "speed",
            "TC",
            title="Sebességszabályozó",
            rects=[(-0.02, 0.455, 0.62, 0.054)],
        )
        self._make_card(spd, -0.02, 0.455, 0.62, 0.054)
        self.btn_pause = self._make_btn(
            spd,
            "Szünet",
            (-0.278, 0.455),
            (0.086, 0.036),
            self.toggle_pause,
            col=UI.PRIMARY,
            hover=UI.PRIMARY_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=rgb(8, 20, 16),
            text_scale=0.60,
        )
        self.btn_step = self._make_btn(
            spd, "1 tick", (-0.184, 0.455), (0.074, 0.036), self.step_once, text_scale=0.58
        )
        self.btn_spd_m = self._make_btn(
            spd, "-", (-0.120, 0.455), (0.032, 0.032), self.speed_down, text_scale=0.68
        )
        self.spd_text = Text(
            parent=spd,
            text=f"{self.speed_multiplier}x",
            position=(-0.078, 0.455, -0.012),
            origin=(0, 0),
            scale=0.60,
            color=UI.WARN,
        )
        self.btn_spd_p = self._make_btn(
            spd, "+", (-0.036, 0.455), (0.032, 0.032), self.speed_up, text_scale=0.68
        )
        self.tick_progress_label = Text(
            parent=spd,
            text="Tick 0 / 600",
            position=(0.005, 0.469, -0.012),
            scale=0.50,
            color=UI.TEXT_DIM,
        )
        self.tick_progress_bar = ProgressBarUI(
            spd, left_x=0.005, cy=0.443, width=0.265, height=0.006, fg_color=UI.PRIMARY
        )

        # ------------------------------------------------------------ 3. JOBB FELSŐ SZIGET (profiler)
        prof = new_panel(
            "profiler",
            "TR",
            title="Teljesítmény és Menü (F3)",
            default_visible=self.show_profiler,
            rects=[(0.68, 0.455, 0.38, 0.054)],
        )
        self.profiler_bg, _ = self._make_card(prof, 0.68, 0.455, 0.38, 0.054)
        self.profiler_text = Text(
            parent=prof,
            text="60 FPS   entitás 0   ütköző 0",
            position=(0.505, 0.463, -0.012),
            scale=0.54,
            color=UI.TEXT_DIM,
        )
        self.btn_menu = self._make_btn(
            prof,
            "Menü",
            (0.825, 0.455),
            (0.070, 0.036),
            self.open_main_menu,
            col=UI.DANGER_BG,
            hover=UI.DANGER_HOVER,
            edge=UI.DANGER_EDGE,
            text_col=UI.TEXT,
            text_scale=0.58,
        )

        # ------------------------------------------------------------ 4. FPV OVERLAY (középen)
        fpv_root = new_panel("fpv", "C", title="FPV szenzor-overlay")
        self.fpv_overlay = Entity(parent=fpv_root, enabled=False)
        Entity(
            parent=self.fpv_overlay,
            model="quad",
            color=rgba(52, 211, 153, 80),
            scale=(0.50, 0.003),
            position=(0.0, 0.04, -0.02),
        )
        Entity(
            parent=self.fpv_overlay,
            model="quad",
            color=rgba(52, 211, 153, 80),
            scale=(0.003, 0.34),
            position=(0.0, 0.04, -0.02),
        )
        self.fpv_hud_text = Text(
            parent=self.fpv_overlay,
            text=" ",
            position=(-0.32, 0.25, -0.03),
            scale=0.68,
            color=UI.TITLE,
        )

        # ------------------------------------------------------------ 5. BAL OSZLOP: STATISZTIKÁK (stats)
        sts = new_panel(
            "stats",
            "TL",
            title="Statisztikák",
            rects=[(-0.735, 0.265, 0.27, 0.27)],
        )
        self._make_card(sts, -0.735, 0.265, 0.27, 0.27)
        stat_labels = [
            "Élő / kiesett",
            "Takarított kosz",
            "Szennyezett góc",
            "Célban",
            "Energiaátadás",
            "Rádióüzenet",
        ]
        self.stat_value_texts: List[Text] = []
        for idx, s_lbl in enumerate(stat_labels):
            ry = 0.372 - idx * 0.041
            Text(
                parent=sts,
                text=s_lbl,
                position=(-0.855, ry, -0.012),
                scale=0.54,
                color=UI.TEXT_DIM,
            )
            val_t = Text(
                parent=sts,
                text="0",
                position=(-0.615, ry - 0.006, -0.012),
                origin=(0.5, 0),
                scale=0.60,
                color=UI.TEXT,
            )
            self.stat_value_texts.append(val_t)
            if idx < len(stat_labels) - 1:
                Entity(
                    parent=sts,
                    model="quad",
                    color=UI.DIVIDER,
                    scale=(0.24, 0.0015),
                    position=(-0.735, ry - 0.024, -0.011),
                )
        self.sidebar_stats_text = Text(parent=sts, text=" ", position=(-0.85, 0.38, -0.01), scale=0.5, enabled=False)

        # ------------------------------------------------------------ 6. BAL OSZLOP: KAMERA ÉS SZEKTOROK (controls)
        ctl = new_panel(
            "controls",
            "BL",
            title="Kamera és Szektorok",
            rects=[(-0.735, -0.030, 0.27, 0.265), (-0.735, -0.335, 0.27, 0.29)],
        )
        # Kamera kártya
        self._make_card(ctl, -0.735, -0.030, 0.27, 0.265)
        Text(
            parent=ctl,
            text="Kamera",
            position=(-0.855, 0.088, -0.012),
            scale=0.54,
            color=UI.TEXT_DIM,
        )
        self.btn_cam = self._make_btn(
            ctl,
            "Stratégiai nézet",
            (-0.735, 0.042),
            (0.242, 0.036),
            self.cycle_camera_mode,
            col=UI.ACTIVE_BG,
            hover=UI.ACTIVE_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=UI.TITLE,
            text_scale=0.58,
        )
        self.btn_follow = self._make_btn(
            ctl, "Követés", (-0.798, -0.012), (0.115, 0.044), self.toggle_follow, text_scale=0.56
        )
        self.btn_fit_map = self._make_btn(
            ctl, "Egész\npálya", (-0.672, -0.012), (0.115, 0.044), self.fit_whole_map, text_scale=0.54
        )
        self.btn_zoom_out = self._make_btn(
            ctl, "Zoom -", (-0.798, -0.066), (0.115, 0.034), lambda: self._zoom_camera(5.5), text_scale=0.56
        )
        self.btn_zoom_in = self._make_btn(
            ctl, "Zoom +", (-0.672, -0.066), (0.115, 0.034), lambda: self._zoom_camera(-5.5), text_scale=0.56
        )
        self.btn_sound_card = self._make_btn(
            ctl, "Hang: ki", (-0.798, -0.114), (0.115, 0.034), self.toggle_sound, text_scale=0.56
        )
        self.btn_vol_card = self._make_btn(
            ctl, "60%", (-0.672, -0.114), (0.115, 0.034), self.cycle_sound_volume, text_scale=0.56
        )

        # Szektorok kártya
        self._make_card(ctl, -0.735, -0.335, 0.27, 0.29)
        Text(
            parent=ctl,
            text="Szektorok",
            position=(-0.855, -0.205, -0.012),
            scale=0.54,
            color=UI.TEXT_DIM,
        )
        self.sector_bars: List[ProgressBarUI] = []
        for s_idx in range(6):
            scol = s_idx % 3
            srow = s_idx // 3
            scx = -0.818 + scol * 0.083
            scy = -0.252 - srow * 0.052
            self._make_card(ctl, scx, scy, 0.076, 0.042, bg=UI.PANEL_INNER, edge=UI.PANEL_EDGE, z=-0.005)
            Text(
                parent=ctl,
                text=f"#{s_idx + 1}",
                position=(scx - 0.030, scy + 0.012, -0.012),
                scale=0.52,
                color=UI.TEXT,
            )
            sbar = ProgressBarUI(
                ctl,
                left_x=scx - 0.030,
                cy=scy - 0.010,
                width=0.060,
                height=0.0055,
                fg_color=UI.PRIMARY,
                z=-0.012,
            )
            self.sector_bars.append(sbar)

        self.sector_summary_text = Text(
            parent=ctl,
            text="0 kész, 6 folyamatban, 0 lefoglalva",
            position=(-0.855, -0.352, -0.012),
            scale=0.46,
            color=UI.TEXT_DIM,
        )
        self.btn_gui_settings = self._make_btn(
            ctl, "Panelek", (-0.798, -0.405), (0.115, 0.038), self.toggle_settings_panel, text_scale=0.56
        )
        self.btn_plot = self._make_btn(
            ctl, "Grafikonok", (-0.672, -0.405), (0.115, 0.038), self.open_matplotlib, text_scale=0.56
        )

        # ------------------------------------------------------------ 7. JOBB OLDALI ROBOT INSPECTOR ÉS AKCIÓSÁV
        side_bg_root = new_panel(
            "sidebar_bg",
            "TR",
            title="Oldalsáv háttér",
            group="main",
            rects=[(0.675, 0.035, 0.39, 0.72), (0.675, -0.415, 0.39, 0.105)],
            depends_any=["inspector"],
        )
        self.sidebar_bg, _ = self._make_card(side_bg_root, 0.675, 0.035, 0.39, 0.72)
        self._make_card(side_bg_root, 0.675, -0.415, 0.39, 0.105)

        ins = new_panel("inspector", "TR", title="Robot adatlap", group="main")

        # Felső fülválasztó gombok: AI döntés | Genom és memória
        self.btn_tab_dec = self._make_btn(
            ins,
            "AI döntés",
            (0.582, 0.358),
            (0.174, 0.036),
            lambda: self.set_inspector_tab("DECISION"),
            col=UI.ACTIVE_BG,
            hover=UI.ACTIVE_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=UI.TITLE,
            text_scale=0.58,
        )
        self.btn_tab_gen = self._make_btn(
            ins,
            "Genom és memória",
            (0.768, 0.358),
            (0.174, 0.036),
            lambda: self.set_inspector_tab("GENOME"),
            col=UI.BTN,
            hover=UI.BTN_HOVER,
            edge=UI.PANEL_EDGE,
            text_col=UI.TEXT,
            text_scale=0.56,
        )

        # Robot név, állapot és szerep/személyiség
        self.insp_name_text = Text(
            parent=ins,
            text="R-01",
            position=(0.495, 0.318, -0.012),
            scale=0.98,
            color=UI.TEXT,
        )
        self.insp_state_text = Text(
            parent=ins,
            text="Együttműködik",
            position=(0.852, 0.298, -0.012),
            origin=(0.5, 0),
            scale=0.54,
            color=UI.TITLE,
        )
        self.insp_traits_text = Text(
            parent=ins,
            text="Takarító · Ragadozó · Óvatos",
            position=(0.495, 0.278, -0.012),
            scale=0.52,
            color=UI.TEXT_DIM,
        )

        # Akku és Fitnesz mini kártyák
        self._make_card(ins, 0.582, 0.208, 0.174, 0.082, bg=UI.PANEL_INNER, edge=UI.PANEL_EDGE, z=-0.005)
        self.insp_bat_val = Text(
            parent=ins,
            text="100%",
            position=(0.508, 0.236, -0.012),
            scale=0.66,
            color=UI.WARN,
        )
        Text(
            parent=ins,
            text="akku",
            position=(0.508, 0.208, -0.012),
            scale=0.48,
            color=UI.TEXT_DIM,
        )
        self.insp_bat_bar = ProgressBarUI(
            ins, left_x=0.508, cy=0.180, width=0.146, height=0.006, fg_color=UI.WARN, z=-0.014
        )

        self._make_card(ins, 0.768, 0.208, 0.174, 0.082, bg=UI.PANEL_INNER, edge=UI.PANEL_EDGE, z=-0.005)
        self.insp_fit_val = Text(
            parent=ins,
            text="0",
            position=(0.694, 0.236, -0.012),
            scale=0.66,
            color=UI.TEXT,
        )
        Text(
            parent=ins,
            text="fitnesz",
            position=(0.694, 0.208, -0.012),
            scale=0.48,
            color=UI.TEXT_DIM,
        )

        # --- AI DÖNTÉS FÜL TARTALMA
        self.insp_dec_root = Entity(parent=ins)
        Text(
            parent=self.insp_dec_root,
            text="Döntés",
            position=(0.495, 0.145, -0.012),
            scale=0.52,
            color=UI.TEXT_DIM,
        )
        self.insp_dec_headline = Text(
            parent=self.insp_dec_root,
            text="A társ által jelzett gócpont felé tart.",
            position=(0.495, 0.118, -0.012),
            scale=0.56,
            color=UI.TEXT,
        )
        self.insp_dec_sub = Text(
            parent=self.insp_dec_root,
            text="Cél 0 m-re, nincs lefoglalva.",
            position=(0.495, 0.088, -0.012),
            scale=0.48,
            color=UI.TEXT_DIM,
        )
        Entity(
            parent=self.insp_dec_root,
            model="quad",
            color=UI.DIVIDER,
            scale=(0.355, 0.0015),
            position=(0.675, 0.036, -0.011),
        )
        Text(
            parent=self.insp_dec_root,
            text="Mérlegelt alternatívák",
            position=(0.495, 0.022, -0.012),
            scale=0.52,
            color=UI.TEXT_DIM,
        )
        self.alt_rows: List[Tuple[Text, ProgressBarUI, Text]] = []
        for a_idx in range(4):
            ary = -0.008 - a_idx * 0.027
            a_lbl = Text(
                parent=self.insp_dec_root,
                text="-",
                position=(0.495, ary + 0.006, -0.012),
                scale=0.50,
                color=UI.TEXT,
            )
            a_bar = ProgressBarUI(
                self.insp_dec_root,
                left_x=0.692,
                cy=ary,
                width=0.118,
                height=0.006,
                fg_color=UI.PRIMARY,
                z=-0.014,
            )
            a_val = Text(
                parent=self.insp_dec_root,
                text="0",
                position=(0.852, ary, -0.012),
                origin=(0.5, 0),
                scale=0.50,
                color=UI.TEXT,
            )
            self.alt_rows.append((a_lbl, a_bar, a_val))

        Entity(
            parent=self.insp_dec_root,
            model="quad",
            color=UI.DIVIDER,
            scale=(0.355, 0.0015),
            position=(0.675, -0.112, -0.011),
        )
        Text(
            parent=self.insp_dec_root,
            text="Kommunikációs lánc",
            position=(0.495, -0.126, -0.012),
            scale=0.52,
            color=UI.TEXT_DIM,
        )
        self.comm_chain_texts: List[Text] = []
        for c_idx in range(3):
            cry = -0.156 - c_idx * 0.026
            ct = Text(
                parent=self.insp_dec_root,
                text="-",
                position=(0.495, cry, -0.012),
                scale=0.48,
                color=UI.TEXT_DIM,
            )
            self.comm_chain_texts.append(ct)

        # --- GENOM ÉS MEMÓRIA FÜL TARTALMA
        self.insp_gen_root = Entity(parent=ins, enabled=False)
        Text(
            parent=self.insp_gen_root,
            text="Genom profil (8 evolúciós gén)",
            position=(0.495, 0.145, -0.012),
            scale=0.52,
            color=UI.TEXT_DIM,
        )
        gene_labels = [
            "Agresszivitás",
            "Kockázatvállalás",
            "Mozgási sebesség",
            "Kosz-affinitás",
            "Fal-elkerülés",
            "Kooperáció",
            "Hazatérési óvatosság",
            "Kommunikációs hajlam",
        ]
        self.gene_rows: List[Tuple[ProgressBarUI, Text]] = []
        for g_idx, g_name in enumerate(gene_labels):
            gry = 0.115 - g_idx * 0.024
            Text(
                parent=self.insp_gen_root,
                text=g_name,
                position=(0.495, gry + 0.005, -0.012),
                scale=0.46,
                color=UI.TEXT,
            )
            g_bar = ProgressBarUI(
                self.insp_gen_root,
                left_x=0.692,
                cy=gry,
                width=0.115,
                height=0.0055,
                fg_color=UI.PRIMARY,
                z=-0.014,
            )
            g_val = Text(
                parent=self.insp_gen_root,
                text="50%",
                position=(0.852, gry, -0.012),
                origin=(0.5, 0),
                scale=0.46,
                color=UI.TEXT_DIM,
            )
            self.gene_rows.append((g_bar, g_val))

        self.insp_mem_text = Text(
            parent=self.insp_gen_root,
            text=" ",
            position=(0.495, -0.085, -0.012),
            scale=0.47,
            color=UI.TEXT_DIM,
        )

        # Háttér-kompatibilitási szövegmező
        self.inspector_text = Text(parent=ins, text=" ", position=(0.495, 0.12, -0.01), scale=0.5, enabled=False)

        # Jobb alsó 4 akciógomb (Stat CSV | Újra | Köv. gen. | Rádió)
        self.btn_csv = self._make_btn(
            ins, "Stat\nCSV", (0.536, -0.415), (0.084, 0.068), self.save_stats_csv, text_scale=0.54
        )
        self.btn_restart = self._make_btn(
            ins, "Újra", (0.628, -0.415), (0.084, 0.068), lambda: self.start_simulation(keep_ea=True), text_scale=0.56
        )
        self.btn_nextgen = self._make_btn(
            ins, "Köv.\ngen.", (0.720, -0.415), (0.084, 0.068), self.advance_generation, text_scale=0.54
        )
        self.btn_radio = self._make_btn(
            ins,
            "Rádió",
            (0.812, -0.415),
            (0.084, 0.068),
            self.toggle_radio,
            col=UI.ACTIVE_BG,
            hover=UI.ACTIVE_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=UI.TITLE,
            text_scale=0.56,
        )
        self.btn_labels = self._make_btn(
            ins, "3D Név (L)", (0.812, -0.415), (0.01, 0.01), self.toggle_3d_labels
        )
        self.btn_labels.enabled = False

        # ------------------------------------------------------------ 8. ALSÓ ROBOT CHAT LOG KÁRTYA (chat)
        chat = new_panel(
            "chat",
            "BL",
            title="Chat log / eseménynapló",
            rects=[(-0.055, -0.385, 1.03, 0.195)],
        )
        self.bottom_bg, _ = self._make_card(chat, -0.055, -0.385, 1.03, 0.195)
        self.chat_header_text = Text(
            parent=chat,
            text="Robot chat log",
            position=(-0.545, -0.304, -0.012),
            scale=0.56,
            color=UI.TEXT_DIM,
        )
        Entity(
            parent=chat,
            model="quad",
            color=UI.DIVIDER,
            scale=(0.99, 0.0015),
            position=(-0.055, -0.328, -0.011),
        )

        self.btn_chat_filter = self._make_btn(
            chat,
            "Mind",
            (0.165, -0.308),
            (0.074, 0.028),
            self.cycle_chat_filter,
            col=UI.ACTIVE_BG,
            hover=UI.ACTIVE_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=UI.TITLE,
            text_scale=0.54,
        )
        self.btn_chat_mode = self._make_btn(
            chat,
            "Napló",
            (0.248, -0.308),
            (0.074, 0.028),
            self.toggle_bottom_log_mode,
            text_scale=0.54,
        )
        self.btn_chat_exp = self._make_btn(
            chat,
            "Mentés",
            (0.332, -0.308),
            (0.078, 0.028),
            self.save_chat_export,
            text_scale=0.54,
        )
        self.btn_chat_live = self._make_btn(
            chat,
            "Élő",
            (0.408, -0.308),
            (0.056, 0.028),
            lambda: self.scroll_chat(-9999),
            col=UI.PRIMARY,
            hover=UI.PRIMARY_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=rgb(8, 20, 16),
            text_scale=0.54,
        )

        self.chat_row_buttons: List[UButton] = []
        self.chat_row_LeftTexts: List[Text] = []
        self.chat_row_MidTexts: List[Text] = []
        self.chat_row_RightTexts: List[Text] = []

        for r_idx in range(6):
            ry = -0.348 - r_idx * 0.030
            r_btn = UButton(
                parent=chat,
                text=" ",
                position=(-0.055, ry, -0.010),
                scale=(0.99, 0.027),
                color=rgba(18, 25, 38, 0),
                highlight_color=rgba(28, 72, 62, 160),
                pressed_color=UI.BTN_PRESSED,
                enabled=(r_idx < 4),
            )
            if r_btn.text_entity is not None:
                r_btn.text_entity.enabled = False
            r_btn.on_click = self._clicked(lambda idx=r_idx: self.on_chat_row_clicked(idx))
            self.chat_row_buttons.append(r_btn)

            t_left = Text(
                parent=chat,
                text="",
                position=(-0.540, ry + 0.006, -0.013),
                scale=0.50,
                color=UI.TEXT_DIM,
                enabled=(r_idx < 4),
            )
            t_mid = Text(
                parent=chat,
                text="",
                position=(-0.455, ry + 0.006, -0.013),
                scale=0.52,
                color=UI.TEXT,
                enabled=(r_idx < 4),
            )
            t_right = Text(
                parent=chat,
                text="",
                position=(0.355, ry + 0.006, -0.013),
                scale=0.50,
                color=UI.PRIMARY,
                enabled=(r_idx < 4),
            )
            self.chat_row_LeftTexts.append(t_left)
            self.chat_row_MidTexts.append(t_mid)
            self.chat_row_RightTexts.append(t_right)

        self.event_log_text = Text(
            parent=chat,
            text=" ",
            position=(-0.540, -0.342, -0.013),
            scale=0.52,
            color=UI.TEXT_DIM,
            enabled=False,
        )

        # ------------------------------------------------------------ 9. FORDULÓ-ÖSSZEGZŐ MODAL (summary)
        summ = new_panel("summary", "C", title="Forduló-összegző", hideable=False)
        self.summary_modal = Entity(
            parent=summ,
            model="quad",
            color=UI.PANEL,
            scale=(0.86, 0.52),
            position=(0.0, 0.04, -0.05),
            enabled=False,
        )
        self.summary_text = Text(
            parent=summ,
            text=" ",
            position=(-0.39, 0.25, -0.06),
            scale=0.68,
            color=UI.TEXT,
            enabled=False,
        )

        # ------------------------------------------------------------ 10. PANELEK ÉS KAMERA BEÁLLÍTÁSOK MODAL (settings)
        set_root = new_panel(
            "settings",
            "C",
            title="GUI és kamera beállítások",
            default_visible=False,
            rects=[(0.0, 0.02, 0.46, 0.42)],
        )
        self._make_card(set_root, 0.0, 0.02, 0.46, 0.42, bg=UI.PANEL, edge=UI.ACTIVE_EDGE, z=-0.04)
        Text(
            parent=set_root,
            text="Panelek és Kamera Beállítások (O / Tab)",
            position=(-0.21, 0.205, -0.05),
            scale=0.56,
            color=UI.TITLE,
        )

        self.settings_toggle_btns: Dict[str, Tuple[Any, str]] = {}
        toggle_defs = [
            ("chat", "Chat log"),
            ("inspector", "Robot adat"),
            ("stats", "Statisztika"),
            ("topbar", "Küldetés"),
            ("profiler", "Teljesítmény"),
            ("controls", "Kamera/Szektor"),
            ("speed", "Időzítő"),
        ]
        for i, (pname, plabel) in enumerate(toggle_defs):
            cx = -0.105 if i % 2 == 0 else 0.105
            cy = 0.155 - (i // 2) * 0.036
            b = self._make_btn(
                set_root,
                f"{plabel}: BE",
                (cx, cy),
                (0.195, 0.028),
                lambda n=pname: self.toggle_panel_local(n),
                col=UI.ACTIVE_BG,
                hover=UI.ACTIVE_HOVER,
                edge=UI.ACTIVE_EDGE,
                text_scale=0.54,
                z=-0.05,
            )
            self.settings_toggle_btns[pname] = (b, plabel)

        self._make_btn(
            set_root,
            "Mind BE",
            (0.105, 0.155 - 3 * 0.036),
            (0.195, 0.028),
            self.show_all_local_panels,
            text_scale=0.54,
            z=-0.05,
        )

        self.cam_cfg_text = Text(
            parent=set_root,
            text=" ",
            position=(-0.21, 0.012, -0.05),
            scale=0.52,
            color=UI.WARN,
        )
        cam_btn_defs = [
            ("Mozgás -", (-0.105, -0.025), lambda: self.adj_cam("move", -0.25)),
            ("Mozgás +", (0.105, -0.025), lambda: self.adj_cam("move", 0.25)),
            ("Zoom seb. -", (-0.105, -0.061), lambda: self.adj_cam("zoom", -0.25)),
            ("Zoom seb. +", (0.105, -0.061), lambda: self.adj_cam("zoom", 0.25)),
        ]
        for lbl, pos, cb in cam_btn_defs:
            self._make_btn(set_root, lbl, pos, (0.195, 0.028), cb, text_scale=0.54, z=-0.05)

        self.btn_edge_scroll = self._make_btn(
            set_root,
            "Szélgörgetés: KI",
            (-0.105, -0.097),
            (0.195, 0.028),
            self.toggle_edge_scroll,
            text_scale=0.54,
            z=-0.05,
        )
        self.btn_follow_set = self._make_btn(
            set_root,
            "Követés: KI",
            (0.105, -0.097),
            (0.195, 0.028),
            self.toggle_follow,
            text_scale=0.54,
            z=-0.05,
        )
        self.btn_sound_set = self._make_btn(
            set_root,
            "Hang: BE",
            (-0.105, -0.133),
            (0.195, 0.028),
            self.toggle_sound,
            text_scale=0.54,
            z=-0.05,
        )
        self.btn_volume_set = self._make_btn(
            set_root,
            "Hangerő: 60%",
            (0.105, -0.133),
            (0.195, 0.028),
            self.cycle_sound_volume,
            text_scale=0.54,
            z=-0.05,
        )
        self._make_btn(
            set_root,
            "Bezárás (O)",
            (0.0, -0.170),
            (0.18, 0.028),
            self.toggle_settings_panel,
            col=UI.DANGER_BG,
            hover=UI.DANGER_HOVER,
            edge=UI.DANGER_EDGE,
            text_scale=0.54,
            z=-0.05,
        )

        # ------------------------------------------------------------ 11. OLDALSÓ FÜL (tab)
        tab_root = new_panel("tab", "CR", title="Oldalsó fül", default_visible=False)
        self.side_tab_btn = UButton(
            parent=tab_root,
            text=">",
            position=(0.875, 0.05),
            scale=(0.022, 0.10),
            color=UI.ACTIVE_BG,
            highlight_color=UI.ACTIVE_HOVER,
            pressed_color=UI.BTN_PRESSED,
        )
        self.side_tab_btn.on_click = self._clicked(self.toggle_side_panel)
        self.side_sound_btn = UButton(
            parent=tab_root,
            text="H",
            position=(0.875, -0.04),
            scale=(0.022, 0.04),
            color=UI.PRIMARY,
            highlight_color=UI.PRIMARY_HOVER,
            pressed_color=UI.BTN_PRESSED,
        )
        self.side_sound_btn.on_click = self.toggle_sound

        self.hint_text = Text(
            parent=camera.ui,
            text=" ",
            position=(0, -0.465, -0.2),
            origin=(0, 0),
            scale=0.72,
            color=UI.TEXT_DIM,
            enabled=False,
        )
        self.hint_timer: float = 0.0

        self.gui.refresh()

    def _layout_topbar(self, panel, aspect: float, s: float) -> None:
        """A bal felső információs kártya mindig a képernyőn belül marad."""
        panel.rects = [(-0.62, 0.455, 0.50, 0.054)]

    def _zoom_camera(self, delta: float) -> None:
        step = delta * self.zoom_speed
        self.cam_distance = float(np.clip(self.cam_distance + step, self.CAM_MIN_DIST, self.CAM_MAX_DIST))

    def toggle_bottom_log_mode(self) -> None:
        self.bottom_log_mode = "EVENT" if self.bottom_log_mode == "CHAT" else "CHAT"
        is_ev = self.bottom_log_mode == "EVENT"
        self.btn_chat_mode.color = UI.ACTIVE_BG if is_ev else UI.BTN
        if self.btn_chat_mode.text_entity is not None:
            self.btn_chat_mode.text_entity.color = UI.TITLE if is_ev else UI.TEXT

    # ---------------------------------------------------------------- GUI-vezérlés
    def _on_gui_state_changed(self) -> None:
        if not hasattr(self, "settings_toggle_btns"):
            return
        for pname, (btn, plabel) in self.settings_toggle_btns.items():
            on = self.gui.is_local(pname)
            btn.text = f"{plabel}: {'BE' if on else 'KI'}"
            btn.color = UI.ACTIVE_BG if on else UI.BTN
        if hasattr(self, "side_tab_btn"):
            self.side_tab_btn.text = "<" if self.gui.is_group_collapsed("main") else ">"
        self._refresh_cam_settings_labels()

    def _refresh_cam_settings_labels(self) -> None:
        if not hasattr(self, "cam_cfg_text"):
            return
        self.cam_cfg_text.text = f"Mozgás: {self.cam_speed:.2f}x   |   Zoom: {self.zoom_speed:.2f}x"
        self.btn_edge_scroll.text = f"Szélgörgetés: {'BE' if self.edge_scroll else 'KI'}"
        self.btn_edge_scroll.color = UI.ACTIVE_BG if self.edge_scroll else UI.BTN
        following = self.camera_mode == "FOLLOW"
        self.btn_follow_set.text = f"Követés: {'BE' if following else 'KI'}"
        self.btn_follow_set.color = UI.ACTIVE_BG if following else UI.BTN
        if hasattr(self, "btn_follow"):
            self.btn_follow.color = UI.ACTIVE_BG if following else UI.BTN
            if self.btn_follow.text_entity is not None:
                self.btn_follow.text_entity.color = UI.TITLE if following else UI.TEXT

    def toggle_gui(self) -> None:
        visible = self.gui.toggle_global()
        if visible:
            self.hint_text.enabled = False
            self.hint_timer = 0.0
            if self.ea is not None:
                self._update_hud_texts(self._get_selected_agent())
        else:
            self.hint_text.text = "Kezelőfelület elrejtve  -  Tab: visszaállítás"
            self.hint_text.enabled = True
            self.hint_timer = 3.0

    def toggle_side_panel(self) -> None:
        self.gui.toggle_group("main")

    def toggle_panel_local(self, name: str) -> None:
        on = self.gui.toggle_local(name)
        if name == "profiler":
            self.show_profiler = on

    def show_all_local_panels(self) -> None:
        self.gui.reset_locals(["chat", "inspector", "stats", "topbar", "profiler", "controls", "speed"], True)
        self.show_profiler = True

    def toggle_settings_panel(self) -> None:
        if not self.gui.global_visible:
            self.gui.set_local("settings", True)
            self.toggle_gui()
            return
        self.gui.toggle_local("settings")

    # ---------------------------------------------------------------- ablakméret / képarány
    def _poll_window_layout(self) -> None:
        try:
            aspect = float(window.aspect_ratio)
        except Exception:
            aspect = 16.0 / 9.0
        try:
            size = (int(window.size[0]), int(window.size[1]))
        except Exception:
            size = None
        key = (round(aspect, 4), size)
        if key == self._layout_key:
            return
        self._layout_key = key
        self._relayout(aspect)

    def _relayout(self, aspect: float) -> None:
        self.gui.layout(aspect)
        self._layout_menu(aspect)

    def _layout_menu(self, aspect: float) -> None:
        s = min(1.0, aspect / 1.7)
        self.menu_root.scale = (s, s, 1)
        self.menu_bg.scale = (aspect / s + 0.2, 1.0 / s + 0.2)

    def request_fullscreen_toggle(self) -> None:
        self._pending_fullscreen = (not self._fs_prev, 0.12)

    def _process_pending_fullscreen(self, dt: float) -> None:
        try:
            current = bool(window.fullscreen)
        except Exception:
            return
        if self._pending_fullscreen is not None:
            target, timer = self._pending_fullscreen
            timer -= dt
            if timer <= 0.0:
                self._pending_fullscreen = None
                if current != target:
                    try:
                        window.fullscreen = target
                        current = target
                    except Exception as exc:
                        self.status_banner = f"Teljes képernyő váltás sikertelen: {exc}"
            else:
                self._pending_fullscreen = (target, timer)
        self._fs_prev = current

    def _ensure_cursor(self) -> None:
        try:
            if mouse.locked:
                mouse.locked = False
            if not mouse.visible:
                mouse.visible = True
        except Exception:
            pass

    # ---------------------------------------------------------------- kamera-segédek
    def _snapshot_outer(self) -> OuterCameraState:
        return OuterCameraState(
            float(self.cam_pivot.x), float(self.cam_pivot.y), float(self.cam_pivot.z),
            float(self.cam_yaw), float(self.cam_pitch), float(self.cam_distance),
        )

    def _restore_outer(self, st: OuterCameraState) -> None:
        st = st.clamped(self.GRID_W, self.GRID_H)
        self.cam_pivot = Vec3(st.pivot_x, st.pivot_y, st.pivot_z)
        self.cam_yaw, self.cam_pitch, self.cam_distance = st.yaw, st.pitch, st.distance

    def _enter_fpv(self) -> None:
        self._saved_outer = self._snapshot_outer()
        try:
            self._outer_fov = float(camera.fov)
            self._outer_near = float(camera.clip_plane_near)
            camera.fov = camera_rig.FPV_FOV
            camera.clip_plane_near = camera_rig.FPV_NEAR_CLIP
        except Exception:
            pass
        sel = self._get_selected_agent()
        self._fpv_agent = sel
        if sel is not None:
            self.fpv_smooth_angle = float(sel.render_angle)

    def _exit_fpv(self, restore: bool = True) -> None:
        try:
            camera.fov = self._outer_fov
            camera.clip_plane_near = self._outer_near
        except Exception:
            pass
        if restore and self._saved_outer is not None:
            self._restore_outer(self._saved_outer)
        self._saved_outer = None
        self._fpv_agent = None

    def _set_camera_mode(self, mode: str, restore_outer: bool = True) -> None:
        prev = self.camera_mode
        if mode == "FPV" and prev != "FPV":
            self._enter_fpv()
        elif prev == "FPV" and mode != "FPV":
            self._exit_fpv(restore=restore_outer)
        if mode != prev:
            self._play("camera")
        self.camera_mode = mode
        labels = {
            "STRATEGIC": "Stratégiai nézet",
            "FOLLOW": "Követő nézet",
            "FPV": "Belső (FPV) nézet",
            "TOPDOWN": "Felülnézet",
        }
        self.btn_cam.text = labels.get(mode, "Stratégiai nézet")
        self._refresh_cam_settings_labels()

    def toggle_follow(self) -> None:
        self._set_camera_mode("STRATEGIC" if self.camera_mode == "FOLLOW" else "FOLLOW")

    def fit_whole_map(self) -> None:
        if self.camera_mode != "STRATEGIC":
            self._set_camera_mode("STRATEGIC")
        pitch = 65.0
        aspect = max(0.5, float(self.gui.aspect))
        wpu = 2.0 * math.tan(math.radians(self._camera_fov() / 2.0))
        d_w = self.GRID_W * 1.12 / (wpu * aspect)
        d_h = self.GRID_H * 1.12 * math.sin(math.radians(pitch)) / wpu
        self.cam_pivot = Vec3(self.GRID_W / 2.0, 0.0, self.GRID_H / 2.0)
        self.cam_yaw = 0.0
        self.cam_pitch = pitch
        self.cam_distance = float(np.clip(max(d_w, d_h), self.CAM_MIN_DIST, self.CAM_MAX_DIST))

    def adj_cam(self, key: str, delta: float) -> None:
        if key == "move":
            self.cam_speed = round(float(np.clip(self.cam_speed + delta, 0.25, 3.0)), 2)
        else:
            self.zoom_speed = round(float(np.clip(self.zoom_speed + delta, 0.25, 3.0)), 2)
        self._refresh_cam_settings_labels()

    def toggle_edge_scroll(self) -> None:
        self.edge_scroll = not self.edge_scroll
        self._refresh_cam_settings_labels()

    @staticmethod
    def _camera_fov() -> float:
        try:
            return float(camera.fov)
        except Exception:
            return 40.0

    def _clamp_pivot(self) -> None:
        self.cam_pivot = Vec3(
            float(np.clip(self.cam_pivot.x, 0.0, self.GRID_W)),
            float(self.cam_pivot.y),
            float(np.clip(self.cam_pivot.z, 0.0, self.GRID_H)),
        )

    def _on_robot_clicked(self, agent_id: int) -> None:
        try:
            if self.gui.mouse_over_gui(mouse.x, mouse.y):
                return
        except Exception:
            pass
        self.select_agent(agent_id)

    # ==========================================================================
    # FŐMENÜ (EGYSÉGESÍTVE A REFERENCIA KÉP DIZÁJNRENDSZERÉVEL)
    # ==========================================================================
    def _build_main_menu_ui(self) -> None:
        self.menu_root = Entity(parent=camera.ui, z=-0.10)
        self.menu_mode_btns: Dict[str, Any] = {}
        self.menu_map_btns: Dict[str, Any] = {}

        self.menu_bg = Entity(
            parent=self.menu_root, model="quad", color=UI.BG, scale=(3.4, 1.4), position=(0, 0, 0)
        )

        # Felső fejléckártya
        self._make_card(self.menu_root, 0.0, 0.41, 1.54, 0.13, bg=UI.PANEL, edge=UI.PANEL_EDGE, z=-0.001)
        Text(
            parent=self.menu_root,
            text="3D ROBOTPORSZÍVÓ AI · KOMMUNIKÁCIÓS ÉS EVOLÚCIÓS SZIMULÁTOR",
            position=(0, 0.44, -0.01),
            origin=(0, 0),
            scale=0.88,
            color=UI.TITLE,
        )
        self.menu_info_text = Text(
            parent=self.menu_root,
            text=" ",
            position=(0, 0.382, -0.01),
            origin=(0, 0),
            scale=0.58,
            color=UI.TEXT_DIM,
        )

        # 3 oszlopos kártyaelrendezés a referencia kép stílusában
        self._make_card(self.menu_root, -0.53, -0.06, 0.48, 0.74, bg=UI.PANEL, edge=UI.PANEL_EDGE, z=-0.001)
        self._make_card(self.menu_root, 0.0, -0.06, 0.52, 0.74, bg=UI.PANEL, edge=UI.PANEL_EDGE, z=-0.001)
        self._make_card(self.menu_root, 0.53, -0.06, 0.48, 0.74, bg=UI.PANEL, edge=UI.PANEL_EDGE, z=-0.001)

        for hx, hlabel in (
            (-0.53, "JÁTÉKMÓDOK"),
            (0.0, "PÁLYAVÁLASZTÓ"),
            (0.53, "BEÁLLÍTÁSOK ÉS INDÍTÁS"),
        ):
            Text(
                parent=self.menu_root,
                text=hlabel,
                position=(hx, 0.27, -0.01),
                origin=(0, 0),
                scale=0.60,
                color=UI.TEXT_DIM,
            )

        for idx, (m_key, m_val) in enumerate(MODE_CATALOG.items()):
            self.menu_mode_btns[m_key] = self._make_btn(
                self.menu_root,
                m_val["title"],
                (-0.53, 0.18 - idx * 0.115),
                (0.43, 0.082),
                lambda k=m_key: self.select_menu_mode(k),
                text_scale=0.58,
            )

        for idx, (map_key, map_val) in enumerate(MAP_CATALOG.items()):
            self.menu_map_btns[map_key] = self._make_btn(
                self.menu_root,
                f"{map_val['name']} ({map_val['difficulty']})",
                (0.0, 0.19 - idx * 0.092),
                (0.47, 0.068),
                lambda k=map_key: self.select_menu_map(k),
                text_scale=0.56,
            )

        self._make_btn(self.menu_root, "Populáció -2", (0.42, 0.19), (0.20, 0.052), lambda: self.adj_param("pop", -2), text_scale=0.56)
        self._make_btn(self.menu_root, "Populáció +2", (0.64, 0.19), (0.20, 0.052), lambda: self.adj_param("pop", 2), text_scale=0.56)
        self._make_btn(self.menu_root, "Kosz/Tick -1", (0.42, 0.12), (0.20, 0.052), lambda: self.adj_param("dirt", -1), text_scale=0.56)
        self._make_btn(self.menu_root, "Kosz/Tick +1", (0.64, 0.12), (0.20, 0.052), lambda: self.adj_param("dirt", 1), text_scale=0.56)
        self._make_btn(self.menu_root, "Mutáció -3%", (0.42, 0.05), (0.20, 0.052), lambda: self.adj_param("mut", -0.03), text_scale=0.56)
        self._make_btn(self.menu_root, "Mutáció +3%", (0.64, 0.05), (0.20, 0.052), lambda: self.adj_param("mut", 0.03), text_scale=0.56)
        self._make_btn(self.menu_root, "Pálya méret -", (0.42, -0.02), (0.20, 0.052), lambda: self.adj_param("scale", -1), text_scale=0.56)
        self._make_btn(self.menu_root, "Pálya méret +", (0.64, -0.02), (0.20, 0.052), lambda: self.adj_param("scale", 1), text_scale=0.56)
        self._make_btn(self.menu_root, "3D Árnyékolás (Lit/Unlit)", (0.53, -0.09), (0.42, 0.048), self.toggle_shader_mode, text_scale=0.56)
        self.btn_menu_sound = self._make_btn(self.menu_root, "Hangeffektek: BE", (0.53, -0.15), (0.42, 0.048), self.toggle_sound, text_scale=0.56)
        self._make_btn(
            self.menu_root,
            "Evolúciós Labor Grafikonok",
            (0.53, -0.21),
            (0.42, 0.048),
            self.open_matplotlib,
            col=UI.ACTIVE_BG,
            hover=UI.ACTIVE_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=UI.TITLE,
            text_scale=0.56,
        )
        self._make_btn(
            self.menu_root,
            "3D SZIMULÁCIÓ INDÍTÁSA",
            (0.53, -0.29),
            (0.42, 0.072),
            self.launch_from_menu,
            col=UI.PRIMARY,
            hover=UI.PRIMARY_HOVER,
            edge=UI.ACTIVE_EDGE,
            text_col=rgb(8, 20, 16),
            text_scale=0.64,
        )
        self._make_btn(
            self.menu_root,
            "KILÉPÉS",
            (0.53, -0.37),
            (0.42, 0.048),
            sys.exit,
            col=UI.DANGER_BG,
            hover=UI.DANGER_HOVER,
            edge=UI.DANGER_EDGE,
            text_col=UI.TEXT,
            text_scale=0.58,
        )

        self._refresh_menu_text()

    def _refresh_menu_text(self) -> None:
        m_info = MAP_CATALOG[self.selected_map]
        shd_str = "Lit 3D Fények" if self.use_lit_shader else "Gyors Unlit"
        self.menu_info_text.text = (
            f"{MODE_CATALOG[self.selected_mode]['short']}   ·   {m_info['name']}   ·   "
            f"{self.cfg_pop_size} robot   ·   Kosz: {self.cfg_dirt_rate}/tick   ·   "
            f"Mutáció: {int(self.cfg_mutation_rate*100)}%   ·   Pálya: {self.cfg_world_scale}x   ·   {shd_str}\n"
            f"{m_info['desc']}"
        )
        for k, b in getattr(self, "menu_mode_btns", {}).items():
            sel = k == self.selected_mode
            b.color = UI.ACTIVE_BG if sel else UI.BTN
            b.highlight_color = UI.ACTIVE_HOVER if sel else UI.BTN_HOVER
            if b.text_entity is not None:
                b.text_entity.color = UI.TITLE if sel else UI.TEXT
        for k, b in getattr(self, "menu_map_btns", {}).items():
            sel = k == self.selected_map
            b.color = UI.ACTIVE_BG if sel else UI.BTN
            b.highlight_color = UI.ACTIVE_HOVER if sel else UI.BTN_HOVER
            if b.text_entity is not None:
                b.text_entity.color = UI.TITLE if sel else UI.TEXT

    def select_menu_mode(self, mode_key: str) -> None:
        self.selected_mode = mode_key
        self.selected_map = MODE_CATALOG[mode_key]["default_map"]
        self._refresh_menu_text()

    def select_menu_map(self, map_key: str) -> None:
        self.selected_map = map_key
        self._refresh_menu_text()

    def adj_param(self, key: str, delta: float) -> None:
        if key == "pop":
            self.cfg_pop_size = int(np.clip(self.cfg_pop_size + delta, 6, 50))
        elif key == "dirt":
            self.cfg_dirt_rate = int(np.clip(self.cfg_dirt_rate + delta, 0, 8))
        elif key == "scale":
            self.cfg_world_scale = int(np.clip(self.cfg_world_scale + delta, 1, MAX_WORLD_SCALE))
        elif key == "mut":
            self.cfg_mutation_rate = round(float(np.clip(self.cfg_mutation_rate + delta, 0.03, 0.45)), 2)
        self._refresh_menu_text()

    def toggle_shader_mode(self) -> None:
        self.use_lit_shader = not self.use_lit_shader
        self._refresh_menu_text()

    def open_main_menu(self) -> None:
        self.in_menu = True
        self.paused = True
        self.menu_root.enabled = True
        self.hud_root.enabled = False
        self._ensure_cursor()
        self._refresh_menu_text()

    def launch_from_menu(self) -> None:
        self.menu_root.enabled = False
        self.hud_root.enabled = True
        self.in_menu = False
        self.start_simulation(keep_ea=False)

    def select_agent(self, agent_id: int) -> None:
        self.selected_agent_id = agent_id
        sel = self._get_selected_agent()
        if sel is not None:
            self.fpv_smooth_angle = sel.render_angle
        self.status_banner = f"Kijelölve a 3D térben: R-{agent_id+1:02d}"

    def on_chat_row_clicked(self, row_idx: int) -> None:
        if 0 <= row_idx < len(self.visible_chat_senders):
            sender_id = self.visible_chat_senders[row_idx]
            if sender_id >= 0:
                self.select_agent(sender_id)
                if self.camera_mode not in ("FOLLOW", "FPV"):
                    self._set_camera_mode("FOLLOW")

    def toggle_pause(self) -> None:
        if self.show_summary_modal:
            self.advance_generation()
        else:
            self.paused = not self.paused
            self.btn_pause.text = "Folytat" if self.paused else "Szünet"

    def step_once(self) -> None:
        self.paused = True
        self.btn_pause.text = "Folytat"
        if not self.show_summary_modal:
            self._execute_single_tick()
            self.tick_accumulator = 1.0

    def speed_down(self) -> None:
        self.speed_idx = max(0, self.speed_idx - 1)
        self.spd_text.text = f"{self.speed_multiplier}x"

    def speed_up(self) -> None:
        self.speed_idx = min(len(self.SPEED_LEVELS) - 1, self.speed_idx + 1)
        self.spd_text.text = f"{self.speed_multiplier}x"

    def cycle_camera_mode(self) -> None:
        modes = ["STRATEGIC", "FOLLOW", "FPV", "TOPDOWN"]
        self._set_camera_mode(modes[(modes.index(self.camera_mode) + 1) % len(modes)])

    def toggle_fpv_camera(self) -> None:
        self._set_camera_mode("STRATEGIC" if self.camera_mode == "FPV" else "FPV")

    def reset_camera(self) -> None:
        self._set_camera_mode("STRATEGIC", restore_outer=False)
        self.cam_pivot = Vec3(self.GRID_W / 2.0, 0.0, self.GRID_H / 2.0)
        self.cam_distance = 68.0
        self.cam_yaw = 0.0
        self.cam_pitch = 52.0
        self._saved_outer = None
        self._refresh_cam_settings_labels()

    # ---------------------------------------------------------------- hang
    def _play(self, name: str) -> None:
        snd = getattr(self, "sound", None)
        if snd is not None:
            snd.play(name)

    def _clicked(self, cb):
        def handler():
            self._play("click")
            cb()
        return handler

    def toggle_sound(self) -> None:
        self.sound.toggle()
        self._refresh_sound_labels()
        if self.sound.enabled:
            self._play("click")

    def cycle_sound_volume(self) -> None:
        self.sound.cycle_volume()
        self._refresh_sound_labels()
        self._play("click")

    def _refresh_sound_labels(self) -> None:
        on = self.sound.enabled
        vol_pct = f"{int(round(self.sound.volume * 100))}%"
        state = "BE" if on else ("KI" if self.sound.available else "KI")
        if hasattr(self, "btn_menu_sound"):
            self.btn_menu_sound.text = f"Hangeffektek: {state}"
            self.btn_menu_sound.color = UI.ACTIVE_BG if on else UI.BTN
        if hasattr(self, "btn_sound_set"):
            self.btn_sound_set.text = f"Hang: {'BE' if on else 'KI'}"
            self.btn_sound_set.color = UI.ACTIVE_BG if on else UI.BTN
            self.btn_volume_set.text = f"Hangerő: {vol_pct}"
        if hasattr(self, "btn_sound_card"):
            self.btn_sound_card.text = f"Hang: {'be' if on else 'ki'}"
            self.btn_sound_card.color = UI.ACTIVE_BG if on else UI.BTN
            self.btn_vol_card.text = vol_pct
        if hasattr(self, "side_sound_btn"):
            self.side_sound_btn.color = UI.PRIMARY if on else UI.DANGER_BG

    def _tick_sounds(self, sel_agent: Optional[RobotAgent]) -> None:
        if self.env.total_dirt_cleaned > self._snd_cleaned:
            self._play("collect")
        self._snd_cleaned = self.env.total_dirt_cleaned
        if self.env.combat_events:
            self._play("attack")
        if self.env.charge_events:
            self._play("charge")
        for msg in self.env.comm_events:
            if msg.intent in ("LOW_BAT_SOS", "THREAT_DETECTED"):
                self._play("help")
                break
        if sel_agent is not None:
            if any(m.sender_id == sel_agent.id or sel_agent.id in m.processed_by for m in self.env.comm_events):
                self._play("message")
            if sel_agent.alive and sel_agent.energy_ratio < 0.25 and sel_agent.id not in self._low_energy_notified:
                self._low_energy_notified.add(sel_agent.id)
                self._play("low_energy")
            elif sel_agent.energy_ratio > 0.4:
                self._low_energy_notified.discard(sel_agent.id)

    # ---------------------------------------------------------------- kommunikációs elemek
    def toggle_comm_display(self) -> None:
        show = not self.gui.is_local("chat")
        self.gui.set_local("chat", show)
        self.show_role_tags = show

    def toggle_radio(self) -> None:
        self.env.communication_enabled = not self.env.communication_enabled
        on = self.env.communication_enabled
        self.btn_radio.text = "Rádió" if on else "Rádió: KI"
        self.btn_radio.color = UI.ACTIVE_BG if on else UI.BTN
        if self.btn_radio.text_entity is not None:
            self.btn_radio.text_entity.color = UI.TITLE if on else UI.TEXT_DIM

    def toggle_3d_labels(self) -> None:
        self.show_all_3d_labels = not self.show_all_3d_labels

    def set_inspector_tab(self, tab: str) -> None:
        self.inspector_tab = tab
        is_dec = tab == "DECISION"
        self.insp_dec_root.enabled = is_dec
        self.insp_gen_root.enabled = not is_dec
        self.btn_tab_dec.color = UI.ACTIVE_BG if is_dec else UI.BTN
        self.btn_tab_gen.color = UI.ACTIVE_BG if not is_dec else UI.BTN
        if self.btn_tab_dec.text_entity is not None:
            self.btn_tab_dec.text_entity.color = UI.TITLE if is_dec else UI.TEXT
        if self.btn_tab_gen.text_entity is not None:
            self.btn_tab_gen.text_entity.color = UI.TITLE if not is_dec else UI.TEXT

    def cycle_chat_filter(self) -> None:
        cats = ["MIND", "FONTOS", "INFO/TASK", "ENERGY/HELP", "COMBAT", "ROBOT"]
        labels = {
            "MIND": "Mind",
            "FONTOS": "Fontos",
            "INFO/TASK": "Feladat",
            "ENERGY/HELP": "Energia",
            "COMBAT": "Harc",
            "ROBOT": "Robot",
        }
        self.chat_filter = cats[(cats.index(self.chat_filter) + 1) % len(cats)]
        self.chat_scroll_offset = 0
        if hasattr(self, "btn_chat_filter"):
            self.btn_chat_filter.text = labels.get(self.chat_filter, "Mind")

    def scroll_chat(self, delta: int) -> None:
        self.chat_scroll_offset = max(0, self.chat_scroll_offset + delta)

    def open_matplotlib(self) -> None:
        if self.ea is not None and len(self.ea.history["generation"]) > 0:
            show_matplotlib_stats(self.ea.history, self.selected_mode)
        else:
            self.status_banner = "Még nincs lezárt generáció a Matplotlib grafikonokhoz!"

    def save_stats_csv(self) -> None:
        self.status_banner = self.ea.export_history_to_csv()

    def save_chat_export(self) -> None:
        self.status_banner = self.env.comm_bus.export_chat_log(self.selected_mode, self.ea.generation)

    def start_simulation(self, keep_ea: bool = False) -> None:
        if not keep_ea or self.ea is None or self.ea.mode != self.selected_mode:
            self.ea = EvolutionaryAlgorithm(
                pop_size=self.cfg_pop_size,
                mutation_rate=self.cfg_mutation_rate,
                mode=self.selected_mode,
                num_teams=self.cfg_num_teams,
                start_energy=self.cfg_start_energy,
            )

        self.env = SimulationEnvironment(
            width=self.BASE_W,
            height=self.BASE_H,
            initial_dirt=130,
            dirt_spawn_rate=self.cfg_dirt_rate,
            map_key=self.selected_map,
            mode=self.selected_mode,
            seed=self.cfg_seed + self.ea.generation,
            world_scale=self.cfg_world_scale,
        )
        self._apply_world_dims()
        self._snd_cleaned = 0
        self._low_energy_notified.clear()
        self.env.reset(self.ea.agents)
        self._build_3d_world()
        if getattr(self, "_initialized", False):
            self._play("spawn")

        self.current_step = 0
        self.tick_accumulator = 0.0
        self.paused = False
        self.btn_pause.text = "Szünet"
        self.show_summary_modal = False
        self.summary_modal.enabled = False
        self.summary_text.enabled = False
        self.chat_scroll_offset = 0
        self.selected_agent_id = 0
        self.reset_camera()

    def _apply_world_dims(self) -> None:
        self.GRID_W = int(self.env.width)
        self.GRID_H = int(self.env.height)
        wpu = 2.0 * math.tan(math.radians(self._camera_fov() / 2.0))
        fit = max(self.GRID_W * 1.12 / wpu, self.GRID_H * 1.12 * math.sin(math.radians(65.0)) / wpu)
        self.CAM_MAX_DIST = float(max(115.0, fit * 1.15))
        self._dirt_signature = None

    def _check_round_termination(self) -> Tuple[bool, str]:
        living = [a for a in self.ea.agents if a.alive]
        if len(living) == 0:
            return True, "Minden robotporszívó lemerült vagy kiesett!"

        if self.current_step >= self.max_steps:
            return True, f"Elérve a forduló időkerete ({self.max_steps} tick)."

        if self.selected_mode == "ANTARCTIC":
            if all(a.reached_goal for a in living):
                return True, "Minden túlélő robot sikeresen elérte a B kutatóállomást!"

        elif self.selected_mode == "COOP":
            rem_dirt = int((self.env.dirt_grid > 0).sum())
            if rem_dirt == 0 and all(a.returned_to_base for a in living):
                return True, "A teljes 3D létesítmény tiszta, és minden élő robot visszatért a bázisra!"

        elif self.selected_mode == "TDM":
            alive_teams = set(a.team_id for a in living)
            if len(alive_teams) <= 1:
                winner = list(alive_teams)[0] if alive_teams else 0
                t_names = {1: "ALPHA (Kék)", 2: "BRAVO (Vörös)", 3: "GAMMA (Zöld)"}
                return True, f"A(z) {t_names.get(winner, 'Ismeretlen')} csapat győzött az Arénában!"

        elif self.selected_mode == "CLASSIC":
            if len(living) <= 1 and self.ea.pop_size > 2:
                return True, "Utolsó talpon maradt túlélő!"

        return False, ""

    def _execute_single_tick(self) -> None:
        self.env.step(self.ea.agents)
        self.current_step += 1
        self.perf.record_logical_tick(self.env.last_step_ai_decisions)

        self._sync_3d_dirt(spawn_dust=True)
        self._tick_sounds(self._get_selected_agent())

        for msg in self.env.comm_events:
            self._spawn_3d_comm_ring(msg.pos)
        for d_pos, r_pos in self.env.energy_share_events:
            self._spawn_3d_beam(d_pos, r_pos, rgb(16, 185, 129), 0.35)
            self.particles.spawn_energy_orbs(float(r_pos[0]), float(r_pos[1]), count=6)
        for a_pos, v_pos in self.env.combat_events:
            self._spawn_3d_beam(a_pos, v_pos, rgb(244, 63, 94), 0.28)
            self.particles.spawn_combat_sparks(float(v_pos[0]), float(v_pos[1]), count=8)

        ended, reason = self._check_round_termination()
        if ended:
            agents = self.ea.agents
            n = max(1, len(agents))
            alive = sum(1 for a in agents if a.alive)
            at_goal = sum(1 for a in agents if a.reached_goal)
            at_base = sum(1 for a in agents if a.alive and a.returned_to_base)

            self.last_summary_text = (
                f"=== 3D KÜLDETÉS ÉS GENERÁCIÓ LEZÁRVA (#{self.ea.generation}. Generáció) ===\n\n"
                f"Lezárás oka: {reason}\n"
                f"Eltelt szimulációs idő: {self.current_step} tick\n"
                f"Működőképes robotok: {alive} / {n} db ({alive/n*100:.0f}%)\n"
                f"Feltakarított szennyeződés: {self.env.total_dirt_cleaned} egység\n"
                f"B-állomásra / Bázisra beérkezett: {max(at_goal, at_base)} / {n} robot\n"
                f"Sikeres energiaátadások: {self.env.total_energy_transfers} db ({self.env.total_energy_shared_amount:.1f} E)\n"
                f"Rádiókommunikációs üzenetek: {len(self.env.comm_bus.chat_archive)} db\n\n"
                f"[Nyomd meg a SPACE vagy N billentyűt a következő evolúciós generáció indításához!]"
            )
            self.summary_text.text = self.last_summary_text
            self.show_summary_modal = True
            self._play("complete")
            self.summary_modal.enabled = True
            self.summary_text.enabled = True
            self.paused = True
            self.btn_pause.text = "Folytat"

    def advance_generation(self) -> None:
        new_pop = self.ea.evolve_next_generation(goal_pos=self.env.goal_pos)
        self.env.seed = self.cfg_seed + self.ea.generation
        self.env.rng = np.random.default_rng(self.env.seed)
        self.env.reset(new_pop)
        self._build_3d_world()
        self._play("spawn")
        self.current_step = 0
        self.tick_accumulator = 0.0
        self.show_summary_modal = False
        self.summary_modal.enabled = False
        self.summary_text.enabled = False
        self.paused = False
        self.btn_pause.text = "Szünet"

    def _get_selected_agent(self) -> Optional[RobotAgent]:
        if not self.ea or not self.ea.agents:
            return None
        for a in self.ea.agents:
            if a.id == self.selected_agent_id:
                return a
        return self.ea.agents[0]

    def _poll_camera_drag(self) -> Tuple[float, float, bool]:
        st = self._drag
        if st is None:
            return 0.0, 0.0, False
        key = "right mouse" if st["btn"] == "right" else "middle mouse"
        if not held_keys[key]:
            self._drag = None
            return 0.0, 0.0, False
        mx, my = float(mouse.x), float(mouse.y)
        dx, dy = mx - st["last_x"], my - st["last_y"]
        st["last_x"], st["last_y"] = mx, my
        if not st["active"]:
            if math.hypot(mx - st["x0"], my - st["y0"]) < st["threshold"]:
                return 0.0, 0.0, False
            st["active"] = True
            return mx - st["x0"], my - st["y0"], True
        return dx, dy, True

    def _begin_camera_drag(self, button: str) -> None:
        if self.gui.mouse_over_gui(mouse.x, mouse.y):
            self._drag = None
            return
        self._drag = {
            "btn": button,
            "x0": float(mouse.x),
            "y0": float(mouse.y),
            "last_x": float(mouse.x),
            "last_y": float(mouse.y),
            "active": False,
            "threshold": self.RMB_DRAG_THRESHOLD if button == "right" else 0.0,
        }

    def _end_camera_drag(self, button: str) -> None:
        if self._drag is not None and self._drag["btn"] == button:
            self._drag = None

    def _update_camera(self, dt: float) -> None:
        sel = self._get_selected_agent()
        drag_dx, drag_dy, dragging = self._poll_camera_drag()

        if self.camera_mode == "FPV":
            if sel is None or not sel.alive:
                self.status_banner = "A kiválasztott robot kiesett - visszatérés a külső nézetbe."
                self._set_camera_mode("STRATEGIC")
            else:
                if sel is not self._fpv_agent:
                    self._fpv_agent = sel
                    self.fpv_smooth_angle = float(sel.render_angle)
                self.fpv_smooth_angle = rate_limited_angle(self.fpv_smooth_angle, float(sel.render_angle), dt)
                pos, target = fpv_pose(float(sel.render_pos[0]), float(sel.render_pos[1]), self.fpv_smooth_angle)
                camera.position = Vec3(*pos)
                camera.look_at(Vec3(*target))
                return

        if self.camera_mode == "FOLLOW" and (
            held_keys["w"]
            or held_keys["a"]
            or held_keys["s"]
            or held_keys["d"]
            or held_keys["up arrow"]
            or held_keys["down arrow"]
            or held_keys["left arrow"]
            or held_keys["right arrow"]
            or dragging
        ):
            self._set_camera_mode("STRATEGIC")

        eff_pitch, eff_dist = self.cam_pitch, self.cam_distance
        if self.camera_mode == "TOPDOWN":
            eff_pitch = 84.0
        if self.camera_mode == "FOLLOW" and sel is not None:
            target_pivot = Vec3(float(sel.render_pos[0]), 0.5, float(sel.render_pos[1]))
            self.cam_pivot = self.cam_pivot + (target_pivot - self.cam_pivot) * min(1.0, dt * 8.0)
            eff_pitch, eff_dist = follow_profile(self.cam_pitch, self.cam_distance)
        else:
            rad_yaw = math.radians(self.cam_yaw)
            forward = Vec3(math.sin(rad_yaw), 0, math.cos(rad_yaw))
            right = Vec3(math.cos(rad_yaw), 0, -math.sin(rad_yaw))
            pan_speed = 38.0 * dt * self.cam_speed * max(0.5, self.cam_distance / 68.0)

            move = Vec3(0, 0, 0)
            if held_keys["w"] or held_keys["up arrow"]:
                move += forward
            if held_keys["s"] or held_keys["down arrow"]:
                move -= forward
            if held_keys["a"] or held_keys["left arrow"]:
                move -= right
            if held_keys["d"] or held_keys["right arrow"]:
                move += right

            if self.edge_scroll and self._drag is None:
                half_w = 0.5 * float(self.gui.aspect)
                mx, my = float(mouse.x), float(mouse.y)
                inside = abs(mx) <= half_w and abs(my) <= 0.5
                if inside and not self.gui.mouse_over_gui(mx, my):
                    edge = 0.012
                    if mx >= half_w - edge:
                        move += right
                    elif mx <= -half_w + edge:
                        move -= right
                    if my >= 0.5 - edge:
                        move += forward
                    elif my <= -0.5 + edge:
                        move -= forward

            if dragging:
                wpu = 2.0 * self.cam_distance * math.tan(math.radians(self._camera_fov() / 2.0))
                sin_p = max(0.3, math.sin(math.radians(eff_pitch)))
                self.cam_pivot = self.cam_pivot - right * (drag_dx * wpu) - forward * (drag_dy * wpu / sin_p)

            if move.length() > 0:
                self.cam_pivot = self.cam_pivot + move.normalized() * pan_speed
            if move.length() > 0 or dragging:
                self._clamp_pivot()

            if held_keys["q"]:
                self.cam_yaw -= 65.0 * dt
            if held_keys["e"]:
                self.cam_yaw += 65.0 * dt

        pitch_r = math.radians(eff_pitch)
        yaw_r = math.radians(self.cam_yaw)
        horiz = eff_dist * math.cos(pitch_r)
        vert = eff_dist * math.sin(pitch_r)

        cam_pos = Vec3(
            self.cam_pivot.x - horiz * math.sin(yaw_r),
            self.cam_pivot.y + vert,
            self.cam_pivot.z - horiz * math.cos(yaw_r),
        )
        camera.position = cam_pos
        camera.look_at(self.cam_pivot)

    def _update_3d_effects(self, dt: float) -> None:
        self.particles.update(dt)

        rem_beams = []
        for ent, t_pair in self.beam_entities:
            t_pair[0] += dt
            if t_pair[0] >= t_pair[1]:
                destroy(ent)
            else:
                rem_beams.append((ent, t_pair))
        self.beam_entities = rem_beams

        rem_rings = []
        for ent, t_pair in self.ring_entities:
            t_pair[0] += dt
            if t_pair[0] >= t_pair[1]:
                destroy(ent)
            else:
                prog = t_pair[0] / t_pair[1]
                sc = 1.2 + prog * 6.5
                ent.scale = (sc, 0.03, sc)
                rem_rings.append((ent, t_pair))
        self.ring_entities = rem_rings

    # ==========================================================================
    # SZÖVEGFORMÁZÓ SEGÉDEK A REFERENCIA KÉP STÍLUSÁHOZ
    # ==========================================================================
    @staticmethod
    def _short_alt_name(name: str) -> str:
        mapping = {
            "Társ által jelzett gócpont": "Jelzett gócpont",
            "Észlelt kosz takarítása": "Észlelt kosz",
            "Kosz feltakarítása": "Észlelt kosz",
            "Visszatérés a bázisra": "Vissza a bázisra",
            "Visszafordulás A-bázisra": "Vissza a bázisra",
            "Csapattárs töltése": "Társ töltése",
            "Csapattárs támogatása": "Társ támogatása",
            "Átvett célpont teljesítése": "Átvett célpont",
            "Töltőállomás felkeresése": "Töltődokk",
            "Töltődokk keresése": "Töltődokk",
            "Taktikai visszavonulás": "Visszavonulás",
            "Menekülés erősebb elől": "Menekülés",
            "Ellenféltámadása": "Támadás",
            "Kannibál támadás": "Kannibál támadás",
            "Takarítás / Energiagyűjtés": "Takarítás",
        }
        if name in mapping:
            return mapping[name]
        if name.startswith("Szektor #"):
            sec_num = name.split("#")[-1].split()[0]
            return f"{sec_num}. szektor"
        return name[:18]

    @staticmethod
    def _format_chat_summary(msg: AgentMessage) -> Tuple[str, str, Any]:
        rec = "csapat" if msg.recipient_label == "TEAM" else msg.recipient_label
        if msg.intent == "SECTOR_CLAIM":
            body = f"{msg.sender_name} -> {rec}: lefoglalja a #{msg.sector_id + 1}. szektort"
        elif msg.intent == "SECTOR_CLEAR":
            body = f"{msg.sender_name} -> {rec}: az #{msg.sector_id + 1}. szektor kész"
        elif msg.intent in ("TASK_ASSIGN", "TASK_OFFER"):
            body = f"{msg.sender_name} -> {rec}: gócpont átadás ({msg.target_key or f'#{msg.sector_id+1}'})"
        elif msg.intent == "TARGET_CLAIM":
            body = f"{msg.sender_name} -> {rec}: átvette a(z) {msg.target_key} célpontot"
        elif msg.intent == "TARGET_RELEASE":
            body = f"{msg.sender_name} -> {rec}: felszabadította ({msg.target_key})"
        elif msg.intent == "LOW_BAT_SOS":
            body = f"{msg.sender_name} -> {rec}: alacsony akku ({msg.value:.0f}%), segélykérés"
        elif msg.intent == "OFFER_ENERGY":
            body = f"{msg.sender_name} -> {rec}: elindult energiát átadni"
        elif msg.intent == "HELP_ACCEPTED":
            body = f"{msg.sender_name} -> {rec}: várja az energiaátadást"
        elif msg.intent == "THREAT_DETECTED":
            body = f"{msg.sender_name} -> {rec}: veszélyjelzés ({msg.target_key})"
        elif msg.intent == "DIRT_SPOT":
            body = f"{msg.sender_name} -> {rec}: {int(msg.value)} kosz-góc a #{msg.sector_id + 1}. szektorban"
        else:
            body = f"{msg.sender_name} -> {rec}: {msg.text[:44]}"

        if msg.intent == "SECTOR_CLAIM" or (msg.intent == "TARGET_CLAIM" and msg.status == "ACCEPTED"):
            return body, "Foglalt", UI.WARN
        if msg.status == "ACCEPTED":
            return body, "OK", UI.TITLE
        if msg.status == "REJECTED":
            return body, "Elutasítva", UI.DANGER
        if msg.status == "DELIVERED":
            return body, "OK", UI.TITLE
        if msg.status == "DROPPED":
            return body, "Elveszett", UI.DANGER
        return body, "Lejárt", UI.TEXT_MUTED

    @staticmethod
    def _format_comm_link_line(lnk: str) -> Tuple[str, Any]:
        clean = lnk
        if "): " in clean:
            sender_part = clean.split("(")[1].split(")")[0] if "(" in clean else ""
            rest = clean.split("): ", 1)[1]
            clean = f"{sender_part}: {rest}" if sender_part else rest
        elif ": " in clean and clean.startswith("MSG-"):
            clean = clean.split(": ", 1)[1]
        clean = clean.replace("Szektor #", "az #").replace("->", "->")
        col = UI.DANGER if ("ELUTASÍTVA" in lnk.upper() or "elutasítva" in lnk.lower()) else UI.TEXT_DIM
        return clean[:46], col

    def update(self) -> None:
        dt = float(utime.dt)
        self.perf.record_frame(dt)
        self.perf.update_rates()

        self._ensure_cursor()
        self._poll_window_layout()
        self._process_pending_fullscreen(dt)
        if self.hint_timer > 0.0:
            self.hint_timer -= dt
            if self.hint_timer <= 0.0:
                self.hint_text.enabled = False

        if self.in_menu:
            return

        if not self.paused and not self.show_summary_modal:
            self.tick_accumulator += dt * self.BASE_TICKS_PER_SEC * self.speed_multiplier
            executed = 0
            while self.tick_accumulator >= 1.0 and executed < 10:
                self.tick_accumulator -= 1.0
                self._execute_single_tick()
                executed += 1
                if self.show_summary_modal:
                    break
            if self.tick_accumulator > 2.0:
                self.tick_accumulator = 0.99

        alpha = 1.0 if self.paused else min(1.0, self.tick_accumulator)
        sel_agent = self._get_selected_agent()
        in_fpv = self.camera_mode == "FPV"

        for rv in self.robot_visuals:
            rv.agent.sync_render_state(alpha)
            is_sel = sel_agent is not None and rv.agent.id == sel_agent.id
            rv.update_visual(dt, is_sel, self.show_all_3d_labels, in_fpv=in_fpv, show_role=self.show_role_tags)

        if sel_agent is not None and sel_agent.alive and sel_agent.target_pos is not None:
            tx = float(np.clip(sel_agent.target_pos[0], 1.0, self.GRID_W - 1.0))
            tz = float(np.clip(sel_agent.target_pos[1], 1.0, self.GRID_H - 1.0))
            rx, rz = float(sel_agent.render_pos[0]), float(sel_agent.render_pos[1])

            self.target_marker_3d.enabled = True
            self.target_marker_3d.position = Vec3(tx, 0.05, tz)

            dist_t = max(0.1, math.hypot(tx - rx, tz - rz))
            ang_t = math.degrees(math.atan2(tz - rz, tx - rx))
            self.target_line_3d.enabled = True
            self.target_line_3d.position = Vec3((rx + tx) * 0.5, 0.08, (rz + tz) * 0.5)
            self.target_line_3d.scale = (dist_t, 0.04, 0.08)
            self.target_line_3d.rotation_y = -ang_t
        else:
            if self.target_marker_3d:
                self.target_marker_3d.enabled = False
            if self.target_line_3d:
                self.target_line_3d.enabled = False

        self._update_3d_effects(dt)
        self._update_camera(dt)
        if self.gui.global_visible:
            self._update_hud_texts(sel_agent)

    def _update_hud_texts(self, sel_agent: Optional[RobotAgent]) -> None:
        agents = self.ea.agents
        n = max(1, len(agents))
        living = [a for a in agents if a.alive]
        at_base = sum(1 for a in agents if a.alive and (a.returned_to_base or a.reached_goal))
        rem_dirt = int((self.env.dirt_grid > 0).sum())

        # 1. Bal felső sziget
        self.top_mode_text.text = MODE_CATALOG[self.selected_mode]["short"]
        self.top_map_text.text = MAP_CATALOG[self.selected_map]["name"]
        self.top_gen_text.text = f"Generáció #{self.ea.generation}"

        # 2. Középső felső sziget (Tick + progress bar)
        self.tick_progress_label.text = f"Tick {self.current_step} / {self.max_steps}"
        self.tick_progress_bar.set_progress(self.current_step / max(1, self.max_steps), UI.PRIMARY)

        # 3. Jobb felső sziget (FPS, entitás, ütköző)
        ent_count = len(self.map_entities) + len(self.dirt_entities) + len(self.robot_visuals) * 8 + len(self.particles.particles)
        col_count = len(self.map_entities) + len(self.robot_visuals) * 2
        self.profiler_text.text = f"{self.perf.avg_fps:.0f} FPS    entitás {ent_count}    ütköző {col_count}"

        # 4. FPV overlay
        self.fpv_overlay.enabled = self.camera_mode == "FPV" and sel_agent is not None
        if self.fpv_overlay.enabled and sel_agent is not None:
            hdg = int(math.degrees(sel_agent.render_angle)) % 360
            self.fpv_hud_text.text = (
                f"[FPV OPTIKAI SZENZOR: {sel_agent.name}]   IRÁNY: {hdg:03d} fok   "
                f"AKKU: {sel_agent.energy:.1f}%   ÁLLAPOT: {STATE_HU_LABELS.get(sel_agent.state, sel_agent.state)}\n"
                f"CÉLPONT: {sel_agent.current_target_desc}"
            )

        # 5. Bal oldali Statisztika kártya
        stat_vals = [
            f"{len(living)} / {n - len(living)}",
            f"{self.env.total_dirt_cleaned}",
            f"{rem_dirt}",
            f"{at_base}",
            f"{self.env.total_energy_transfers}",
            f"{len(self.env.comm_bus.chat_archive)}",
        ]
        for idx, val_str in enumerate(stat_vals):
            self.stat_value_texts[idx].text = val_str

        # 6. Bal alsó Szektorok kártya
        cleared_cnt = 0
        claimed_cnt = 0
        in_prog_cnt = 0
        claimed_owners: List[str] = []
        for s_idx in range(6):
            _, s_dirt = self.env._get_sector_info(s_idx)
            assigned_here = [a for a in living if (a.assigned_sector % 6) == s_idx]
            if s_dirt == 0:
                cleared_cnt += 1
                self.sector_bars[s_idx].set_progress(1.0, UI.PRIMARY)
            elif assigned_here:
                if len(assigned_here) == 1:
                    claimed_cnt += 1
                    claimed_owners.append(assigned_here[0].name)
                    ratio = max(0.25, min(0.90, 1.0 - s_dirt / 30.0))
                    self.sector_bars[s_idx].set_progress(ratio, UI.WARN)
                else:
                    in_prog_cnt += 1
                    ratio = max(0.25, min(0.90, 1.0 - s_dirt / 30.0))
                    self.sector_bars[s_idx].set_progress(ratio, UI.PRIMARY)
            else:
                in_prog_cnt += 1
                ratio = max(0.12, min(0.65, 1.0 - s_dirt / 35.0))
                self.sector_bars[s_idx].set_progress(ratio, rgb(56, 189, 248))

        owners_sfx = f" ({', '.join(claimed_owners[:2])})" if claimed_owners else ""
        self.sector_summary_text.text = (
            f"{cleared_cnt} kész, {in_prog_cnt} folyamatban, {claimed_cnt} lefoglalva{owners_sfx}"
        )

        # 7. Jobb oldali Robot Inspector
        if sel_agent is not None:
            fit_val = self.ea.compute_fitness(sel_agent, goal_pos=self.env.goal_pos)
            exp = sel_agent.explanation
            self.insp_name_text.text = sel_agent.name
            self.insp_state_text.text = STATE_HU_LABELS.get(sel_agent.state, sel_agent.state)
            self.insp_traits_text.text = sel_agent.role.replace(" / ", " · ")

            bat_pct = sel_agent.energy
            bat_col = UI.PRIMARY if bat_pct > 50.0 else (UI.WARN if bat_pct > 25.0 else UI.DANGER)
            self.insp_bat_val.text = f"{bat_pct:.0f}%"
            self.insp_bat_val.color = bat_col
            self.insp_bat_bar.set_progress(sel_agent.energy_ratio, bat_col)
            self.insp_fit_val.text = f"{fit_val:.0f}"

            if self.inspector_tab == "DECISION":
                chosen_clean = exp.chosen_action.split(" (")[0]
                if "jelzett gócpont" in chosen_clean.lower():
                    headline = "A társ által jelzett gócpont felé tart."
                elif "átvett célpont" in chosen_clean.lower():
                    headline = "Az átvett célpont teljesítésén dolgozik."
                elif "észlelt kosz" in chosen_clean.lower() or "kosz feltakarítása" in chosen_clean.lower():
                    headline = "A közeli szennyeződés feltakarítását végzi."
                elif "szektor" in chosen_clean.lower():
                    headline = f"A(z) #{sel_agent.assigned_sector + 1}. szektor megtisztításán dolgozik."
                elif "bázis" in chosen_clean.lower():
                    headline = "Visszatér a bázisra töltés és dokkolás céljából."
                else:
                    headline = f"{chosen_clean}."

                self.insp_dec_headline.text = headline[:46]
                sub_line1 = f"Cél: {exp.current_goal[:36]}. {exp.perceived_env.replace(' | ', ', ')[:38]}."
                sub_line2 = exp.measurable_reason[:48]
                self.insp_dec_sub.text = f"{sub_line1}\n{sub_line2}"

                alts = exp.considered_alternatives[:4]
                for i in range(4):
                    lbl_t, bar_u, val_t = self.alt_rows[i]
                    if i < len(alts):
                        aname, ascore = alts[i]
                        lbl_t.text = self._short_alt_name(aname)
                        bar_u.set_enabled(True)
                        bar_u.set_progress(ascore / 100.0, UI.PRIMARY if i == 0 else rgb(40, 145, 110))
                        val_t.text = f"{ascore:.0f}"
                    else:
                        lbl_t.text = "-"
                        bar_u.set_enabled(False)
                        val_t.text = ""

                links = sel_agent.recent_comm_links[-3:]
                for i in range(3):
                    if i < len(links):
                        txt, col = self._format_comm_link_line(links[i])
                        self.comm_chain_texts[i].text = txt
                        self.comm_chain_texts[i].color = col
                    else:
                        self.comm_chain_texts[i].text = "Nincs új rádióreakció" if i == 0 else ""
                        self.comm_chain_texts[i].color = UI.TEXT_MUTED
            else:
                g = sel_agent.genome.genes
                for i in range(8):
                    g_bar, g_val = self.gene_rows[i]
                    g_bar.set_progress(float(g[i]), UI.PRIMARY)
                    g_val.text = f"{g[i]*100:3.0f}%"
                log_str = "\n".join([f"· {ln[:44]}" for ln in sel_agent.decision_log[-3:]])
                self.insp_mem_text.text = (
                    f"Ismert kosz-gócok: {len(sel_agent.memory.dirt_spots)} db  ·  "
                    f"Tiszta szektorok: {sorted([s+1 for s in sel_agent.memory.cleared_sectors.keys()])}\n"
                    f"Utolsó döntések:\n{log_str}"
                )

        # 8. Alsó Chat Log / Eseménynapló kártya
        if self.bottom_log_mode == "EVENT":
            self.chat_header_text.text = "Rendszer- és eseménynapló"
            self.event_log_text.enabled = True
            for r_idx in range(6):
                self.chat_row_buttons[r_idx].enabled = False
                self.chat_row_LeftTexts[r_idx].enabled = False
                self.chat_row_MidTexts[r_idx].enabled = False
                self.chat_row_RightTexts[r_idx].enabled = False
            ev_lines = []
            for entry in reversed(self.env.event_log[-4:]):
                ev_lines.append(f"T:{entry.tick:03d}   [{entry.agent_name}] ({entry.category}): {entry.message[:68]}")
            self.event_log_text.text = "\n".join(ev_lines) if ev_lines else "Még nincs naplózott esemény."
        else:
            self.chat_header_text.text = (
                "Robot chat log" if self.chat_filter == "MIND" else f"Robot chat log [{self.chat_filter}]"
            )
            self.event_log_text.enabled = False

            filtered_chat: List[AgentMessage] = []
            for m in self.env.comm_bus.chat_archive:
                if self.chat_filter == "FONTOS" and m.priority < 2:
                    continue
                elif self.chat_filter == "INFO/TASK" and m.category not in ("[INFO]", "[TASK]"):
                    continue
                elif self.chat_filter == "ENERGY/HELP" and m.category not in ("[ENERGY]", "[HELP]"):
                    continue
                elif self.chat_filter == "COMBAT" and m.category not in ("[COMBAT]", "[WARNING]"):
                    continue
                elif self.chat_filter == "ROBOT" and sel_agent is not None:
                    if m.sender_id != sel_agent.id and m.recipient_id != sel_agent.id:
                        continue
                filtered_chat.append(m)

            total_m = len(filtered_chat)
            max_off = max(0, total_m - 4)
            self.chat_scroll_offset = min(self.chat_scroll_offset, max_off)
            end_i = total_m - self.chat_scroll_offset
            start_i = max(0, end_i - 4)
            vis_slice = filtered_chat[start_i:end_i]

            self.visible_chat_senders.clear()
            for r_idx in range(6):
                btn = self.chat_row_buttons[r_idx]
                t_l = self.chat_row_LeftTexts[r_idx]
                t_m = self.chat_row_MidTexts[r_idx]
                t_r = self.chat_row_RightTexts[r_idx]
                if r_idx >= 4:
                    btn.enabled = False
                    t_l.enabled = False
                    t_m.enabled = False
                    t_r.enabled = False
                    self.visible_chat_senders.append(-1)
                    continue

                btn.enabled = True
                t_l.enabled = True
                t_m.enabled = True
                t_r.enabled = True

                if r_idx < len(vis_slice):
                    msg = vis_slice[r_idx]
                    self.visible_chat_senders.append(msg.sender_id)
                    body_str, status_str, status_col = self._format_chat_summary(msg)
                    is_hl = (sel_agent is not None and msg.sender_id == sel_agent.id) or (
                        r_idx == len(vis_slice) - 1 and msg.status == "ACCEPTED"
                    )
                    btn.color = UI.ACTIVE_BG if is_hl else rgba(18, 25, 38, 0)
                    t_l.text = f"T:{msg.sent_tick:03d}"
                    t_m.text = body_str[:58]
                    t_m.color = UI.TEXT if is_hl else UI.TEXT_DIM
                    t_r.text = status_str
                    t_r.color = status_col
                else:
                    self.visible_chat_senders.append(-1)
                    btn.color = rgba(18, 25, 38, 0)
                    t_l.text = ""
                    t_m.text = "-" if r_idx == 0 else ""
                    t_r.text = ""

    def handle_input(self, key: str) -> None:
        if key == KEYS["fullscreen"]:
            self.request_fullscreen_toggle()
            return

        if key == "escape" or key == "m":
            if self.in_menu:
                self.launch_from_menu()
            else:
                self.open_main_menu()
            return

        if self.in_menu:
            return

        if key == KEYS["gui_toggle"]:
            self.toggle_gui()
            return
        if key == KEYS["settings"]:
            self.toggle_settings_panel()
            return
        if key == KEYS["comm"]:
            self.toggle_comm_display()
            return
        if key == KEYS["sound"]:
            self.toggle_sound()
            return

        if key == "right mouse down":
            self._begin_camera_drag("right")
            return
        if key == "right mouse up":
            self._end_camera_drag("right")
            return
        if key == "middle mouse down":
            self._begin_camera_drag("middle")
            return
        if key == "middle mouse up":
            self._end_camera_drag("middle")
            return

        if key == "space":
            self.toggle_pause()
        elif key == KEYS["step"]:
            self.step_once()
        elif key in ("+", "=", "numpad +"):
            self.speed_up()
        elif key in ("-", "_", "numpad -"):
            self.speed_down()
        elif key == "v":
            self.cycle_camera_mode()
        elif key == "f":
            self.toggle_fpv_camera()
        elif key == KEYS["camera_home"]:
            self.reset_camera()
        elif key == KEYS["fit_map"]:
            self.fit_whole_map()
        elif key == KEYS["follow"]:
            self.toggle_follow()
        elif key == "l":
            self.toggle_3d_labels()
        elif key == "f3":
            self.toggle_panel_local("profiler")
        elif key == "r":
            self.start_simulation(keep_ea=True)
        elif key == "n":
            self.advance_generation()
        elif key == "p":
            self.open_matplotlib()
        elif key == "c":
            self.save_stats_csv()
        elif key == "j":
            self.save_chat_export()
        elif key in ("scroll up", "scroll down"):
            up = key == "scroll up"
            over = self.gui.panel_at(mouse.x, mouse.y)
            if over == "chat":
                self.scroll_chat(2 if up else -2)
            elif over is None:
                step = 3.5 * self.zoom_speed
                delta = -step if up else step
                self.cam_distance = float(np.clip(self.cam_distance + delta, self.CAM_MIN_DIST, self.CAM_MAX_DIST))


controller: Optional[UrsinaSimulationController] = None


def update():
    if controller is not None:
        controller.update()


def input(key):
    if controller is not None:
        controller.handle_input(key)


if __name__ == "__main__":
    ensure_app_icon()
    app = Ursina(
        title="3D Robotporszívó AI & Evolúciós Szimulátor (Ursina Engine + FPV & Particles)",
        borderless=False,
        fullscreen=False,
        size=(1280, 720),
        vsync=True,
    )
    window.color = UI.BG
    try:
        _sr = window.screen_resolution
        window.position = Vec2(max(0, (_sr[0] - window.size[0]) / 2), max(0, (_sr[1] - window.size[1]) / 2 - 30))
    except Exception as _exc:
        print(f"[ablak] Az ablak középre igazítása nem sikerült: {_exc}")
    controller = UrsinaSimulationController()
    app.run()
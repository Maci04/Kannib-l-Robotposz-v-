"""
Kamera-segédmodul: a belső (FPV) és a külső (stratégiai) nézet tiszta, Ursina-független logikája.

A korábbi hibák okai és az itteni javításuk:
  * A külső kamera állapota (pivot, elforgatás, dőlés, távolság) FPV-re váltáskor nem mentődött,
    a követő (FOLLOW) mód pedig felülírta a dőlésszöget/távolságot -> az `OuterCameraState`
    pillanatképet készít belépéskor, és pontosan visszaállítja kilépéskor.
  * A belső nézet 40 fokos látószöggel és szinte vízszintes iránnyal, a robot irányának azonnali
    követésével működött -> `FPV_*` paraméterek (széles látószög, enyhe lefelé dőlés) és
    `rate_limited_angle` (sebességkorlátos, simított iránykövetés).
  * A nézetváltások több helyen, közvetlenül írták a módot -> a vezérlő egyetlen belépési pontot használ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# ---- belső (FPV) kamera beállításai
FPV_FOV: float = 78.0            # széles látószög: a robot előtti terület jól belátható
FPV_HEIGHT: float = 0.78         # a kamera magassága a talaj fölött (a lidar-kupola fölött)
FPV_FORWARD: float = 0.10        # eltolás a robot közepétől előre (a karosszériába nem lóg bele)
FPV_PITCH_DEG: float = 16.0      # enyhe lefelé dőlés: a padló és a szemét is látszik
FPV_NEAR_CLIP: float = 0.05
FPV_MAX_TURN_RATE: float = 2.6   # rad/s: a kamera legfeljebb ekkora szögsebességgel fordul (~150°/s)
FPV_SMOOTH: float = 7.0          # simítási tényező (1/s)

# ---- külső kamera tartomány
OUTER_FOV_DEFAULT: float = 40.0
OUTER_NEAR_CLIP: float = 0.1
CAM_MIN_DIST: float = 14.0
CAM_MAX_DIST: float = 115.0
CAM_MIN_PITCH: float = 20.0
CAM_MAX_PITCH: float = 85.0


def wrap_angle(a: float) -> float:
    """Szög normalizálása a [-pi, pi) tartományba."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rate_limited_angle(current: float, target: float, dt: float,
                       max_rate: float = FPV_MAX_TURN_RATE, smooth: float = FPV_SMOOTH) -> float:
    """
    A `current` szöget a legrövidebb íven a `target` felé mozgatja: exponenciális simítással,
    de legfeljebb `max_rate` rad/s szögsebességgel. Így a robot hirtelen irányváltása
    (pl. falnak ütközés utáni ~150°-os fordulat) nem "csavarja el" a kamerát.
    """
    diff = wrap_angle(target - current)
    step = diff * min(1.0, max(0.0, dt) * smooth)
    limit = max_rate * max(0.0, dt)
    if abs(step) > limit:
        step = math.copysign(limit, step)
    return current + step


def fpv_pose(x: float, z: float, heading: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    A belső kamera helyzete és nézési célpontja a robot (x, z) világkoordinátája és `heading` szöge alapján.
    A robot modellje a (cos a, sin a) irányba néz az (x, z) síkon (lásd RobotVisual3D.update_visual),
    ezért a kamera ugyanebbe az irányba néz. Visszaad: (kamera-pozíció, nézési célpont).
    """
    dx, dz = math.cos(heading), math.sin(heading)
    pos = (x + dx * FPV_FORWARD, FPV_HEIGHT, z + dz * FPV_FORWARD)
    look_dist = 10.0
    drop = math.tan(math.radians(FPV_PITCH_DEG)) * look_dist
    target = (x + dx * look_dist, FPV_HEIGHT - drop, z + dz * look_dist)
    return pos, target


@dataclass
class OuterCameraState:
    """A külső (stratégiai) kamera menthető állapota."""
    pivot_x: float
    pivot_y: float
    pivot_z: float
    yaw: float
    pitch: float
    distance: float

    def clamped(self, grid_w: float, grid_h: float) -> "OuterCameraState":
        return OuterCameraState(
            min(max(self.pivot_x, 0.0), grid_w), self.pivot_y, min(max(self.pivot_z, 0.0), grid_h),
            self.yaw % 360.0,
            min(max(self.pitch, CAM_MIN_PITCH), CAM_MAX_PITCH),
            min(max(self.distance, CAM_MIN_DIST), CAM_MAX_DIST),
        )


def follow_profile(pitch: float, distance: float) -> Tuple[float, float]:
    """
    A követő (FOLLOW) mód *hatékony* dőlésszöge és távolsága. A felhasználó által beállított
    külső állapotot nem írja felül, így a követésből visszalépve a korábbi nézet marad.
    """
    return 40.0, max(16.0, min(40.0, distance))


def outer_camera_position(pivot: Tuple[float, float, float], yaw_deg: float, pitch_deg: float,
                          distance: float) -> Tuple[float, float, float]:
    """A külső kamera pozíciója a pivot körül (a pivotra néz)."""
    pr, yr = math.radians(pitch_deg), math.radians(yaw_deg)
    horiz = distance * math.cos(pr)
    vert = distance * math.sin(pr)
    return (pivot[0] - horiz * math.sin(yr), pivot[1] + vert, pivot[2] - horiz * math.cos(yr))

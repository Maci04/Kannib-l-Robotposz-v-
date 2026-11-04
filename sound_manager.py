"""
Helyi hangeffektek: saját generálású WAV-fájlok + korlátozott, kikapcsolható lejátszó.

  * A hangok szintetizálva készülnek (standard `wave` modul, nincs külső függőség / hálózat), az
    `assets/sounds/` mappába, és csak akkor jönnek létre, ha még nem léteznek.
  * `SoundManager.play(name)`: BE/KI kapcsolóhoz kötött, eseményenkénti várakozási idővel (cooldown)
    és közös polifónia-korláttal, így gyakori eseményekből sem lesz hangáradat.
  * Hiányzó / sérült fájl vagy hiányzó hangrendszer esetén csendben kihagyja a hangot (a program indul).
  * A BE/KI állapot és a hangerő a `settings.json`-ban megmarad az újraindítás után is.
"""
from __future__ import annotations

import builtins
import json
import math
import os
import random
import struct
import time
import wave
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

SAMPLE_RATE = 22050

# ------------------------------------------------------------------ hangok szintézise
def _env(i: int, n: int, attack: float = 0.01, release: float = 0.4) -> float:
    t = i / max(1, n)
    a = min(1.0, t / max(1e-6, attack))
    r = min(1.0, (1.0 - t) / max(1e-6, release))
    return min(a, r)


def _tone(freq_start: float, freq_end: float, dur: float, vol: float = 0.5, noise: float = 0.0,
          square: float = 0.0, rng: Optional[random.Random] = None) -> List[float]:
    n = int(SAMPLE_RATE * dur)
    rng = rng or random.Random(7)
    out, phase = [], 0.0
    for i in range(n):
        f = freq_start + (freq_end - freq_start) * (i / max(1, n))
        phase += 2.0 * math.pi * f / SAMPLE_RATE
        s = math.sin(phase)
        if square:
            s = (1.0 - square) * s + square * (1.0 if s >= 0 else -1.0)
        if noise:
            s = (1.0 - noise) * s + noise * (rng.random() * 2.0 - 1.0)
        out.append(s * vol * _env(i, n))
    return out


def _silence(dur: float) -> List[float]:
    return [0.0] * int(SAMPLE_RATE * dur)


SOUND_RECIPES: Dict[str, Callable[[], List[float]]] = {
    "click": lambda: _tone(1100, 700, 0.045, 0.45, noise=0.15),
    "spawn": lambda: _tone(260, 780, 0.28, 0.40),
    "collect": lambda: _tone(660, 660, 0.06, 0.40) + _tone(990, 990, 0.09, 0.40),
    "charge": lambda: _tone(420, 600, 0.22, 0.32),
    "low_energy": lambda: _tone(330, 330, 0.12, 0.45, square=0.5) + _silence(0.07) + _tone(300, 300, 0.14, 0.45, square=0.5),
    "attack": lambda: _tone(180, 70, 0.20, 0.50, noise=0.55, square=0.4),
    "message": lambda: _tone(1250, 1250, 0.07, 0.30),
    "help": lambda: _tone(700, 700, 0.09, 0.40) + _silence(0.04) + _tone(880, 880, 0.09, 0.40) + _silence(0.04) + _tone(1100, 1100, 0.13, 0.40),
    "complete": lambda: _tone(523, 523, 0.13, 0.42) + _tone(659, 659, 0.13, 0.42) + _tone(784, 784, 0.13, 0.42) + _tone(1047, 1047, 0.30, 0.42),
    "camera": lambda: _tone(500, 380, 0.09, 0.22, noise=0.45),
}

# eseményenkénti minimális várakozás (másodperc) - ennél gyakrabban nem szólal meg
COOLDOWNS: Dict[str, float] = {
    "click": 0.05, "spawn": 1.0, "collect": 0.18, "charge": 0.7, "low_energy": 4.0, "attack": 0.28,
    "message": 0.6, "help": 1.6, "complete": 2.0, "camera": 0.25,
}


def write_wav(path: str, samples: List[float]) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32000)) for s in samples))


def ensure_sound_files(folder: str) -> Dict[str, str]:
    """Létrehozza a hiányzó hangfájlokat; visszaadja a név -> útvonal leképezést (csak a létező fájlokra)."""
    paths: Dict[str, str] = {}
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return paths
    for name, recipe in SOUND_RECIPES.items():
        p = os.path.join(folder, f"{name}.wav")
        try:
            if not os.path.exists(p) or os.path.getsize(p) < 100:
                write_wav(p, recipe())
            paths[name] = p
        except Exception:
            continue          # egy hang hiánya nem akadályozhat semmit
    return paths


# ------------------------------------------------------------------ lejátszó háttér
class _PandaSound:
    def __init__(self, handle):
        self.h = handle

    def play(self, volume: float) -> None:
        self.h.setVolume(volume)
        self.h.play()


def default_loader(path: str):
    """Panda3D hangbetöltő (az Ursina alatt van). Hiba / hiányzó hangrendszer esetén None."""
    base = getattr(builtins, "base", None)
    if base is None:
        return None
    try:
        from panda3d.core import Filename
        handle = base.loader.loadSfx(Filename.from_os_specific(path))
        return _PandaSound(handle) if handle is not None else None
    except Exception:
        return None


class SoundManager:
    MAX_PLAYS_PER_WINDOW = 5          # polifónia-korlát: ennyi hang indulhat el ...
    WINDOW = 0.25                     # ... ekkora időablakon belül

    def __init__(self, sounds_dir: str, settings_path: Optional[str] = None,
                 loader: Callable[[str], object] = default_loader, clock: Callable[[], float] = time.monotonic):
        self.sounds_dir = sounds_dir
        self.settings_path = settings_path
        self.clock = clock
        self.enabled: bool = True
        self.volume: float = 0.6
        self.handles: Dict[str, object] = {}
        self._last_play: Dict[str, float] = {}
        self._recent: Deque[float] = deque()
        self.play_count: Dict[str, int] = {}
        self.load_settings()
        for name, path in ensure_sound_files(sounds_dir).items():
            try:
                h = loader(path)
            except Exception:
                h = None
            if h is not None:
                self.handles[name] = h

    @property
    def available(self) -> bool:
        return len(self.handles) > 0

    def play(self, name: str) -> bool:
        """Lejátssza a hangot, ha be van kapcsolva, létezik és a várakozási/polifónia-korlát engedi."""
        if not self.enabled:
            return False
        h = self.handles.get(name)
        if h is None:
            return False
        now = self.clock()
        if now - self._last_play.get(name, -1e9) < COOLDOWNS.get(name, 0.2):
            return False
        while self._recent and now - self._recent[0] > self.WINDOW:
            self._recent.popleft()
        if len(self._recent) >= self.MAX_PLAYS_PER_WINDOW:
            return False
        try:
            h.play(self.volume)
        except Exception:
            return False
        self._last_play[name] = now
        self._recent.append(now)
        self.play_count[name] = self.play_count.get(name, 0) + 1
        return True

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)
        if not self.enabled:
            for h in self.handles.values():
                try:
                    stop = getattr(getattr(h, "h", h), "stop", None)
                    if stop:
                        stop()
                except Exception:
                    pass
        self.save_settings()

    def toggle(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, float(v)))
        self.save_settings()

    def cycle_volume(self) -> float:
        steps = [0.2, 0.4, 0.6, 0.8, 1.0]
        nxt = next((s for s in steps if s > self.volume + 1e-6), steps[0])
        self.set_volume(nxt)
        return self.volume

    # ---- tartós beállítások
    def load_settings(self) -> None:
        if not self.settings_path or not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enabled = bool(data.get("sound_enabled", True))
            self.volume = max(0.0, min(1.0, float(data.get("sound_volume", 0.6))))
        except Exception:
            pass

    def save_settings(self) -> None:
        if not self.settings_path:
            return
        try:
            data: Dict[str, object] = {}
            if os.path.exists(self.settings_path):
                try:
                    with open(self.settings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            data["sound_enabled"] = self.enabled
            data["sound_volume"] = self.volume
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

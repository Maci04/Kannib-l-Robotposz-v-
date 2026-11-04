"""
Központi GUI-kezelő a 3D robotporszívó-szimulációhoz.

Feladatai:
  * RESPONSIVE ELRENDEZÉS: minden panel egy "horgony" (sarok / él) köré van rendezve.
    A panelek tartalma az eredeti, 16:9-es referencia-koordinátákkal készült; a kezelő
    a panel gyökér-entitását úgy pozicionálja és skálázza, hogy a referencia-sarok
    mindig a tényleges ablak megfelelő sarkára essen. Így a gyerekelemek koordinátáit
    nem kell átírni, és az ablakméret/képarány változásakor csak a néhány gyökér-entitás
    pozíciója frissül (nincs GUI-újraépítés).
  * LÁTHATÓSÁG három, egymástól független szinten:
        1. globális GUI-láthatóság (Tab)           -> gui.global_visible
        2. panelenkénti helyi láthatóság (beállítás) -> panel.local_visible
        3. csoportos összecsukás (oldalsó fül)      -> gui.group_collapsed[csoport]
    Egy panel akkor látszik, ha mindhárom megengedi. A három állapot külön tárolódik,
    ezért a globális visszakapcsolás pontosan a korábbi egyéni állapotot állítja vissza.

A modul nem importál Ursinát: a panelek gyökere bármilyen, `enabled`, `x`, `y`, `scale`
attribútumokkal rendelkező objektum lehet (Ursina Entity, vagy tesztben egy egyszerű
helyettesítő objektum).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

REF_ASPECT: float = 16.0 / 9.0          # az eredeti HUD ezen a képarányon készült
REF_HALF_W: float = REF_ASPECT / 2.0    # a referencia-képernyő fél szélessége (ui-egység)
REF_HALF_H: float = 0.5                 # az Ursina camera.ui függőleges fél kiterjedése
MIN_SCALE: float = 0.5                  # ennél jobban nem zsugorítjuk a paneleket

# horgony -> (vízszintes, függőleges) irány: -1 = bal/alsó, 0 = közép, +1 = jobb/felső
ANCHORS: Dict[str, Tuple[int, int]] = {
    "TL": (-1, 1), "TC": (0, 1), "TR": (1, 1),
    "CL": (-1, 0), "C": (0, 0), "CR": (1, 0),
    "BL": (-1, -1), "BC": (0, -1), "BR": (1, -1),
}

Rect = Tuple[float, float, float, float]  # (középpont x, középpont y, szélesség, magasság) referencia-koordinátában


class GuiPanel:
    """Egy önállóan ki/be kapcsolható, horgonyzott panel."""

    def __init__(
        self,
        name: str,
        root,
        anchor: str,
        title: str = "",
        group: str = "",
        hideable: bool = True,
        default_visible: bool = True,
        rects: Iterable[Rect] = (),
        depends_any: Iterable[str] = (),
        on_layout: Optional[Callable[["GuiPanel", float, float], None]] = None,
    ):
        if anchor not in ANCHORS:
            raise ValueError(f"Ismeretlen horgony: {anchor}")
        self.name = name
        self.root = root
        self.anchor = anchor
        self.title = title or name
        self.group = group
        self.hideable = hideable            # False: a globális elrejtés nem érinti (pl. fordulóösszegző)
        self.local_visible = default_visible
        self.rects: List[Rect] = list(rects)
        self.depends_any: Tuple[str, ...] = tuple(depends_any)
        self.on_layout = on_layout
        self.effective: Optional[bool] = None   # utoljára az entitásra alkalmazott állapot
        self.px: float = 0.0                    # a gyökér jelenlegi pozíciója
        self.py: float = 0.0
        self.scale: float = 1.0


class GuiManager:
    def __init__(self, on_change: Optional[Callable[[], None]] = None):
        self.panels: Dict[str, GuiPanel] = {}
        self._order: List[str] = []
        self.global_visible: bool = True
        self.group_collapsed: Dict[str, bool] = {}
        self.aspect: float = REF_ASPECT
        self.scale: float = 1.0
        self.on_change = on_change

    # ------------------------------------------------------------------ regisztráció
    def add_panel(self, name: str, root, anchor: str, **kwargs) -> GuiPanel:
        if name in self.panels:
            raise ValueError(f"Már létező panel: {name}")
        panel = GuiPanel(name, root, anchor, **kwargs)
        self.panels[name] = panel
        self._order.append(name)
        if panel.group and panel.group not in self.group_collapsed:
            self.group_collapsed[panel.group] = False
        return panel

    # ------------------------------------------------------------------ elrendezés
    @staticmethod
    def scale_for_aspect(aspect: float) -> float:
        """Keskenyebb (nem 16:9-es) ablakban egyenletesen zsugorítja a paneleket."""
        if aspect <= 0:
            return 1.0
        return max(MIN_SCALE, min(1.0, aspect / REF_ASPECT))

    def layout(self, aspect: float) -> None:
        """Újraszámolja minden panel gyökerének pozícióját/méretét. Csak méretváltozáskor hívandó."""
        if aspect <= 0:
            return
        self.aspect = float(aspect)
        s = self.scale_for_aspect(self.aspect)
        self.scale = s
        for name in self._order:
            p = self.panels[name]
            kx, ky = ANCHORS[p.anchor]
            ref_x, ref_y = kx * REF_HALF_W, ky * REF_HALF_H          # a sarok a referenciában
            scr_x, scr_y = kx * self.aspect / 2.0, ky * REF_HALF_H   # ugyanaz a sarok a tényleges ablakban
            p.scale = s
            p.px = scr_x - s * ref_x
            p.py = scr_y - s * ref_y
            p.root.x = p.px
            p.root.y = p.py
            p.root.scale = (s, s, 1)
            if p.on_layout is not None:
                p.on_layout(p, self.aspect, s)

    def to_screen(self, panel: GuiPanel, x: float, y: float) -> Tuple[float, float]:
        """Referencia-koordinátát a panel aktuális tényleges képernyő-koordinátájára alakít."""
        return panel.px + panel.scale * x, panel.py + panel.scale * y

    # ------------------------------------------------------------------ láthatóság
    def _compute(self, p: GuiPanel, cache: Dict[str, bool]) -> bool:
        if p.name in cache:
            return cache[p.name]
        cache[p.name] = False  # ciklusvédelem
        if p.hideable and not self.global_visible:
            vis = False
        elif p.group and self.group_collapsed.get(p.group, False):
            vis = False
        elif p.depends_any:
            vis = any(self._compute(self.panels[d], cache) for d in p.depends_any if d in self.panels)
        else:
            vis = p.local_visible
        cache[p.name] = vis
        return vis

    def refresh(self) -> None:
        """A meglévő entitások enabled állapotát illeszti a három láthatósági szinthez (nincs újraépítés)."""
        cache: Dict[str, bool] = {}
        for name in self._order:
            p = self.panels[name]
            vis = self._compute(p, cache)
            if vis != p.effective:
                p.effective = vis
                p.root.enabled = vis
        if self.on_change is not None:
            self.on_change()

    def is_visible(self, name: str) -> bool:
        p = self.panels[name]
        if p.effective is None:
            self.refresh()
        return bool(p.effective)

    # globális
    def set_global(self, visible: bool) -> None:
        self.global_visible = bool(visible)
        self.refresh()

    def toggle_global(self) -> bool:
        self.set_global(not self.global_visible)
        return self.global_visible

    # helyi
    def set_local(self, name: str, visible: bool) -> None:
        self.panels[name].local_visible = bool(visible)
        self.refresh()

    def toggle_local(self, name: str) -> bool:
        self.set_local(name, not self.panels[name].local_visible)
        return self.panels[name].local_visible

    def is_local(self, name: str) -> bool:
        return self.panels[name].local_visible

    def reset_locals(self, names: Optional[Iterable[str]] = None, value: bool = True) -> None:
        for n in (names if names is not None else self._order):
            p = self.panels[n]
            if not p.depends_any:
                p.local_visible = value
        self.refresh()

    # csoportos összecsukás
    def set_group_collapsed(self, group: str, collapsed: bool) -> None:
        self.group_collapsed[group] = bool(collapsed)
        self.refresh()

    def toggle_group(self, group: str) -> bool:
        self.set_group_collapsed(group, not self.group_collapsed.get(group, False))
        return self.group_collapsed[group]

    def is_group_collapsed(self, group: str) -> bool:
        return self.group_collapsed.get(group, False)

    # ------------------------------------------------------------------ egér / találat
    def panel_at(self, mx: float, my: float) -> Optional[str]:
        """Annak a látható panelnek a neve, amelynek hátterén az (mx, my) ui-pozíció van."""
        for name in reversed(self._order):
            p = self.panels[name]
            if not self.is_visible(name):
                continue
            for (cx, cy, w, h) in p.rects:
                sx, sy = self.to_screen(p, cx, cy)
                sw, sh = w * p.scale, h * p.scale
                if abs(mx - sx) <= sw / 2.0 and abs(my - sy) <= sh / 2.0:
                    return name
        return None

    def mouse_over_gui(self, mx: float, my: float) -> bool:
        return self.panel_at(mx, my) is not None

"""
Csapat-koordináció és szerepek a helyben futó Multi-Agent rendszerhez (nincs LLM / hálózat).

Tartalma:
  * ROLES / ROLE_PROFILES  - öt stabil szerep és a döntéseket ténylegesen módosító súlyaik.
  * assign_roles()         - kezdeti szerepkiosztás a genomból (+ minimális csapatösszetétel).
  * rebalance_roles()      - ritkán (60 tickenként) futó, indokolt szerepcsere, ha egy fontos szerep
                             (felderítő / támogató) kiesett; csak zárolási idő lejárta után.
  * dialogue_line()        - eseményalapú szövegsablonok a kommunikációs naplóhoz / későbbi beszédbuborékokhoz.

A szerep NEM véletlen: a genom értékeiből számolódik, `ROLE_MIN_HOLD` tickig zárolt, és csak a
`rebalance_roles` változtathat rajta, amikor ténylegesen hiányzik egy szerep a csapatból.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

ROLES = ("SCOUT", "CLEANER", "SUPPORT", "HUNTER", "GENERAL")
ROLE_LABEL: Dict[str, str] = {
    "SCOUT": "Felderítő", "CLEANER": "Takarító", "SUPPORT": "Támogató", "HUNTER": "Vadász", "GENERAL": "Általános",
}
ROLE_SHORT: Dict[str, str] = {"SCOUT": "FEL", "CLEANER": "TAK", "SUPPORT": "TÁM", "HUNTER": "VAD", "GENERAL": "ÁLT"}

ROLE_MIN_HOLD: int = 150        # ennyi tickig nem változhat a szerep a kiosztás/csere után
REBALANCE_INTERVAL: int = 60    # ennyi tickenként vizsgálja a koordinátor a csapatösszetételt
COMBAT_MODES = ("CLASSIC", "TDM")


@dataclass(frozen=True)
class RoleProfile:
    """A szerep hatása a robot döntéseire (mind a decide_and_move()-ban használt tényleges szorzó/bónusz)."""
    clean_bonus: float        # a takarítás hasznosságához adott pont
    help_bonus: float         # a segítségnyújtás hasznosságához adott pont
    hunt_mult: float          # a vadászat (kannibál támadás) hasznosságának szorzója
    flee_mult: float          # a menekülés hasznosságának szorzója
    comm_cooldown_mult: float  # <1: gyakrabban kommunikál
    explore: bool             # frontier-alapú felderítést végez, ha nincs célpont
    share_threshold: int      # ennyi látott szemétgóc fölött osztja meg a célpontot
    task_accept_bias: float   # a kapott feladat elfogadási hajlandósága (+: szívesebben vállal)
    help_energy_floor: float  # ennyi energia-tartalék (pont) kell a segítségnyújtáshoz


ROLE_PROFILES: Dict[str, RoleProfile] = {
    "SCOUT":   RoleProfile(-6.0, 0.0, 0.80, 1.10, 0.60, True, 3, -0.35, 24.0),
    "CLEANER": RoleProfile(+9.0, 0.0, 0.85, 1.00, 1.00, False, 12, +0.30, 24.0),
    "SUPPORT": RoleProfile(0.0, +12.0, 0.70, 0.95, 0.85, False, 12, 0.0, 16.0),
    "HUNTER":  RoleProfile(-4.0, -6.0, 1.30, 0.80, 1.10, False, 14, -0.15, 30.0),
    "GENERAL": RoleProfile(0.0, 0.0, 1.00, 1.00, 1.00, False, 10, 0.0, 22.0),
}


def role_scores(genome, mode: str) -> Dict[str, float]:
    """Genom-alapú alkalmassági pontszámok (0..~1.2)."""
    hunter = genome.aggressiveness * 1.15 if mode in COMBAT_MODES else 0.0
    return {
        "SCOUT": genome.exploration * 1.00 + (1.0 - genome.return_caution / 2.25) * 0.25,
        "CLEANER": (genome.dirt_affinity - 0.35) / 1.65 * 1.00,
        "SUPPORT": genome.cooperation * 1.10,
        "HUNTER": hunter,
    }


def pick_role(genome, mode: str, current: Optional[str] = None, margin: float = 0.08) -> str:
    """A legalkalmasabb szerep; ha nincs egyértelmű (margin) győztes, Általános. Hiszterézis a jelenlegi javára."""
    scores = role_scores(genome, mode)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_s < 0.45 or best_s - second_s < margin:
        best = "GENERAL"
    if current and current != "GENERAL" and current in scores and scores[current] >= best_s - 0.12:
        return current          # kis különbségnél maradunk a jelenlegi szerepnél (nincs ide-oda ugrálás)
    return best


ROLE_QUOTA: Dict[str, float] = {"SCOUT": 0.20, "SUPPORT": 0.20, "HUNTER": 0.15, "CLEANER": 0.35}


def assign_roles(agents: Iterable, mode: str, tick: int = 0) -> None:
    """
    Kezdeti szerepkiosztás. Minden robot genom-alapú alkalmassági pontszámot kap; a szerepeket a
    populáción belüli RELATÍV alkalmasság szerint, kvótákkal osztjuk ki (kb. 20% felderítő, 20% támogató,
    15% vadász [csak harci módban], 35% takarító, a maradék általános), így mindig kiegyensúlyozott a csapat.
    A szerep ettől függetlenül a robot saját génjeiből következik (a legalkalmasabbak kapják).
    """
    agents = list(agents)
    n = len(agents)
    if n == 0:
        return
    for a in agents:
        a.set_role("GENERAL", tick, hold=ROLE_MIN_HOLD)
    if n < 4:
        for a in agents:
            a.set_role(pick_role(a.genome, mode), tick, hold=ROLE_MIN_HOLD)
        return
    scores = {a.id: role_scores(a.genome, mode) for a in agents}
    order = ["SCOUT", "SUPPORT"] + (["HUNTER"] if mode in COMBAT_MODES else []) + ["CLEANER"]
    free = list(agents)
    for role in order:
        quota = max(1, int(round(ROLE_QUOTA[role] * n)))
        mean = sum(scores[a.id][role] for a in agents) / n
        ranked = sorted(free, key=lambda a: scores[a.id][role] - mean, reverse=True)
        for a in ranked[:quota]:
            a.set_role(role, tick, hold=ROLE_MIN_HOLD)
            free.remove(a)


def _team_groups(agents: List, mode: str) -> List[List]:
    if mode == "TDM":
        teams: Dict[int, List] = {}
        for a in agents:
            teams.setdefault(a.team_id, []).append(a)
        return list(teams.values())
    return [agents]


def _ensure_role(agents: List, role: str, mode: str, tick: int) -> None:
    for group in _team_groups(agents, mode):
        if len(group) < 4:
            continue
        if any(a.role_id == role and a.alive for a in group):
            continue
        candidates = [a for a in group if a.alive and a.role_id in ("GENERAL", "CLEANER", "HUNTER", "SCOUT", "SUPPORT")]
        if not candidates:
            continue
        # a legjobb jelölt, de nem vesszük el egy másik hiányzó szerep egyetlen birtokosától
        def keep_penalty(a) -> float:
            same = sum(1 for b in group if b.role_id == a.role_id and b.alive)
            return 0.0 if same > 1 or a.role_id == "GENERAL" else 1.0
        best = max(candidates, key=lambda a: role_scores(a.genome, mode).get(role, 0.0) - keep_penalty(a))
        best.set_role(role, tick, hold=ROLE_MIN_HOLD)


def rebalance_roles(agents: List, mode: str, tick: int) -> List[str]:
    """
    Ritka (REBALANCE_INTERVAL) csapatösszetétel-ellenőrzés: ha egy fontos szerep birtokosa kiesett,
    a legalkalmasabb, zárolásától mentes robot átveszi. Visszaadja a változások leírását.
    """
    if tick % REBALANCE_INTERVAL != 0:
        return []
    changes: List[str] = []
    living = [a for a in agents if a.alive]
    if len(living) < 4:
        return changes
    for group in _team_groups(living, mode):
        for role in (("SCOUT", "SUPPORT", "HUNTER") if mode in COMBAT_MODES else ("SCOUT", "SUPPORT")):
            if any(a.role_id == role for a in group) or len(group) < 4:
                continue
            free = [a for a in group if tick >= a.role_lock_until and a.role_id in ("GENERAL", "CLEANER")]
            if not free:
                continue
            cand = max(free, key=lambda a: role_scores(a.genome, mode).get(role, 0.0))
            if role_scores(cand.genome, mode).get(role, 0.0) < 0.25:
                continue
            old = cand.role_id
            cand.set_role(role, tick, hold=ROLE_MIN_HOLD)
            changes.append(f"{cand.name}: {ROLE_LABEL[old]} -> {ROLE_LABEL[role]} (a csapatból hiányzott)")
    return changes


# ---------------------------------------------------------------------- szövegsablonok
def dialogue_line(msg, role_of: Optional[Dict[int, str]] = None) -> str:
    """
    Egy üzenet rövid, eseményalapú megjelenítési sora: "R-07 (FEL) -> R-12: szöveg".
    A szöveg az üzenet strukturált mezőiből (intent, status, reaction_note) áll össze, ezért csak
    valóban megtörtént eseményt jelenít meg.
    """
    role = ROLE_SHORT.get(getattr(msg, "sender_role", "") or (role_of or {}).get(msg.sender_id, ""), "")
    who = f"{msg.sender_name}({role})" if role else msg.sender_name
    note = f" [{msg.reaction_note}]" if getattr(msg, "reaction_note", "") else ""
    return f"{who}->{msg.recipient_label}: {msg.text}{note}"

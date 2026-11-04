import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional

from coordination import ROLE_LABEL, ROLE_PROFILES, ROLE_SHORT, RoleProfile


def lerp_angle(a: float, b: float, t: float) -> float:
    """Legrövidebb ív mentén történő sima szöginterpoláció (radiánban)."""
    diff = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + diff * t


@dataclass
class AgentMessage:
    """
    Strukturált kommunikációs protokoll üzenet-objektum robotok közötti adatcseréhez.
    Kategóriák: [INFO], [TASK], [ENERGY], [HELP], [WARNING], [TEAM], [COMBAT]
    Státuszok : QUEUED, DELIVERED, ACCEPTED, REJECTED, EXPIRED, DROPPED
    """
    msg_id: str
    sender_id: int
    sender_name: str
    team_id: int
    recipient_id: Optional[int]
    recipient_label: str
    category: str
    intent: str
    priority: int
    sent_tick: int
    deliver_tick: int
    ttl: int
    pos: np.ndarray
    sector_id: int
    value: float
    text: str
    ref_msg_id: Optional[str] = None
    status: str = "QUEUED"
    reaction_note: str = ""
    # --- Multi-Agent bővítés (mind alapértelmezett értékű, a régi kód változatlanul működik)
    sender_role: str = ""                 # a küldő szerepe a küldéskor (pl. "SCOUT")
    thread_id: str = ""                   # összetartozó üzenetek (kérés -> válasz) közös azonosítója
    target_key: str = ""                  # a célpont / foglalás kulcsa (TASK_*, TARGET_*)
    delivered_tick: int = -1              # mikor ért el legalább egy címzettet
    processed_tick: int = -1              # mikor dolgozta fel az első címzett
    processed_by: List[int] = field(default_factory=list)   # akik ténylegesen feldolgozták
    origin: Optional[np.ndarray] = None   # a küldő pozíciója küldéskor (a hatótáv ehhez képest számolódik)


@dataclass
class MemorySpot:
    """Az ágens memóriájában tárolt térbeli megfigyelés."""
    pos: np.ndarray
    value: float
    tick_seen: int
    ttl: int = 90


class AgentMemory:
    """
    Korlátozott kapacitású, idővel elévülő (TTL) lokális ágens-memória.
    Minden robot csak a saját szenzorai és a kapott rádióüzenetek alapján építi fel világképét.
    """

    def __init__(self, max_spots: int = 12):
        self.max_spots: int = max_spots
        self.dirt_spots: List[MemorySpot] = []
        self.hazard_spots: List[MemorySpot] = []
        self.cleared_sectors: Dict[int, int] = {}
        self.sector_claims: Dict[int, Tuple[int, int]] = {}
        self.partner_battery: Dict[int, Tuple[float, np.ndarray, int]] = {}
        self.unreachable_targets: List[Tuple[np.ndarray, int]] = []
        self.last_positions: List[np.ndarray] = []
        self.stuck_counter: int = 0
        # --- Multi-Agent bővítés
        self.peers: Dict[int, Dict[str, Any]] = {}               # ismert társak (üzenetből / szenzorból)
        self.target_claims: Dict[str, Tuple[int, int, np.ndarray]] = {}   # célpont-kulcs -> (foglaló, tick, pozíció)
        self.rejected_by: Dict[int, int] = {}                    # ki utasította el a legutóbbi felkérésemet (tick)
        self.threats: Dict[int, Tuple[np.ndarray, int, float]] = {}       # veszélyes robot id -> (poz, tick, súly)
        self.explored: Dict[Tuple[int, int], int] = {}           # durva rács: cella -> utolsó látogatás (tick)
        self.sent_offers: Dict[str, int] = {}                    # kiküldött feladat-felkérések: kulcs -> tick
        self.last_dirt_seen: int = 0                             # utolsó tick, amikor szemetet látott / jelzést kapott

    def cleanup(self, current_tick: int) -> None:
        self.dirt_spots = [s for s in self.dirt_spots if current_tick - s.tick_seen <= s.ttl]
        self.hazard_spots = [s for s in self.hazard_spots if current_tick - s.tick_seen <= s.ttl]
        self.cleared_sectors = {
            sec: t for sec, t in self.cleared_sectors.items() if current_tick - t <= 160
        }
        self.sector_claims = {
            sec: (oid, t) for sec, (oid, t) in self.sector_claims.items() if current_tick - t <= 110
        }
        self.partner_battery = {
            pid: (e, p, t) for pid, (e, p, t) in self.partner_battery.items() if current_tick - t <= 80
        }
        self.unreachable_targets = [
            (p, t) for (p, t) in self.unreachable_targets if current_tick - t <= 70
        ]
        self.peers = {i: p for i, p in self.peers.items() if current_tick - p["tick"] <= 140}
        self.target_claims = {k: v for k, v in self.target_claims.items() if current_tick - v[1] <= 120}
        self.rejected_by = {i: t for i, t in self.rejected_by.items() if current_tick - t <= 60}
        self.threats = {i: v for i, v in self.threats.items() if current_tick - v[1] <= 60}
        self.sent_offers = {k: t for k, t in self.sent_offers.items() if current_tick - t <= 50}

    def remember_dirt(self, pos: np.ndarray, amount: float, tick: int) -> None:
        for spot in self.dirt_spots:
            if float(np.linalg.norm(spot.pos - pos)) < 4.5:
                spot.pos = pos.copy()
                spot.value = amount
                spot.tick_seen = tick
                return
        self.dirt_spots.append(MemorySpot(pos=pos.copy(), value=amount, tick_seen=tick, ttl=85))
        if len(self.dirt_spots) > self.max_spots:
            self.dirt_spots.pop(0)

    # ------------------------------------------------------------ Multi-Agent segédek
    CELL: int = 16   # a felderítési rács cellamérete

    @staticmethod
    def target_key(pos: np.ndarray) -> str:
        return f"T{int(pos[0] // 5)}:{int(pos[1] // 5)}"

    def remember_peer(self, pid: int, name: str, pos: np.ndarray, energy: float, tick: int, role: str = "") -> None:
        prev = self.peers.get(pid, {})
        self.peers[pid] = {
            "name": name, "pos": pos.copy(), "energy": float(energy), "tick": tick,
            "role": role or prev.get("role", "GENERAL"),
        }

    def claim_target(self, key: str, owner: int, pos: np.ndarray, tick: int) -> None:
        self.target_claims[key] = (owner, tick, pos.copy())

    def release_target(self, key: str) -> None:
        self.target_claims.pop(key, None)

    def is_claimed_by_other(self, pos: np.ndarray, me: int, radius: float = 6.0) -> bool:
        """Más robot által lefoglalt célpont közelében van-e a pozíció (a saját foglalás nem számít)."""
        for owner, _t, cpos in self.target_claims.values():
            if owner != me and float(np.linalg.norm(cpos - pos)) < radius:
                return True
        return False

    def best_unclaimed_dirt(self, my_pos: np.ndarray, me: int, max_dist: float = 160.0) -> Optional[MemorySpot]:
        """A legközelebbi, más által NEM lefoglalt, emlékezetben tárolt szemétgóc."""
        best, best_d = None, max_dist
        for spot in self.dirt_spots:
            if self.is_claimed_by_other(spot.pos, me):
                continue
            if any(float(np.linalg.norm(spot.pos - u)) < 4.0 for u, _t in self.unreachable_targets):
                continue
            d = float(np.linalg.norm(spot.pos - my_pos))
            if d < best_d:
                best, best_d = spot, d
        return best

    def mark_visit(self, pos: np.ndarray, tick: int) -> None:
        self.explored[(int(pos[0] // self.CELL), int(pos[1] // self.CELL))] = tick

    def pick_frontier(self, my_pos: np.ndarray, grid_w: int, grid_h: int, obstacle_grid: np.ndarray,
                      tick: int) -> Optional[np.ndarray]:
        """
        Felderítési célpont: a legközelebbi, még meg nem látogatott (vagy a legrégebben látogatott) szabad
        rács-cella középpontja. Ritkán (érkezéskor / lejáratkor) hívjuk, nem minden tickben.
        """
        gw, gh = (grid_w + self.CELL - 1) // self.CELL, (grid_h + self.CELL - 1) // self.CELL
        best, best_score = None, 1e18
        for cx in range(gw):
            for cy in range(gh):
                px, py = min(grid_w - 2.0, cx * self.CELL + self.CELL / 2.0), min(grid_h - 2.0, cy * self.CELL + self.CELL / 2.0)
                if obstacle_grid[int(px), int(py)] != 0:
                    continue
                last = self.explored.get((cx, cy))
                d = math.hypot(px - my_pos[0], py - my_pos[1])
                if d < 6.0:
                    continue
                age_pen = 0.0 if last is None else 400.0 + (tick - last) * -0.5    # látogatottak hátrébb sorolva
                score = d + age_pen
                if score < best_score:
                    best, best_score = np.array([px, py], dtype=np.float32), score
        return best

    def remember_hazard(self, pos: np.ndarray, tick: int) -> None:
        for spot in self.hazard_spots:
            if float(np.linalg.norm(spot.pos - pos)) < 5.0:
                spot.tick_seen = tick
                return
        self.hazard_spots.append(MemorySpot(pos=pos.copy(), value=1.0, tick_seen=tick, ttl=140))
        if len(self.hazard_spots) > self.max_spots:
            self.hazard_spots.pop(0)


@dataclass
class DecisionExplanation:
    """Valós idejű döntésmagyarázó struktúra a 3D-s GUI Inspector számára."""
    current_goal: str = "Inicializálás"
    perceived_env: str = "Szenzorok indítása"
    last_message_effect: str = "Nincs feldolgozott üzenet"
    considered_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    chosen_action: str = "Várakozás"
    measurable_reason: str = "Kezdeti állapot"


@dataclass
class Genome:
    """
    A robotporszívó 8-génes kromoszómája ([0.0, 1.0] normalizált NumPy vektor):
      [0] aggressiveness : Támadási hajlam és versengés
      [1] risk_tolerance : Kockázatvállalási energiaküszöb
      [2] speed_gene     : Mozgási sebesség skálázó tényezője
      [3] dirt_affinity  : Szennyeződés-fókusz súlya
      [4] wall_avoidance : Akadály- és veszélyzóna-elkerülés
      [5] cooperation    : Segélynyújtási, energiaátadási és csapatmunkára való hajlam
      [6] return_caution : Hazatérési / biztonsági energiatartalék szorzó
      [7] comm_tendency  : Kommunikációs aktivitás és feladatátvételi rugalmasság
    """
    genes: np.ndarray

    def __post_init__(self) -> None:
        if len(self.genes) < 8:
            padded = np.full(8, 0.5, dtype=np.float32)
            padded[: len(self.genes)] = self.genes
            self.genes = padded
        else:
            self.genes = np.clip(self.genes[:8].astype(np.float32), 0.0, 1.0)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "Genome":
        if rng is not None:
            return cls(genes=rng.uniform(0.1, 0.9, size=8).astype(np.float32))
        return cls(genes=np.random.uniform(0.1, 0.9, size=8).astype(np.float32))

    @property
    def aggressiveness(self) -> float:
        return float(self.genes[0])

    @property
    def risk_tolerance(self) -> float:
        return float(0.15 + 0.70 * self.genes[1])

    @property
    def speed(self) -> float:
        return float(0.75 + 1.60 * self.genes[2])

    @property
    def dirt_affinity(self) -> float:
        return float(0.35 + 1.65 * self.genes[3])

    @property
    def wall_avoidance(self) -> float:
        return float(0.60 + 2.00 * self.genes[4])

    @property
    def cooperation(self) -> float:
        return float(self.genes[5])

    @property
    def return_caution(self) -> float:
        return float(1.10 + 1.15 * self.genes[6])

    @property
    def exploration(self) -> float:
        return float(self.genes[7])

    @property
    def comm_tendency(self) -> float:
        return float(0.25 + 0.75 * self.genes[7])

    @staticmethod
    def crossover(parent1: "Genome", parent2: "Genome") -> "Genome":
        n = len(parent1.genes)
        mask = np.random.rand(n) < 0.5
        child_genes = np.where(mask, parent1.genes, parent2.genes).copy()
        if np.random.rand() < 0.35:
            idx = np.random.randint(0, n)
            alpha = np.random.uniform(0.3, 0.7)
            child_genes[idx] = alpha * parent1.genes[idx] + (1.0 - alpha) * parent2.genes[idx]
        return Genome(genes=np.clip(child_genes, 0.0, 1.0))

    def mutate_shift(self, mutation_rate: float = 0.15, sigma: float = 0.12) -> None:
        n = len(self.genes)
        mutation_mask = np.random.rand(n) < mutation_rate
        noise = np.random.normal(0.0, sigma, size=n).astype(np.float32)
        self.genes = np.clip(self.genes + mutation_mask * noise, 0.0, 1.0)

    def mutate_swap(self, swap_rate: float = 0.05) -> None:
        if np.random.rand() < swap_rate:
            idx1, idx2 = np.random.choice(len(self.genes), size=2, replace=False)
            self.genes[idx1], self.genes[idx2] = self.genes[idx2], self.genes[idx1]


class RobotAgent:
    """
    Autonóm AI ágens saját memóriával, 7-lépéses döntési ciklussal,
    strukturált kommunikációval és folyamatos 3D-s (sub-tick) vizuális interpolációval.
    """

    MAX_ENERGY: float = 100.0
    SENSOR_RANGE: float = 26.0

    def __init__(
        self,
        agent_id: int,
        pos: np.ndarray,
        genome: Optional[Genome] = None,
        team_id: int = 0,
        name: Optional[str] = None,
        max_energy: float = 100.0,
    ):
        self.id: int = agent_id
        self.team_id: int = team_id
        self.name: str = name if name else f"R-{agent_id + 1:02d}"
        self.pos: np.ndarray = pos.astype(np.float32)
        self.prev_pos: np.ndarray = pos.astype(np.float32).copy()
        self.render_pos: np.ndarray = pos.astype(np.float32).copy()

        init_angle = float(np.random.uniform(-math.pi, math.pi))
        self.angle: float = init_angle
        self.prev_angle: float = init_angle
        self.render_angle: float = init_angle
        self.vel: np.ndarray = np.array([math.cos(init_angle), math.sin(init_angle)], dtype=np.float32)

        self.genome: Genome = genome if genome is not None else Genome.random()
        self.memory: AgentMemory = AgentMemory()
        self.explanation: DecisionExplanation = DecisionExplanation()

        self.MAX_ENERGY = float(max_energy)
        self.energy: float = self.MAX_ENERGY
        self.alive: bool = True
        self.steps_alive: int = 0
        self.dirt_cleaned: int = 0
        self.kills: int = 0
        self.damage_dealt: float = 0.0

        self.distance_traveled: float = 0.0
        self.energy_used: float = 0.0
        self.energy_given: float = 0.0
        self.energy_received: float = 0.0
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.assists: int = 0
        self.ai_decisions_count: int = 0

        self.reached_goal: bool = False
        self.returned_to_base: bool = False
        self.mission_aborted: bool = False
        self.death_reason: str = "Aktív"
        self.state: str = "CLEANING"
        self.role: str = self._compute_personality_label()
        # --- Multi-Agent állapot (a szerep és a közös munka állapotai)
        self.personality: str = self.role           # a korábbi, genom-alapú jellemző-címke
        self.role_id: str = "GENERAL"
        self.role_since: int = 0
        self.role_lock_until: int = 0
        self.reset_multiagent_state()

        self.inbox: List[AgentMessage] = []
        self.recent_comm_links: List[str] = []
        self.target_pos: Optional[np.ndarray] = None
        self.current_target_desc: str = "Lokális szektor pásztázása"
        self.next_planned_action: str = "Szenzoradatok elemzése"
        self.assigned_sector: int = agent_id % 6
        self.ally_target_id: Optional[int] = None
        self.ally_target_pos: Optional[np.ndarray] = None
        self.last_comm_tick: int = -30
        self.last_sos_tick: int = -80

        self.trail: List[Tuple[float, float]] = []
        self.decision_log: List[str] = ["[T:0] Autonóm AI ágens inicializálva."]
        self.known_partners: set = set()

    def reset_multiagent_state(self) -> None:
        self.commitment: Optional[Dict[str, Any]] = None      # elfogadott megosztott célpont
        self.awaiting_help: Optional[Dict[str, Any]] = None   # segítséget kértem, és várom
        self.ally_target_tick: int = -999
        self.last_attacker_id: Optional[int] = None
        self.last_attacked_tick: int = -999
        self.last_threat_msg_tick: int = -999
        self.frontier: Optional[np.ndarray] = None
        self.frontier_tick: int = -999
        self.last_share_tick: int = -999
        self._pending_status: Optional[str] = None

    @property
    def profile(self) -> RoleProfile:
        return ROLE_PROFILES.get(self.role_id, ROLE_PROFILES["GENERAL"])

    def set_role(self, role_id: str, tick: int, hold: int = 150) -> None:
        """Szerep beállítása (zárolással). A szerep a döntéseket a ROLE_PROFILES súlyain keresztül módosítja."""
        old = self.role_id
        self.role_id = role_id if role_id in ROLE_PROFILES else "GENERAL"
        self.role_since = tick
        self.role_lock_until = tick + hold
        self.role = f"{ROLE_LABEL[self.role_id]} / {self.personality}"
        if tick > 0 and old != self.role_id:
            self._pending_status = f"Szerepem mostantól: {ROLE_LABEL[self.role_id]} (eddig: {ROLE_LABEL[old]})."
            self.log_decision(tick, f"Szerepváltás: {ROLE_LABEL[old]} -> {ROLE_LABEL[self.role_id]}")

    def forget_peer(self, pid: int, tick: int) -> None:
        """Kiesett / elérhetetlen társ eltávolítása minden hivatkozásból (nincs érvénytelen címzett)."""
        self.memory.peers.pop(pid, None)
        self.memory.threats.pop(pid, None)
        self.memory.partner_battery.pop(pid, None)
        for key in [k for k, v in self.memory.target_claims.items() if v[0] == pid]:
            self.memory.target_claims.pop(key, None)       # a kiesett robot foglalása felszabadul
        if self.ally_target_id == pid:
            self.ally_target_id, self.ally_target_pos = None, None
        if self.awaiting_help is not None and self.awaiting_help.get("helper") == pid:
            self.awaiting_help["helper"] = None
        if self.commitment is not None and self.commitment.get("source") == pid:
            pass    # a feladat attól még érvényes marad: a célpont létezik, a kérő már nem fontos

    def _compute_personality_label(self) -> str:
        g = self.genome
        traits = []
        if g.cooperation >= 0.62:
            traits.append("Kooperatív")
        elif g.aggressiveness >= 0.66:
            traits.append("Ragadozó")
        else:
            traits.append("Önálló")

        if g.return_caution >= 1.75:
            traits.append("Óvatos")
        elif g.risk_tolerance >= 0.60:
            traits.append("Kockázatvállaló")
        elif g.exploration >= 0.65:
            traits.append("Felderítő")
        else:
            traits.append("Takarító")

        return " / ".join(traits)

    @property
    def energy_ratio(self) -> float:
        return max(0.0, min(1.0, self.energy / max(1.0, self.MAX_ENERGY)))

    def sync_render_state(self, alpha: float) -> None:
        """Sub-tick pozíció és szögelfordulás interpolációja a 3D-s megjelenítő számára."""
        a = max(0.0, min(1.0, alpha))
        self.render_pos = self.prev_pos + (self.pos - self.prev_pos) * a
        self.render_angle = lerp_angle(self.prev_angle, self.angle, a)

    def snap_render_state(self) -> None:
        self.prev_pos = self.pos.copy()
        self.render_pos = self.pos.copy()
        self.prev_angle = self.angle
        self.render_angle = self.angle

    def log_decision(self, tick: int, text: str) -> None:
        entry = f"[T:{tick}] {text}"
        if not self.decision_log or self.decision_log[-1].split("] ")[-1] != text:
            self.decision_log.append(entry)
            if len(self.decision_log) > 10:
                self.decision_log.pop(0)

    def record_comm_link(self, note: str) -> None:
        self.recent_comm_links.append(note)
        if len(self.recent_comm_links) > 6:
            self.recent_comm_links.pop(0)

    def receive_message(self, msg: AgentMessage) -> None:
        if not self.alive or msg.sender_id == self.id:
            return
        self.inbox.append(msg)
        self.messages_received += 1
        self.known_partners.add(msg.sender_name)

    def estimate_energy_needed(self, target_pos: np.ndarray, hazard_factor: float = 1.0) -> float:
        dist = float(np.linalg.norm(target_pos - self.pos))
        spd = max(0.6, self.genome.speed)
        steps_needed = dist / spd
        step_cost = (0.10 + 0.052 * (spd ** 2)) * hazard_factor
        return steps_needed * step_cost * self.genome.return_caution

    # ------------------------------------------------------------------ üzenetek feldolgozása
    @staticmethod
    def _spec(cat: str, rec_id: Optional[int], rec_lbl: str, intent: str, prio: int, pos: np.ndarray, sec: int,
              val: float, text: str, ref: Optional[str] = None, key: str = "", thread: str = "") -> Dict[str, Any]:
        return {
            "recipient_id": rec_id, "recipient_label": rec_lbl, "category": cat, "intent": intent, "priority": prio,
            "pos": pos.copy(), "sector_id": sec, "value": val, "text": text, "ref_msg_id": ref,
            "target_key": key, "thread_id": thread,
        }

    def _evaluate_task(self, msg: AgentMessage, dist: float, base_pos: np.ndarray, mode: str) -> Tuple[bool, str]:
        """Egy kapott feladat (TASK_ASSIGN / TASK_OFFER) elfogadásának döntése a saját állapot alapján."""
        prof = self.profile
        if self.state in ("FLEEING", "SEEKING_CHARGER", "RETURNING_BASE"):
            return False, f"most {self.state.lower()} állapotban vagyok"
        if self.energy_ratio < 0.38:
            return False, f"alacsony az akkum ({self.energy:.0f}%)"
        est = self.estimate_energy_needed(msg.pos) + self.estimate_energy_needed(base_pos) * 0.5
        if self.energy - est < 25.0:
            return False, "az energiám nem elég az odaútra és a hazatérésre"
        if dist > 95.0:
            return False, f"túl messze van ({dist:.0f} m)"
        if self.ally_target_pos is not None:
            return False, "egy társ megsegítésén dolgozom"
        score = 0.5 + prof.task_accept_bias + 0.25 * (self.genome.dirt_affinity - 1.0) + 0.2 * self.genome.cooperation - dist / 120.0
        if self.commitment is not None:
            cur_d = float(np.linalg.norm(self.commitment["pos"] - self.pos))
            if dist < cur_d * 0.6 and prof.task_accept_bias > 0:
                return True, "közelebbi célpont, lecserélem a korábbit"
            return False, "már van vállalt feladatom"
        return (score > 0.38), (f"{ROLE_LABEL[self.role_id]} szerepem és a távolság ({dist:.0f} m) alapján" if score > 0.38
                                else f"más a prioritásom ({ROLE_LABEL[self.role_id]}, {dist:.0f} m)")

    def _release_commitment(self, tick: int, reason: str, out: List[Dict[str, Any]], notify: bool = True) -> None:
        c = self.commitment
        if c is None:
            return
        self.memory.release_target(c["key"])
        self.commitment = None
        self.log_decision(tick, f"Célpont ({c['key']}) felszabadítva: {reason}")
        if notify:
            out.append(self._spec(
                "[TASK]", None, "TEAM", "TARGET_RELEASE", 1, c["pos"], self.assigned_sector, 0.0,
                f"Felszabadítom a(z) {c['key']} célpontot: {reason}.", key=c["key"],
            ))

    def _process_inbox(
        self, tick: int, mode: str, base_pos: np.ndarray, out: List[Dict[str, Any]]
    ) -> None:
        """A beérkezett üzenetek tényleges feldolgozása: állapotot módosít, és válaszüzeneteket ad az `out` listához."""
        while self.inbox:
            msg = self.inbox.pop(0)
            if tick - msg.sent_tick > msg.ttl:
                msg.status = "EXPIRED"
                msg.reaction_note = f"{self.name}: lejárt, mielőtt feldolgozhatta volna"
                continue
            msg.processed_by.append(self.id)
            if msg.processed_tick < 0:
                msg.processed_tick = tick
            self.memory.remember_peer(msg.sender_id, msg.sender_name, msg.origin if msg.origin is not None else msg.pos,
                                      self.memory.peers.get(msg.sender_id, {}).get("energy", 50.0), tick, role=msg.sender_role)
            replies = out

            if msg.intent == "SECTOR_CLEAR":
                self.memory.cleared_sectors[msg.sector_id] = tick
                msg.status = "ACCEPTED"
                if self.assigned_sector == msg.sector_id:
                    old_s = self.assigned_sector
                    for cand in range(6):
                        if cand not in self.memory.cleared_sectors:
                            self.assigned_sector = cand
                            break
                    else:
                        self.assigned_sector = (self.assigned_sector + 1) % 6
                    note = f"{msg.msg_id} ({msg.sender_name}): Szektor #{old_s+1} kész -> Váltás #{self.assigned_sector+1}-re"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"Üzenet ({msg.sender_name}) alapján szektorváltás: #{self.assigned_sector+1}")

            elif msg.intent == "SECTOR_CLAIM":
                if msg.sector_id == self.assigned_sector:
                    if self.genome.cooperation >= 0.42 or msg.sender_id < self.id:
                        self.memory.sector_claims[msg.sector_id] = (msg.sender_id, tick)
                        old_s = self.assigned_sector
                        self.assigned_sector = (self.assigned_sector + 1) % 6
                        msg.status = "ACCEPTED"
                        msg.reaction_note = f"{self.name} átengedte Szektor #{old_s+1}-t"
                        note = f"{msg.msg_id}: Szektor #{old_s+1} átengedve {msg.sender_name}-nek"
                        self.explanation.last_message_effect = note
                        self.record_comm_link(note)
                        if tick - self.last_comm_tick >= 18:
                            self.last_comm_tick = tick
                            replies.append(self._spec(
                                "[TASK]", msg.sender_id, msg.sender_name, "ACK_HELP", 1, self.pos, self.assigned_sector,
                                float(old_s),
                                f"Rendben {msg.sender_name}, átengedem a(z) #{old_s+1}. szektort, átveszem a(z) #{self.assigned_sector+1}. zónát.",
                                ref=msg.msg_id, thread=msg.thread_id or msg.msg_id))
                    else:
                        msg.status = "REJECTED"
                        msg.reaction_note = f"{self.name} elutasította a szektorátadást"

            elif msg.intent == "DIRT_SPOT":
                dist = float(np.linalg.norm(msg.pos - self.pos))
                self.memory.remember_dirt(msg.pos, msg.value, tick)
                if dist < 32.0 and self.energy_ratio > 0.38 and self.genome.dirt_affinity > 0.65:
                    msg.status = "ACCEPTED"
                    msg.reaction_note = f"{self.name} átvette a kosz-gócpontot"
                    note = f"{msg.msg_id} ({msg.sender_name}): Kosz-góc átvéve ({dist:.0f}m)"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"Célpont frissítve {msg.sender_name} kosz-észlelése alapján")
                else:
                    msg.status = "DELIVERED"

            elif msg.intent in ("TASK_ASSIGN", "TASK_OFFER"):
                key = msg.target_key or AgentMemory.target_key(msg.pos)
                dist = float(np.linalg.norm(msg.pos - self.pos))
                self.memory.remember_dirt(msg.pos, msg.value, tick)       # a tudás akkor is hasznos, ha nem vállalja
                if self.memory.target_claims.get(key, (None,))[0] not in (None, self.id):
                    msg.status = "REJECTED"
                    msg.reaction_note = f"{self.name}: a célpontot már lefoglalta valaki"
                    continue
                ok, why = self._evaluate_task(msg, dist, base_pos, mode)
                if ok:
                    if self.commitment is not None:
                        self._release_commitment(tick, "közelebbi célpontot vállaltam", replies)
                    self.commitment = {
                        "key": key, "pos": msg.pos.copy(), "since": tick, "expires": tick + int(70 + dist * 3),
                        "source": msg.sender_id, "src_name": msg.sender_name, "msg": msg.msg_id, "dist0": dist,
                    }
                    self.memory.claim_target(key, self.id, msg.pos, tick)
                    msg.status = "ACCEPTED"
                    msg.reaction_note = f"{self.name} elvállalta ({why})"
                    note = f"{msg.msg_id}: {msg.sender_name} feladata ({key}, {dist:.0f}m) ELVÁLLALVA"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"Feladat elvállalva {msg.sender_name} üzenetére: {key}")
                    replies.append(self._spec(
                        "[TASK]", None, "TEAM", "TARGET_CLAIM", 2, msg.pos, self.assigned_sector, dist,
                        f"Átvettem a(z) {key} célpontot ({msg.sender_name} jelzése), {dist:.0f} m-re vagyok. Ne menjetek oda.",
                        ref=msg.msg_id, key=key, thread=msg.thread_id or msg.msg_id))
                else:
                    msg.status = "REJECTED"
                    msg.reaction_note = f"{self.name} nem vállalta: {why}"
                    note = f"{msg.msg_id}: {msg.sender_name} feladata ELUTASÍTVA ({why})"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    if msg.intent == "TASK_ASSIGN":
                        replies.append(self._spec(
                            "[TASK]", msg.sender_id, msg.sender_name, "ACK", 1, self.pos, self.assigned_sector, 0.0,
                            f"Nem tudom átvenni a(z) {key} célpontot: {why}.", ref=msg.msg_id, key=key,
                            thread=msg.thread_id or msg.msg_id))

            elif msg.intent == "TARGET_CLAIM":
                key = msg.target_key
                self.memory.claim_target(key, msg.sender_id, msg.pos, tick)
                msg.status = "ACCEPTED"
                if key in self.memory.sent_offers:
                    self.memory.sent_offers.pop(key, None)
                    note = f"{msg.msg_id}: {msg.sender_name} átvette az általam jelzett célpontot ({key})"
                    msg.reaction_note = f"{self.name} tudomásul vette"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                if self.commitment is not None and self.commitment["key"] == key and msg.sender_id != self.id:
                    my_d = float(np.linalg.norm(self.commitment["pos"] - self.pos))
                    if msg.value < my_d - 0.01 or (abs(msg.value - my_d) <= 0.01 and msg.sender_id < self.id):
                        self._release_commitment(tick, f"{msg.sender_name} közelebb van", replies, notify=False)
                        self.memory.claim_target(key, msg.sender_id, msg.pos, tick)
                        msg.reaction_note = f"{self.name} átengedte a célpontot (távolabb volt)"

            elif msg.intent == "TARGET_RELEASE":
                self.memory.release_target(msg.target_key)
                msg.status = "ACCEPTED"
                note = f"{msg.msg_id}: {msg.sender_name} felszabadította a célpontot ({msg.target_key})"
                self.explanation.last_message_effect = note
                self.record_comm_link(note)

            elif msg.intent == "ACK":
                msg.status = "ACCEPTED"
                if msg.target_key == "HELP_COVERED":
                    if self.ally_target_id == msg.sender_id:
                        self.ally_target_id, self.ally_target_pos = None, None
                        self.log_decision(tick, f"{msg.sender_name} már kapott másik segítséget, a küldetésem megszűnt")
                        msg.reaction_note = f"{self.name} visszavonta a segítség-útját"
                elif msg.value < 0.5:
                    self.memory.rejected_by[msg.sender_id] = tick
                    self.memory.sent_offers.pop(msg.target_key, None)
                    note = f"{msg.msg_id}: {msg.sender_name} nem vállalta a célpontot, másik társat keresek"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)

            elif msg.intent == "LOW_BAT_SOS":
                self.memory.partner_battery[msg.sender_id] = (msg.value, msg.pos.copy(), tick)
                dist = float(np.linalg.norm(msg.pos - self.pos))
                est_help_cost = self.estimate_energy_needed(msg.pos) + self.estimate_energy_needed(base_pos) * 0.5
                can_afford = (self.energy - est_help_cost) > self.profile.help_energy_floor
                busy = self.ally_target_pos is not None and tick - self.ally_target_tick < 90
                willing = self.genome.cooperation >= 0.38 or self.role_id == "SUPPORT"

                if can_afford and willing and dist < 34.0 * (1.25 if self.role_id == "SUPPORT" else 1.0) and not busy:
                    self.ally_target_id = msg.sender_id
                    self.ally_target_pos = msg.pos.copy()
                    self.ally_target_tick = tick
                    msg.status = "ACCEPTED"
                    msg.reaction_note = f"{self.name} elindult energiát adni"
                    note = f"{msg.msg_id}: Segélykérés ELFOGADVA ({msg.sender_name}, {dist:.0f}m)"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"SOS elfogadva: indulás {msg.sender_name} megmentésére!")
                    self.last_comm_tick = min(self.last_comm_tick, tick - 14)
                    replies.append(self._spec(
                        "[ENERGY]", msg.sender_id, msg.sender_name, "OFFER_ENERGY", 2, self.pos, self.assigned_sector,
                        self.energy,
                        f"Vettem a segélykérést ({msg.sender_name})! Saját akkum {self.energy:.0f}%, indulok átadni energiát.",
                        ref=msg.msg_id, thread=msg.thread_id or msg.msg_id))
                else:
                    msg.status = "REJECTED"
                    why = "már segítek valakinek" if busy else f"saját akku: {self.energy:.0f}%"
                    msg.reaction_note = f"{self.name} nem tud segíteni ({why})"
                    note = f"{msg.msg_id}: Segélykérés ELUTASÍTVA ({why})"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)

            elif msg.intent == "OFFER_ENERGY":
                aw = self.awaiting_help
                if aw is None or aw.get("helper") in (None, msg.sender_id):
                    self.awaiting_help = {
                        "helper": msg.sender_id, "name": msg.sender_name, "pos": msg.pos.copy(), "since": tick,
                        "e0": self.energy_received if aw is None else aw["e0"],
                    }
                    msg.status = "ACCEPTED"
                    note = f"{msg.msg_id}: {msg.sender_name} segít, lassítok és közeledek"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"{msg.sender_name} segítséget ígért: energiatakarékos mód, találkozás")
                    replies.append(self._spec(
                        "[HELP]", msg.sender_id, msg.sender_name, "HELP_ACCEPTED", 2, self.pos, self.assigned_sector,
                        self.energy, f"Köszönöm {msg.sender_name}, várlak. Lassítok és közeledek a találkozáshoz.",
                        ref=msg.msg_id, thread=msg.thread_id or msg.msg_id))
                else:
                    msg.status = "REJECTED"
                    msg.reaction_note = f"{self.name}: már van segítője"
                    replies.append(self._spec(
                        "[HELP]", msg.sender_id, msg.sender_name, "ACK", 1, self.pos, self.assigned_sector, 0.0,
                        f"Köszönöm {msg.sender_name}, de már segít valaki.", ref=msg.msg_id, key="HELP_COVERED",
                        thread=msg.thread_id or msg.msg_id))

            elif msg.intent == "HELP_ACCEPTED":
                msg.status = "ACCEPTED"
                note = f"{msg.msg_id}: {msg.sender_name} visszaigazolta a segítséget"
                self.explanation.last_message_effect = note
                self.record_comm_link(note)

            elif msg.intent == "THREAT_DETECTED":
                try:
                    tid = int(msg.target_key[1:])
                except ValueError:
                    tid = -1
                if mode in ("CLASSIC", "TDM") and tid >= 0 and tid != self.id:
                    self.memory.threats[tid] = (msg.pos.copy(), tick, float(msg.value))
                    strong_me = self.genome.aggressiveness > 0.62 and self.energy_ratio > 0.6
                    msg.status = "ACCEPTED"
                    msg.reaction_note = f"{self.name}: {'szembeszáll' if strong_me else 'kitér'}"
                    note = f"{msg.msg_id}: Veszélyjelzés ({msg.sender_name}) -> {'szembeszállás' if strong_me else 'kitérés'} a(z) R-{tid+1:02d} elől"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)
                    self.log_decision(tick, f"Veszélyjelzés: R-{tid+1:02d} támadó, {'ellene fordulok' if strong_me else 'elkerülöm'}")
                else:
                    msg.status = "DELIVERED"

            elif msg.intent == "STATUS_UPDATE":
                msg.status = "ACCEPTED"

            elif msg.intent == "HAZARD_WARN":
                self.memory.remember_hazard(msg.pos, tick)
                msg.status = "ACCEPTED"
                note = f"{msg.msg_id}: Veszélyzóna rögzítve a memóriában ({msg.sender_name})"
                self.explanation.last_message_effect = note
                self.record_comm_link(note)

            elif msg.intent == "ENEMY_PING":
                if mode == "TDM" and self.energy_ratio > 0.35 and self.genome.aggressiveness > 0.42:
                    self.memory.remember_dirt(msg.pos, 10.0, tick)
                    msg.status = "ACCEPTED"
                    note = f"{msg.msg_id}: Csapattárs ({msg.sender_name}) célpontjára ráfordulás"
                    self.explanation.last_message_effect = note
                    self.record_comm_link(note)

    # ------------------------------------------------------------------ Multi-Agent segédek a döntéshez
    def _tend_commitment(self, tick: int, base_pos: np.ndarray, out: List[Dict[str, Any]]) -> bool:
        """Az elfogadott megosztott célpont ellenőrzése; teljesülés / lejárat / energiahiány esetén felszabadítja."""
        c = self.commitment
        if c is None:
            return False
        d = float(np.linalg.norm(c["pos"] - self.pos))
        reason = None
        if d < 4.0:
            reason = "teljesítettem (odaértem)"
        elif tick > c["expires"]:
            reason = "lejárt a vállalási idő"
        elif self.energy - self.estimate_energy_needed(c["pos"]) - self.estimate_energy_needed(base_pos) * 0.5 < 12.0:
            reason = "nincs elég energiám hozzá"
        elif any(float(np.linalg.norm(c["pos"] - u)) < 4.0 for u, _t in self.memory.unreachable_targets):
            reason = "elérhetetlen"
        if reason:
            self._release_commitment(tick, reason, out)
            return False
        return True

    def _commitment_move(self) -> Tuple[np.ndarray, str]:
        c = self.commitment
        v = c["pos"] - self.pos
        d = float(np.linalg.norm(v))
        self.state = "CLEANING"
        self.target_pos = c["pos"].copy()
        self.current_target_desc = f"Vállalt célpont {c['key']} ({d:.0f}m) - {c['src_name']} jelzésére"
        self.next_planned_action = "Odahaladás, felszívás, majd a célpont felszabadítása"
        return (v / max(d, 1e-5)) * 1.85, f"{c['src_name']} üzenete alapján vállalt feladat, saját E={self.energy:.0f}%"

    def _frontier_move(self, tick: int, gw: int, gh: int, obstacle_grid: np.ndarray, weight: float = 1.35
                       ) -> Tuple[np.ndarray, str]:
        """Felderítő mozgás a legközelebbi még nem látott területre (ritkán számolt cél, nem minden tickben)."""
        if (self.frontier is None or tick - self.frontier_tick > 90
                or float(np.linalg.norm(self.frontier - self.pos)) < 6.0):
            self.frontier = self.memory.pick_frontier(self.pos, gw, gh, obstacle_grid, tick)
            self.frontier_tick = tick
        if self.frontier is None:
            return self.vel * 1.1, "Nincs több felderítetlen terület"
        v = self.frontier - self.pos
        d = float(np.linalg.norm(v))
        self.target_pos = self.frontier.copy()
        self.current_target_desc = f"Felderítés: ismeretlen terület ({d:.0f}m)"
        return (v / max(d, 1e-5)) * weight, f"Felderítő szerep: a(z) ({int(self.frontier[0])}, {int(self.frontier[1])}) körzet még nem látott"

    def _pick_task_candidate(self, spot: np.ndarray, tick: int) -> Optional[Tuple[int, str]]:
        """A célpont megosztására legalkalmasabb ismert társ (szerep, távolság, energia, korábbi elutasítás alapján)."""
        role_w = {"CLEANER": 1.0, "GENERAL": 0.6, "SUPPORT": 0.5, "HUNTER": 0.3, "SCOUT": 0.1}
        best, best_s = None, 0.0
        for pid, p in self.memory.peers.items():
            if tick - p["tick"] > 60 or pid in self.memory.rejected_by:
                continue
            if float(np.linalg.norm(p["pos"] - self.pos)) > 60.0:
                continue
            s = role_w.get(p.get("role", "GENERAL"), 0.5) - float(np.linalg.norm(p["pos"] - spot)) / 100.0 + p["energy"] / 250.0
            if s > best_s:
                best, best_s = (pid, p["name"]), s
        return best

    def decide_and_move(
        self,
        sensor_data: Dict[str, Any],
        grid_width: int,
        grid_height: int,
        obstacle_grid: np.ndarray,
        mode: str = "CLASSIC",
        tick: int = 0,
        hazard_grid: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        self.prev_pos = self.pos.copy()
        self.prev_angle = self.angle

        if not self.alive:
            self.state = "DISABLED"
            return []

        if mode == "ANTARCTIC" and self.reached_goal:
            self.state = "REACHED_GOAL"
            self.current_target_desc = "B-Kutatóállomás (Célba ért)"
            self.next_planned_action = "Dokkolva / Küldetés teljesítve"
            return []

        self.ai_decisions_count += 1
        self.memory.cleanup(tick)
        outgoing_specs: List[Dict[str, Any]] = []
        prof = self.profile
        if self.ally_target_pos is not None and tick - self.ally_target_tick > 90:
            self.ally_target_id, self.ally_target_pos = None, None     # lejárt segítő-küldetés

        dirt_vec = sensor_data["dirt_vec"]
        dirt_dist = sensor_data["dirt_dist"]
        visible_dirt_count = sensor_data.get("visible_dirt_count", 0)
        agent_vec = sensor_data["agent_vec"]
        agent_dist = sensor_data["agent_dist"]
        neighbor_energy = sensor_data["agent_energy"]
        neighbor_id = sensor_data.get("neighbor_id", None)
        neighbor_name = sensor_data.get("neighbor_name", "MIND")
        enemy_vec = sensor_data.get("enemy_vec", agent_vec)
        enemy_dist = sensor_data.get("enemy_dist", agent_dist)
        enemy_energy = sensor_data.get("enemy_energy", neighbor_energy)
        ally_low_vec = sensor_data.get("ally_low_vec", np.zeros(2, dtype=np.float32))
        ally_low_dist = sensor_data.get("ally_low_dist", 999.0)
        ally_low_name = sensor_data.get("ally_low_name", "TEAM")
        charger_vec = sensor_data.get("charger_vec", np.zeros(2, dtype=np.float32))
        charger_dist = sensor_data.get("charger_dist", 999.0)
        base_pos = sensor_data.get("base_pos", np.array([10.0, 10.0], dtype=np.float32))
        goal_pos = sensor_data.get("goal_pos", np.array([grid_width - 10.0, grid_height - 10.0], dtype=np.float32))
        sector_center = sensor_data.get("sector_center", np.array([grid_width / 2.0, grid_height / 2.0], dtype=np.float32))
        sector_dirt_count = sensor_data.get("sector_dirt_count", 0)
        wall_vec = sensor_data["wall_vec"]
        hazard_vec = sensor_data.get("hazard_vec", np.zeros(2, dtype=np.float32))

        if dirt_dist <= self.SENSOR_RANGE:
            self.memory.remember_dirt(self.pos + dirt_vec * dirt_dist, float(visible_dirt_count), tick)
            self.memory.last_dirt_seen = tick
        self.memory.mark_visit(self.pos, tick)
        if neighbor_id is not None and agent_dist <= self.SENSOR_RANGE:
            self.memory.remember_peer(neighbor_id, neighbor_name, self.pos + agent_vec * agent_dist, neighbor_energy, tick)

        cx_curr, cy_curr = int(self.pos[0]), int(self.pos[1])
        in_hazard = False
        if hazard_grid is not None and 0 <= cx_curr < grid_width and 0 <= cy_curr < grid_height:
            in_hazard = hazard_grid[cx_curr, cy_curr] > 0
            if in_hazard:
                self.memory.remember_hazard(self.pos.copy(), tick)

        self.memory.last_positions.append(self.pos.copy())
        if len(self.memory.last_positions) > 10:
            old_p = self.memory.last_positions.pop(0)
            if float(np.linalg.norm(self.pos - old_p)) < 1.2 and not self.returned_to_base:
                self.memory.stuck_counter += 1
                if self.memory.stuck_counter >= 4 and self.target_pos is not None:
                    self.memory.unreachable_targets.append((self.target_pos.copy(), tick))
                    self.assigned_sector = (self.assigned_sector + 1) % 6
                    self.memory.stuck_counter = 0
                    self.log_decision(tick, "Visszacsatolás: Célpont akadályozott, új útvonal/szektor választása")
            else:
                self.memory.stuck_counter = 0

        self._process_inbox(tick, mode, base_pos, outgoing_specs)
        commit_active = self._tend_commitment(tick, base_pos, outgoing_specs)

        desire_vec = np.zeros(2, dtype=np.float32)
        speed_multiplier = 1.0
        alternatives: List[Tuple[str, float]] = []
        threat_note = ""
        near_threat = None
        strong_me = False

        vec_to_base = base_pos - self.pos
        dist_to_base = float(np.linalg.norm(vec_to_base))
        dir_to_base = vec_to_base / max(dist_to_base, 1e-5)
        est_base_energy = self.estimate_energy_needed(base_pos, hazard_factor=1.0)

        if mode == "ANTARCTIC":
            vec_to_goal = goal_pos - self.pos
            dist_to_goal = float(np.linalg.norm(vec_to_goal))
            dir_to_goal = vec_to_goal / max(dist_to_goal, 1e-5)
            est_goal_energy = self.estimate_energy_needed(goal_pos, hazard_factor=1.18)

            if dist_to_goal < 4.5:
                self.reached_goal = True
                self.state = "REACHED_GOAL"
                self.log_decision(tick, "Célállomás (B) sikeresen elérve!")
                return outgoing_specs

            u_goal = 75.0 + (1.0 - dist_to_goal / 100.0) * 25.0
            u_help = (
                92.0
                if (ally_low_dist < 18.0 and self.energy_ratio > 0.55 and self.genome.cooperation > 0.45)
                else 10.0
            )
            u_abort = (
                88.0
                if (self.energy < est_goal_energy * 0.72 and dist_to_base < dist_to_goal * 0.60)
                else 12.0
            )
            u_scavenge = 82.0 if (dirt_dist < 10.0 and self.energy_ratio < 0.78) else 20.0
            if u_help > 20.0:
                u_help += prof.help_bonus
            alternatives = [
                ("Haladás B-állomásra", u_goal),
                ("Társ mentése", u_help),
                ("Energiakristály felvétele", u_scavenge),
                ("Visszafordulás A-bázisra", u_abort),
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)
            best_label, _ = alternatives[0]

            if best_label == "Társ mentése":
                self.state = "HELPING_ALLY"
                desire_vec = ally_low_vec * 2.0
                self.target_pos = self.pos + ally_low_vec * ally_low_dist
                self.current_target_desc = f"Bajba jutott társ: {ally_low_name}"
                self.next_planned_action = "Megközelítés és energiaátadás"
                reason = f"Kooperáció={self.genome.cooperation:.2f}, saját E={self.energy:.0f}% > 55%"
            elif best_label == "Visszafordulás A-bázisra":
                self.state = "RETURNING_BASE"
                self.mission_aborted = True
                desire_vec = dir_to_base * 1.9
                speed_multiplier = 0.75
                self.target_pos = base_pos.copy()
                self.current_target_desc = f"A-Bázis ({dist_to_base:.0f}m)"
                self.next_planned_action = "Küldetés megszakítása energiahiány miatt"
                reason = f"Saját E={self.energy:.1f} < Becsült B-igény ({est_goal_energy:.1f})"
            elif best_label == "Energiakristály felvétele":
                self.state = "CLEANING"
                desire_vec = dirt_vec * 1.6 + dir_to_goal * 0.6
                self.target_pos = self.pos + dirt_vec * dirt_dist
                self.current_target_desc = f"Útközbeni energiaforrás ({dirt_dist:.1f}m)"
                self.next_planned_action = "Energia felvétele, majd tovább B-re"
                reason = f"Energiaforrás {dirt_dist:.1f}m-re, akku={self.energy:.0f}%"
            else:
                self.state = "MOVING_TO_GOAL"
                speed_multiplier = 0.74 if self.energy < est_goal_energy else 1.0
                desire_vec = dir_to_goal * 1.70
                self.target_pos = goal_pos.copy()
                self.current_target_desc = f"B-Kutatóállomás ({dist_to_goal:.0f}m)"
                self.next_planned_action = (
                    "Energiatakarékos cirkálás B felé"
                    if speed_multiplier < 1.0
                    else "Célirányos haladás B állomásra"
                )
                reason = f"Cél távolság={dist_to_goal:.0f}m, becsült E-igény={est_goal_energy:.1f}"

            desire_vec += hazard_vec * (0.7 + 1.4 * self.genome.exploration)

        elif mode == "COOP":
            # Hazatérés: lemerülés előtt, vagy ha hosszú ideje nincs több munka (a szenzor 999-et ad, ha nincs kosz a
            # látótávolságon belül - ez önmagában NEM jelenti, hogy a pályán nincs több kosz, főleg nagy pályán)
            idle_no_work = dirt_dist > 500.0 and sector_dirt_count == 0 and tick - self.memory.last_dirt_seen > 120
            u_return = 96.0 if (self.energy <= est_base_energy + 9.0 or idle_no_work) else 18.0
            u_help = (
                89.0
                if (
                    (ally_low_dist < 22.0 or self.ally_target_pos is not None)
                    and self.energy_ratio > 0.56
                    and self.genome.cooperation > 0.42
                )
                else 14.0
            )
            u_clean_local = (84.0 if dirt_dist <= self.SENSOR_RANGE else 35.0) + prof.clean_bonus
            if u_help > 20.0:
                u_help += prof.help_bonus
            u_sector = 72.0
            mem_spot_coop = self.memory.best_unclaimed_dirt(self.pos, self.id, 140.0) if dirt_dist > self.SENSOR_RANGE else None
            alternatives = [
                ("Visszatérés a bázisra", u_return),
                ("Csapattárs töltése", u_help),
                ("Észlelt kosz takarítása", u_clean_local),
                (f"Szektor #{self.assigned_sector+1} tisztítása", u_sector),
            ]
            if commit_active:
                alternatives.append(("Átvett célpont teljesítése", 86.0))
            if mem_spot_coop is not None and not commit_active:
                alternatives.append(("Társ által jelzett gócpont", 78.0))
            alternatives.sort(key=lambda x: x[1], reverse=True)
            best_label, _ = alternatives[0]

            if best_label == "Visszatérés a bázisra":
                if dist_to_base < 4.5:
                    self.returned_to_base = True
                    self.state = "RETURNING_BASE"
                    speed_multiplier = 0.15
                    self.target_pos = base_pos.copy()
                    self.current_target_desc = "Bázison biztonságban dokkolva"
                    self.next_planned_action = "Töltés és készenlét"
                else:
                    self.returned_to_base = False
                    self.state = "RETURNING_BASE"
                    desire_vec = dir_to_base * 2.1
                    speed_multiplier = 0.85
                    self.target_pos = base_pos.copy()
                    self.current_target_desc = f"A-Bázis ({dist_to_base:.0f}m)"
                    self.next_planned_action = "Hazatérés lemerülés előtt"
                reason = f"Hazatérési E-igény={est_base_energy:.1f}, aktuális E={self.energy:.1f}"

            elif best_label == "Csapattárs töltése":
                self.returned_to_base = False
                self.state = "HELPING_ALLY"
                if ally_low_dist < 22.0:
                    desire_vec = ally_low_vec * 2.0
                    self.target_pos = self.pos + ally_low_vec * ally_low_dist
                    t_name = ally_low_name
                elif self.ally_target_pos is not None:
                    v_a = self.ally_target_pos - self.pos
                    d_a = float(np.linalg.norm(v_a))
                    if d_a < 3.5:
                        self.ally_target_pos = None
                    desire_vec = v_a / max(d_a, 1e-5) * 1.9
                    self.target_pos = self.ally_target_pos
                    t_name = f"R-{self.ally_target_id+1:02d}" if self.ally_target_id is not None else "Társ"
                else:
                    t_name = "Társ"
                self.current_target_desc = f"Energiaátadás -> {t_name}"
                self.next_planned_action = "Odahaladás és akkumulátor-megosztás"
                reason = f"Kooperáció={self.genome.cooperation:.2f}, saját tartalék={self.energy:.0f}%"

            elif best_label == "Átvett célpont teljesítése":
                self.returned_to_base = False
                desire_vec, reason = self._commitment_move()
            elif best_label == "Társ által jelzett gócpont":
                self.returned_to_base = False
                v_m = mem_spot_coop.pos - self.pos
                d_m = float(np.linalg.norm(v_m))
                desire_vec = v_m / max(d_m, 1e-5) * 1.7
                self.state = "COOPERATING"
                self.target_pos = mem_spot_coop.pos.copy()
                self.current_target_desc = f"Jelzett gócpont ({d_m:.0f}m), nincs lefoglalva"
                self.next_planned_action = "Odahaladás, felszívás"
                reason = "Társ által jelzett, más által nem foglalt szemétgóc"
            else:
                self.returned_to_base = False
                if sector_dirt_count == 0:
                    old_sec = self.assigned_sector
                    self.memory.cleared_sectors[old_sec] = tick
                    for cand in range(6):
                        if cand not in self.memory.cleared_sectors:
                            self.assigned_sector = cand
                            break
                    else:
                        self.assigned_sector = (self.assigned_sector + 1) % 6

                    if tick - self.last_comm_tick >= 22:
                        self.last_comm_tick = tick
                        outgoing_specs.append({
                            "recipient_id": None,
                            "recipient_label": "TEAM",
                            "category": "[TASK]",
                            "intent": "SECTOR_CLEAR",
                            "priority": 2,
                            "pos": self.pos.copy(),
                            "sector_id": old_sec,
                            "value": 0.0,
                            "text": f"A(z) #{old_sec+1}. szektort teljesen megtisztítottam. Átváltok a(z) #{self.assigned_sector+1}. szektorra.",
                        })

                self.state = "COOPERATING" if len(self.known_partners) > 0 else "CLEANING"
                vec_sec = sector_center - self.pos
                dist_sec = float(np.linalg.norm(vec_sec))
                dir_sec = vec_sec / max(dist_sec, 1e-5)

                if dirt_dist <= self.SENSOR_RANGE:
                    desire_vec = dirt_vec * self.genome.dirt_affinity + dir_sec * 0.45
                    self.target_pos = self.pos + dirt_vec * dirt_dist
                    self.current_target_desc = f"Kosz-góc ({dirt_dist:.1f}m, Szektor #{self.assigned_sector+1})"
                    self.next_planned_action = "Szennyeződés felszívása"
                    reason = f"Szenzor kosz-észlelés {dirt_dist:.1f}m-re (Affinitás={self.genome.dirt_affinity:.2f})"
                elif prof.explore:
                    desire_vec, reason = self._frontier_move(tick, grid_width, grid_height, obstacle_grid)
                    desire_vec = desire_vec + dirt_vec * 0.4
                    self.next_planned_action = "Új terület felderítése, felfedezések megosztása"
                else:
                    desire_vec = dir_sec * 1.45 + dirt_vec * 0.4
                    self.target_pos = sector_center.copy()
                    self.current_target_desc = f"Szektor #{self.assigned_sector+1} munkaterület"
                    self.next_planned_action = "Szektor felderítése és tisztítása"
                    reason = f"Kijelölt munkaterület: Szektor #{self.assigned_sector+1} ({sector_dirt_count} góc)"

        elif mode == "TDM":
            attack_score = (self.genome.aggressiveness * (1.25 - 0.30 * (1.0 - self.energy_ratio))) * 100.0
            u_flee = 94.0 if (self.energy_ratio < 0.24 and enemy_dist < 15.0) else 15.0
            u_charge = 88.0 if (self.energy_ratio < 0.30 and charger_dist < 28.0) else 12.0
            u_help = 83.0 if (ally_low_dist < 15.0 and self.energy_ratio > 0.62 and self.genome.cooperation > 0.50) else 10.0
            u_hunt = max(30.0, attack_score) if enemy_dist <= self.SENSOR_RANGE else 25.0
            u_clean = 62.0 + prof.clean_bonus
            u_hunt *= prof.hunt_mult
            u_flee *= prof.flee_mult
            if u_help > 20.0:
                u_help += prof.help_bonus

            alternatives = [
                ("Ellenfél támadása", u_hunt),
                ("Takarítás / Energiagyűjtés", u_clean),
                ("Töltőállomás felkeresése", u_charge),
                ("Taktikai visszavonulás", u_flee),
                ("Csapattárs támogatása", u_help),
            ]
            if commit_active:
                alternatives.append(("Átvett célpont teljesítése", 66.0 + prof.clean_bonus * 0.2))
            alternatives.sort(key=lambda x: x[1], reverse=True)
            best_label, _ = alternatives[0]

            if best_label == "Taktikai visszavonulás":
                self.state = "FLEEING"
                desire_vec = -enemy_vec * 2.1 + charger_vec * 0.8
                self.target_pos = self.pos - enemy_vec * 10.0
                self.current_target_desc = "Kitérés ellenséges tűz elől"
                self.next_planned_action = "Távolságnövelés és fedezékkeresés"
                reason = f"Kritikus akku ({self.energy:.0f}%) és közeli ellenfél ({enemy_dist:.1f}m)"
            elif best_label == "Töltőállomás felkeresése":
                self.state = "SEEKING_CHARGER"
                desire_vec = charger_vec * 2.0 - enemy_vec * 0.8
                self.target_pos = self.pos + charger_vec * charger_dist
                self.current_target_desc = f"Töltődokk ({charger_dist:.1f}m)"
                self.next_planned_action = "Akkumulátor újratöltése"
                reason = f"Alacsony energia ({self.energy:.0f}%), töltő {charger_dist:.1f}m-re"
            elif best_label == "Csapattárs támogatása":
                self.state = "HELPING_ALLY"
                desire_vec = ally_low_vec * 1.9
                self.target_pos = self.pos + ally_low_vec * ally_low_dist
                self.current_target_desc = f"Csapattárs: {ally_low_name}"
                self.next_planned_action = "Harci energiaátadás"
                reason = f"Csapattárs ({ally_low_name}) akkuja kritikus, saját E={self.energy:.0f}%"
            elif best_label == "Átvett célpont teljesítése":
                desire_vec, reason = self._commitment_move()
            elif best_label == "Ellenfél támadása":
                self.state = "HUNTING"
                desire_vec = enemy_vec * (1.6 + self.genome.aggressiveness)
                self.target_pos = self.pos + enemy_vec * enemy_dist
                self.current_target_desc = f"Ellenség ({enemy_dist:.1f}m, E:{enemy_energy:.0f}%)"
                self.next_planned_action = "Közelharc és energiaelszívás"
                reason = f"Agresszió={self.genome.aggressiveness:.2f}, ellenfél={enemy_dist:.1f}m"
            else:
                self.state = "CLEANING"
                desire_vec = dirt_vec * self.genome.dirt_affinity + enemy_vec * 0.35
                self.target_pos = self.pos + dirt_vec * min(dirt_dist, 18.0)
                self.current_target_desc = "Járőrözés és energiagyűjtés"
                self.next_planned_action = "Kosz felszívása energiapótláshoz"
                reason = f"Nincs közvetlen célpont, energiagyűjtés (E={self.energy:.0f}%)"

        else:
            wants_to_hunt = (
                self.energy_ratio < self.genome.risk_tolerance
                or self.genome.aggressiveness > 0.70
            )
            u_charge = 86.0 if (self.energy_ratio < 0.26 and charger_dist < 18.0 and self.genome.aggressiveness < 0.65) else 10.0
            u_hunt = (
                (self.genome.aggressiveness * (1.3 - 0.5 * self.energy_ratio)) * 100.0
                if (agent_dist <= self.SENSOR_RANGE and wants_to_hunt)
                else 15.0
            )
            u_flee = (
                ((1.0 - self.genome.aggressiveness) * (neighbor_energy / 100.0)) * 110.0
                if agent_dist < 11.0
                else 10.0
            )
            u_clean = (68.0 if dirt_dist <= self.SENSOR_RANGE else 52.0) + prof.clean_bonus
            u_hunt *= prof.hunt_mult
            u_flee *= prof.flee_mult

            # Veszélyjelzés hatása (THREAT_DETECTED): a jelzett támadó közelében a gyengébb menekül, az erős szembeszáll
            threat_note = ""
            fresh_threats = [(tid, tp, tt) for tid, (tp, tt, _sev) in self.memory.threats.items() if tick - tt <= 40]
            near_threat = None
            for tid, tp, tt in fresh_threats:
                dtp = float(np.linalg.norm(tp - self.pos))
                if dtp < 32.0 and (near_threat is None or dtp < near_threat[2]):
                    near_threat = (tid, tp, dtp)
            strong_me = self.genome.aggressiveness > 0.62 and self.energy_ratio > 0.6
            if near_threat is not None and neighbor_id == near_threat[0] and agent_dist < 14.0:
                if strong_me:
                    u_hunt = max(u_hunt, 60.0) * 1.3
                    threat_note = f" | Veszélyjelzés: R-{near_threat[0]+1:02d} ellen fordul"
                else:
                    u_flee += 28.0 * (1.0 - self.genome.aggressiveness) + 8.0
                    threat_note = f" | Veszélyjelzés: R-{near_threat[0]+1:02d} elől menekül"

            alternatives = [
                ("Kosz feltakarítása", u_clean),
                ("Kannibál támadás", u_hunt),
                ("Menekülés erősebb elől", u_flee),
                ("Töltődokk keresése", u_charge),
            ]
            if commit_active:
                alternatives.append(("Átvett célpont teljesítése", 72.0))
            alternatives.sort(key=lambda x: x[1], reverse=True)
            best_label, _ = alternatives[0]

            if best_label == "Töltődokk keresése":
                self.state = "SEEKING_CHARGER"
                desire_vec = charger_vec * 1.9
                self.target_pos = self.pos + charger_vec * charger_dist
                self.current_target_desc = f"Töltődokk ({charger_dist:.1f}m)"
                self.next_planned_action = "Akkumulátor töltése"
                reason = f"Energia {self.energy:.0f}% < 26%, dokk {charger_dist:.1f}m-re"
            elif best_label == "Kannibál támadás":
                self.state = "HUNTING"
                desire_vec = agent_vec * (1.5 + self.genome.aggressiveness)
                self.target_pos = self.pos + agent_vec * agent_dist
                self.current_target_desc = f"Célpont: {neighbor_name} ({agent_dist:.1f}m)"
                self.next_planned_action = "Akkumulátor-elszívás megkísérlése"
                reason = f"Agresszió={self.genome.aggressiveness:.2f}, E={self.energy:.0f}% < küszöb ({self.genome.risk_tolerance*100:.0f}%)"
            elif best_label == "Átvett célpont teljesítése":
                desire_vec, reason = self._commitment_move()
            elif best_label == "Menekülés erősebb elől":
                self.state = "FLEEING"
                desire_vec = -agent_vec * 1.85
                self.target_pos = self.pos - agent_vec * 9.0
                self.current_target_desc = f"Kitérés {neighbor_name} elől"
                self.next_planned_action = "Menekülés ellenkező irányba"
                reason = f"Szomszéd ({neighbor_name}) túl közel ({agent_dist:.1f}m) és erős ({neighbor_energy:.0f}%)"
            else:
                self.state = "CLEANING"
                if dirt_dist <= self.SENSOR_RANGE:
                    desire_vec = dirt_vec * self.genome.dirt_affinity
                    self.target_pos = self.pos + dirt_vec * dirt_dist
                    self.current_target_desc = f"Észlelt kosz ({dirt_dist:.1f}m)"
                    reason = f"Szenzoron belül kosz ({dirt_dist:.1f}m), affinitás={self.genome.dirt_affinity:.2f}"
                elif (mem_spot := self.memory.best_unclaimed_dirt(self.pos, self.id)) is not None:
                    v_m = mem_spot.pos - self.pos
                    d_m = float(np.linalg.norm(v_m))
                    desire_vec = (v_m / max(d_m, 1e-5)) * 1.3
                    self.target_pos = mem_spot.pos.copy()
                    self.current_target_desc = f"Memóriában tárolt kosz ({d_m:.0f}m)"
                    reason = "Nincs közeli kosz, memóriában rögzített gócpont felkeresése"
                elif prof.explore:
                    desire_vec, reason = self._frontier_move(tick, grid_width, grid_height, obstacle_grid)
                else:
                    desire_vec = self.vel * 1.1
                    self.target_pos = self.pos + self.vel * 12.0
                    self.current_target_desc = "Ismeretlen zóna felderítése"
                    reason = "Szenzoron belül nincs kosz, felderítő mozgás"
                self.next_planned_action = "Takarítás és energiagyűjtés"

        reason += threat_note

        # ---- módoktól független kiegészítések (üzenetekből származó, tényleges viselkedésváltozások)
        best_u = alternatives[0][1] if alternatives else 0.0

        # 1) Segítő-küldetés (SOS elfogadva): ANTARCTIC/TDM/CLASSIC módban is ténylegesen odamegy (COOP-ban már van saját ága)
        if mode != "COOP" and self.ally_target_pos is not None and best_u < 88.0 and self.energy_ratio > 0.45:
            if ally_low_dist < 22.0 and sensor_data.get("ally_low_id") == self.ally_target_id:
                v_a, d_a = ally_low_vec * ally_low_dist, float(ally_low_dist)       # élő szenzor-adat
            else:
                v_a = self.ally_target_pos - self.pos
                d_a = float(np.linalg.norm(v_a))
            if d_a < 3.5:
                self.ally_target_id, self.ally_target_pos = None, None
            else:
                desire_vec = v_a / max(d_a, 1e-5) * 1.9
                self.state = "HELPING_ALLY"
                self.target_pos = self.pos + v_a
                t_name = f"R-{self.ally_target_id+1:02d}" if self.ally_target_id is not None else "Társ"
                self.current_target_desc = f"Segítség -> {t_name} ({d_a:.0f}m)"
                self.next_planned_action = "Odahaladás és energiaátadás"
                reason = f"SOS-üzenet alapján vállalt segítség, saját E={self.energy:.0f}%"
                best_label = "Segélykérés teljesítése"
                alternatives.insert(0, (best_label, 90.0))

        # 2) Veszélyjelzés: a jelzett támadó körzetét a gyengébb robot kerüli (nem szembeszálló, nem menekülő állapotban)
        if mode == "CLASSIC" and near_threat is not None and not strong_me and self.state not in ("HUNTING", "SEEKING_CHARGER"):
            away = self.pos - near_threat[1]
            d_t = max(float(np.linalg.norm(away)), 1e-5)
            desire_vec = desire_vec + (away / d_t) * (0.9 * (1.0 - self.genome.aggressiveness) + 0.2) * (1.0 - d_t / 32.0) * 2.0
            if "Veszélyjelzés" not in reason:
                reason += f" | Veszélyjelzés: R-{near_threat[0]+1:02d} körzetét kerüli ({d_t:.0f}m)"

        # 3) Segítségre várok: energiatakarékos mód + találkozás a segítővel (a kérő is módosítja a viselkedését)
        if self.awaiting_help is not None:
            aw = self.awaiting_help
            if self.energy_received - aw["e0"] >= 6.0:
                self.log_decision(tick, "Megkaptam a segítséget (energia érkezett)")
                self.awaiting_help = None
            elif tick - aw["since"] > 80 or self.energy_ratio > 0.5:
                self.awaiting_help = None
            else:
                speed_multiplier *= 0.6
                if aw.get("helper") is not None and self.state in ("CLEANING", "COOPERATING", "WALKING"):
                    v_h = aw["pos"] - self.pos
                    d_h = float(np.linalg.norm(v_h))
                    if d_h < 45.0:
                        desire_vec = desire_vec * 0.4 + (v_h / max(d_h, 1e-5)) * 1.4
                        self.state = "WAITING_HELP"
                        self.current_target_desc = f"Találkozás a segítővel: {aw['name']} ({d_h:.0f}m)"
                        self.next_planned_action = "Lassított mozgás, energiaátvétel"
                        reason += f" | {aw['name']} segít: találkozó, lassított mód"

        self.explanation.current_goal = self.current_target_desc
        self.explanation.perceived_env = (
            f"Kosz:{dirt_dist:.0f}m ({visible_dirt_count}db) | Társ:{agent_dist:.0f}m | Akku:{self.energy:.0f}%"
        )
        self.explanation.considered_alternatives = alternatives[:4]
        self.explanation.chosen_action = f"{best_label} ({self.state})"
        self.explanation.measurable_reason = reason

        comm_cooldown = int(28 / max(0.35, self.genome.comm_tendency) * prof.comm_cooldown_mult)

        # ---- Sürgős üzenetek: saját időzítésük van, ezért az általános kommunikációs várakozás nem késlelteti őket
        if (self.energy_ratio < 0.26 and tick - self.last_sos_tick >= 55 and self.genome.cooperation > 0.30
                and self.awaiting_help is None):
            self.last_comm_tick = tick
            self.last_sos_tick = tick
            self.awaiting_help = {"helper": None, "name": "", "pos": self.pos.copy(), "since": tick, "e0": self.energy_received}
            rec_lbl = neighbor_name if (neighbor_id is not None and agent_dist < 30.0) else "TEAM"
            outgoing_specs.append({
                "recipient_id": neighbor_id if agent_dist < 30.0 else None,
                "recipient_label": rec_lbl,
                "category": "[HELP]",
                "intent": "LOW_BAT_SOS",
                "priority": 3,
                "pos": self.pos.copy(),
                "sector_id": self.assigned_sector,
                "value": self.energy,
                "text": f"Az energiaszintem {self.energy:.0f}%-ra csökkent (Szektor #{self.assigned_sector+1}). Kockázatos a hazatérés, segítséget kérek!",
            })
        if (mode in ("CLASSIC", "TDM") and self.last_attacker_id is not None
                and tick - self.last_attacked_tick <= 3 and tick - self.last_threat_msg_tick >= 25):
            # Veszélyjelzés: megtámadtak -> a hatótávon belüli társak megtudják, ki a támadó és hol van
            self.last_threat_msg_tick = tick
            atk = self.last_attacker_id
            atk_pos = self.last_attacker_pos if self.last_attacker_pos is not None else self.pos
            outgoing_specs.append(self._spec(
                "[WARNING]", None, "TEAM" if mode != "TDM" else f"TEAM-{self.team_id}", "THREAT_DETECTED", 3, atk_pos,
                self.assigned_sector, float(neighbor_energy),
                f"Veszély! R-{atk+1:02d} megtámadott ({int(atk_pos[0])}, {int(atk_pos[1])}), akkum {self.energy:.0f}%. Kerüljétek vagy lépjetek fel ellene!",
                key=f"R{atk}"))

        if tick - self.last_comm_tick >= comm_cooldown:
            if in_hazard and mode in ("ANTARCTIC", "COOP"):
                self.last_comm_tick = tick
                outgoing_specs.append({
                    "recipient_id": None,
                    "recipient_label": "TEAM",
                    "category": "[WARNING]",
                    "intent": "HAZARD_WARN",
                    "priority": 2,
                    "pos": self.pos.copy(),
                    "sector_id": self.assigned_sector,
                    "value": self.energy,
                    "text": f"Fagy-/veszélyzónát észleltem a(z) ({int(self.pos[0])}, {int(self.pos[1])}) koordinátánál! Kerüljétek el ezt a sávot.",
                })
            elif self._pending_status is not None:
                self.last_comm_tick = tick
                outgoing_specs.append(self._spec(
                    "[INFO]", None, "TEAM", "STATUS_UPDATE", 1, self.pos, self.assigned_sector, float(self.energy),
                    self._pending_status))
                self._pending_status = None
            elif visible_dirt_count >= 7 and self.genome.cooperation > 0.45 and mode in ("COOP", "CLASSIC"):
                self.last_comm_tick = tick
                rec_lbl = neighbor_name if (neighbor_id is not None and agent_dist < 32.0) else "TEAM"
                outgoing_specs.append({
                    "recipient_id": neighbor_id if agent_dist < 32.0 else None,
                    "recipient_label": rec_lbl,
                    "category": "[INFO]",
                    "intent": "DIRT_SPOT",
                    "priority": 1,
                    "pos": self.pos + dirt_vec * min(dirt_dist, 12.0),
                    "sector_id": self.assigned_sector,
                    "value": float(visible_dirt_count),
                    "text": f"A(z) #{self.assigned_sector+1}. szektorban {visible_dirt_count} szennyezett gócpontot találtam. Át tudja valaki venni a szélső részt?",
                })
            elif mode == "COOP" and tick % 45 == (self.id * 7) % 45 and not self.returned_to_base:
                self.last_comm_tick = tick
                outgoing_specs.append({
                    "recipient_id": neighbor_id if agent_dist < 28.0 else None,
                    "recipient_label": neighbor_name if agent_dist < 28.0 else "TEAM",
                    "category": "[TASK]",
                    "intent": "SECTOR_CLAIM",
                    "priority": 1,
                    "pos": self.pos.copy(),
                    "sector_id": self.assigned_sector,
                    "value": float(sector_dirt_count),
                    "text": f"Lefoglalom a(z) #{self.assigned_sector+1}. szektort ({sector_dirt_count} kosz-góc maradt, akkum {self.energy:.0f}%).",
                })
            elif mode == "TDM" and self.state == "HUNTING" and enemy_dist < 16.0:
                self.last_comm_tick = tick
                outgoing_specs.append({
                    "recipient_id": None,
                    "recipient_label": f"TEAM-{self.team_id}",
                    "category": "[COMBAT]",
                    "intent": "ENEMY_PING",
                    "priority": 2,
                    "pos": self.pos + enemy_vec * enemy_dist,
                    "sector_id": self.assigned_sector,
                    "value": enemy_energy,
                    "text": f"Ellenséges egység {enemy_dist:.0f}m-re (akkuja ~{enemy_energy:.0f}%)! Közös támadást indítok!",
                })

        # ---- Célpontmegosztás: saját időzítéssel, hogy a szektor-üzenetek ne éheztessék ki
        if (mode in ("COOP", "CLASSIC") and visible_dirt_count >= prof.share_threshold
                and dirt_dist <= self.SENSOR_RANGE and (self.genome.cooperation > 0.30 or prof.explore)
                and tick - self.last_share_tick >= max(10, comm_cooldown // 2)):
            spot_info = self.pos + dirt_vec * min(dirt_dist, 12.0)
            key = AgentMemory.target_key(spot_info)
            if (key not in self.memory.sent_offers
                    and not self.memory.is_claimed_by_other(spot_info, self.id)
                    and (self.commitment is None or self.commitment["key"] != key)):
                self.last_share_tick = tick
                self.last_comm_tick = max(self.last_comm_tick, tick - comm_cooldown // 2)
                self.memory.sent_offers[key] = tick
                cand = self._pick_task_candidate(spot_info, tick)
                if cand is not None:
                    outgoing_specs.append(self._spec(
                        "[TASK]", cand[0], cand[1], "TASK_ASSIGN", 2, spot_info, self.assigned_sector, float(visible_dirt_count),
                        f"Szemétgóc ({visible_dirt_count} db) a(z) ({int(spot_info[0])}, {int(spot_info[1])}) körzetben. {cand[1]}, átvennéd?",
                        key=key))
                else:
                    outgoing_specs.append(self._spec(
                        "[TASK]", None, "TEAM", "TASK_OFFER", 2, spot_info, self.assigned_sector, float(visible_dirt_count),
                        f"Szemétgóc ({visible_dirt_count} db) a(z) ({int(spot_info[0])}, {int(spot_info[1])}) körzetben. Ki veszi át?",
                        key=key))

        total_force = (
            desire_vec
            + wall_vec * self.genome.wall_avoidance
            + self.vel * 0.30
            + np.random.normal(0.0, 0.04, size=2).astype(np.float32)
        )

        norm = np.linalg.norm(total_force)
        if norm > 1e-5:
            desired_dir = total_force / norm
            desired_angle = math.atan2(float(desired_dir[1]), float(desired_dir[0]))
            self.angle = lerp_angle(self.angle, desired_angle, 0.42)
            self.vel = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)

        effective_speed = self.genome.speed * speed_multiplier * (0.78 if in_hazard else 1.0)
        new_pos = self.pos + self.vel * effective_speed
        new_pos[0] = np.clip(new_pos[0], 1.0, grid_width - 2.0)
        new_pos[1] = np.clip(new_pos[1], 1.0, grid_height - 2.0)

        cx, cy = int(new_pos[0]), int(new_pos[1])
        old_pos = self.pos.copy()
        if obstacle_grid[cx, cy] == 0:
            self.pos = new_pos
        else:
            if obstacle_grid[cx, int(old_pos[1])] == 0:
                self.pos = np.array([new_pos[0], old_pos[1]], dtype=np.float32)
            elif obstacle_grid[int(old_pos[0]), cy] == 0:
                self.pos = np.array([old_pos[0], new_pos[1]], dtype=np.float32)
            else:
                self.angle = (self.angle + math.pi * 0.85) % (2.0 * math.pi)
                self.vel = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)
                self.pos = np.clip(
                    self.pos + self.vel * 0.4,
                    [1.0, 1.0],
                    [grid_width - 2.0, grid_height - 2.0],
                )

        step_dist = float(np.linalg.norm(self.pos - old_pos))
        self.distance_traveled += step_dist

        if tick % 2 == 0:
            self.trail.append((float(self.pos[0]), float(self.pos[1])))
            if len(self.trail) > 40:
                self.trail.pop(0)

        hazard_mult = 1.65 if in_hazard else 1.0
        energy_cost = (0.10 + 0.052 * (effective_speed ** 2)) * hazard_mult
        if self.returned_to_base and mode == "COOP":
            energy_cost = 0.01

        self.energy -= energy_cost
        self.energy_used += energy_cost
        self.steps_alive += 1

        if self.energy <= 0.0:
            self.energy = 0.0
            self.alive = False
            self.state = "DISABLED"
            self.death_reason = (
                "Antarktiszi fagyzóna" if (mode == "ANTARCTIC" and in_hazard) else "Akkumulátor lemerült"
            )
            self.log_decision(tick, f"Kiesett: {self.death_reason}")

        return outgoing_specs
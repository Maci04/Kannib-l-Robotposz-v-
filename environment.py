import math
import csv
import json
import os
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from agent import RobotAgent, AgentMessage
import coordination


@dataclass
class EventLogEntry:
    """Rendszerszintű és fizikai eseménynapló-bejegyzés."""
    tick: int
    agent_name: str
    category: str  # "TAKARÍTÁS", "ENERGIA", "HARC", "KÜLDETÉS"
    message: str


def extract_merged_rectangles(grid: np.ndarray, target_val: int) -> List[Tuple[int, int, int, int]]:
    """
    Greedy Meshing (2D téglalap-összevonó) algoritmus a 3D-s Ursina pályákhoz.
    Az összefüggő azonos típusú rácscellákat (falakat, állványokat, padlózónákat)
    minimális számú (x0, y0, w, h) téglalappá vonja össze, így 5000+ kocka helyett
    mindössze ~40 optimalizált 3D-s objektumot kell renderelni!
    """
    w_grid, h_grid = grid.shape
    visited = np.zeros((w_grid, h_grid), dtype=bool)
    rects: List[Tuple[int, int, int, int]] = []

    for x in range(w_grid):
        for y in range(h_grid):
            if grid[x, y] == target_val and not visited[x, y]:
                rw = 1
                while x + rw < w_grid and grid[x + rw, y] == target_val and not visited[x + rw, y]:
                    rw += 1

                rh = 1
                can_expand = True
                while y + rh < h_grid and can_expand:
                    for ix in range(x, x + rw):
                        if grid[ix, y + rh] != target_val or visited[ix, y + rh]:
                            can_expand = False
                            break
                    if can_expand:
                        rh += 1

                visited[x : x + rw, y : y + rh] = True
                rects.append((x, y, rw, rh))

    return rects


class CommunicationBus:
    """Dedikált kommunikációs réteg a robotok közötti üzenetküldéshez és archiváláshoz."""

    def __init__(
        self,
        comm_range: float = 38.0,
        delivery_delay_ticks: int = 1,
        packet_loss_rate: float = 0.02,
    ):
        self.comm_range: float = comm_range
        self.delivery_delay_ticks: int = delivery_delay_ticks
        self.packet_loss_rate: float = packet_loss_rate
        self.enabled: bool = True

        self.next_seq: int = 1
        self.pending_queue: List[AgentMessage] = []
        self.chat_archive: List[AgentMessage] = []
        self.delivered_this_tick: List[AgentMessage] = []
        # --- spam-védelem: azonos üzenet ismétlésének tiltása időablakon belül + tickenkénti sávszélesség-korlát
        self._recent: Dict[tuple, int] = {}
        self._tick_counter: Tuple[int, int] = (-1, 0)
        self.max_new_per_tick: int = 8
        self.suppressed_duplicates: int = 0
        self.suppressed_rate: int = 0
        self.unreachable_count: int = 0

    DEDUPE_WINDOW: Dict[str, int] = {
        "TASK_ASSIGN": 30, "TASK_OFFER": 30, "TARGET_CLAIM": 20, "TARGET_RELEASE": 10, "DIRT_SPOT": 25,
        "LOW_BAT_SOS": 40, "THREAT_DETECTED": 20, "STATUS_UPDATE": 30,
    }

    def reset(self) -> None:
        self.next_seq = 1
        self.pending_queue.clear()
        self.chat_archive.clear()
        self.delivered_this_tick.clear()
        self._recent.clear()
        self._tick_counter = (-1, 0)
        self.suppressed_duplicates = 0
        self.suppressed_rate = 0
        self.unreachable_count = 0

    def history_for(self, agent_id: int, limit: int = 20) -> List[AgentMessage]:
        """Egy robot kommunikációs előzménye (küldött, neki címzett és általa feldolgozott üzenetek) - későbbi panelekhez."""
        out = [m for m in self.chat_archive
               if m.sender_id == agent_id or m.recipient_id == agent_id or agent_id in m.processed_by]
        return out[-limit:]

    def thread(self, thread_id: str) -> List[AgentMessage]:
        """Egy kérés-válasz lánc (többfordulós párbeszédekhez)."""
        return [m for m in self.chat_archive if m.thread_id == thread_id]

    def enqueue_from_spec(
        self, sender: RobotAgent, spec: Dict[str, Any], current_tick: int, rng: np.random.Generator
    ) -> Optional[AgentMessage]:
        # ---- duplikáció- és sávszélesség-szűrés (None = az üzenet nem kerül sorba)
        intent = spec.get("intent", "DIRT_SPOT")
        p = spec["pos"]
        tk = spec.get("target_key") or f"{int(p[0] // 6)}:{int(p[1] // 6)}:{spec.get('sector_id', 0)}"
        dkey = (sender.id, intent, spec.get("recipient_id"), tk, spec.get("ref_msg_id") or "")
        window = self.DEDUPE_WINDOW.get(intent, 15)
        last = self._recent.get(dkey)
        if last is not None and current_tick - last < window:
            self.suppressed_duplicates += 1
            return None
        t_now, n_now = self._tick_counter
        if t_now != current_tick:
            t_now, n_now = current_tick, 0
        if n_now >= self.max_new_per_tick and int(spec.get("priority", 1)) < 3:
            self.suppressed_rate += 1
            self._tick_counter = (t_now, n_now)
            return None
        self._tick_counter = (t_now, n_now + 1)
        self._recent[dkey] = current_tick
        if len(self._recent) > 4000:
            self._recent = {k: t for k, t in self._recent.items() if current_tick - t < 60}

        msg_id = f"MSG-{self.next_seq:04d}"
        self.next_seq += 1

        msg = AgentMessage(
            msg_id=msg_id,
            sender_id=sender.id,
            sender_name=sender.name,
            team_id=sender.team_id,
            recipient_id=spec.get("recipient_id", None),
            recipient_label=spec.get("recipient_label", "TEAM"),
            category=spec.get("category", "[INFO]"),
            intent=spec.get("intent", "DIRT_SPOT"),
            priority=int(spec.get("priority", 1)),
            sent_tick=current_tick,
            deliver_tick=current_tick + self.delivery_delay_ticks,
            ttl=30,
            pos=spec["pos"].copy(),
            sector_id=int(spec.get("sector_id", 0)),
            value=float(spec.get("value", 0.0)),
            text=spec["text"],
            ref_msg_id=spec.get("ref_msg_id", None),
            status="QUEUED",
            sender_role=getattr(sender, "role_id", ""),
            target_key=spec.get("target_key", ""),
            origin=sender.pos.copy(),
        )
        ref = spec.get("ref_msg_id")
        msg.thread_id = spec.get("thread_id") or (ref if ref else msg_id)

        sender.messages_sent += 1
        if rng.random() < self.packet_loss_rate:
            msg.status = "DROPPED"
            msg.reaction_note = "Rádiójel elveszett (interferencia)"
            self.chat_archive.append(msg)
        else:
            self.pending_queue.append(msg)
            self.chat_archive.append(msg)

        if len(self.chat_archive) > 600:
            self.chat_archive.pop(0)

        return msg

    def process_deliveries(
        self, current_tick: int, living_agents: List[RobotAgent], mode: str
    ) -> List[AgentMessage]:
        self.delivered_this_tick.clear()
        if not self.enabled:
            return []

        remaining_queue: List[AgentMessage] = []
        for msg in self.pending_queue:
            if current_tick - msg.sent_tick > msg.ttl:
                msg.status = "EXPIRED"
                continue

            if current_tick < msg.deliver_tick:
                remaining_queue.append(msg)
                continue

            delivered_count = 0
            addressee_seen = msg.recipient_id is None
            origin = msg.origin if msg.origin is not None else msg.pos
            for receiver in living_agents:
                if receiver.id == msg.sender_id or not receiver.alive:
                    continue
                if mode == "TDM" and receiver.team_id != msg.team_id:
                    continue
                if msg.recipient_id is not None and receiver.id != msg.recipient_id:
                    continue
                addressee_seen = True
                dist = float(np.linalg.norm(receiver.pos - origin))
                if dist <= self.comm_range:
                    receiver.receive_message(msg)
                    delivered_count += 1

            if delivered_count > 0:
                msg.status = "DELIVERED"
                msg.delivered_tick = current_tick
                self.delivered_this_tick.append(msg)
            else:
                msg.status = "EXPIRED"
                if not addressee_seen:
                    msg.reaction_note = "Címzett elérhetetlen (kiesett / nincs a csapatban)"
                    self.unreachable_count += 1
                else:
                    msg.reaction_note = "Hatótávon kívül"

        self.pending_queue = remaining_queue
        return self.delivered_this_tick

    def export_chat_log(self, mode: str, generation: int) -> str:
        if not self.chat_archive:
            return "A Chat Log még üres!"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"chatlog_{mode.lower()}_gen{generation}_{stamp}"
        json_path = os.path.abspath(f"{base_name}.json")
        csv_path = os.path.abspath(f"{base_name}.csv")

        records = []
        for m in self.chat_archive:
            records.append({
                "msg_id": m.msg_id,
                "sent_tick": m.sent_tick,
                "deliver_tick": m.deliver_tick,
                "sender": m.sender_name,
                "recipient": m.recipient_label,
                "team_id": m.team_id,
                "category": m.category,
                "intent": m.intent,
                "priority": m.priority,
                "sector_id": m.sector_id + 1,
                "status": m.status,
                "reaction": m.reaction_note,
                "text": m.text,
            })

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"mode": mode, "generation": generation, "messages": records}, jf, ensure_ascii=False, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        return f"Chat mentve: {os.path.basename(json_path)} & .csv"


MAP_CATALOG: Dict[str, Dict[str, Any]] = {
    "COMPACT": {
        "name": "1. Compact Room",
        "difficulty": "Könnyű *--",
        "desc": "Kisebb, átlátható tesztcsarnok központi tartóoszlopokkal és töltődokkokkal.",
        "theme": "STANDARD",
        "floor_tex": "floor_concrete.png",
        "wall_tex": "wall_industrial.png",
        "rack_tex": "rack_metal.png",
    },
    "WAREHOUSE": {
        "name": "2. Large Warehouse",
        "difficulty": "Közepes **-",
        "desc": "Hatalmas logisztikai raktárcsarnok magas acél polcsorokkal és szűk folyosókkal.",
        "theme": "INDUSTRIAL",
        "floor_tex": "floor_concrete.png",
        "wall_tex": "wall_industrial.png",
        "rack_tex": "rack_metal.png",
    },
    "APARTMENT": {
        "name": "3. Complex Apartment",
        "difficulty": "Nehéz ***",
        "desc": "Több helyiségből, ajtókeretekből, szőnyegekből és bútorokból álló lakótér.",
        "theme": "APARTMENT",
        "floor_tex": "floor_parquet.png",
        "wall_tex": "wall_brick.png",
        "rack_tex": "rack_metal.png",
    },
    "ANTARCTIC": {
        "name": "4. Antarctic Research Facility",
        "difficulty": "Extrém ***",
        "desc": "Jeges kutatóbázis A indulási és B célállomással, töltődokk nélkül, fagyzónákkal.",
        "theme": "ANTARCTIC",
        "floor_tex": "floor_ice.png",
        "wall_tex": "wall_industrial.png",
        "rack_tex": "rack_metal.png",
    },
    "INDUSTRIAL": {
        "name": "5. Industrial Facility",
        "difficulty": "Nehéz ***",
        "desc": "6 szektorra osztott ipari üzem veszélyes sav-/hőzónákkal és központi bázisdokkal.",
        "theme": "INDUSTRIAL",
        "floor_tex": "floor_industrial.png",
        "wall_tex": "wall_hazard.png",
        "rack_tex": "rack_metal.png",
    },
    "ARENA": {
        "name": "6. Tactical Combat Arena",
        "difficulty": "Közepes **-",
        "desc": "Szimmetrikus taktikai harci aréna fedezékekkel, energia-gócpontokkal és csapatbázisokkal.",
        "theme": "ARENA",
        "floor_tex": "floor_arena.png",
        "wall_tex": "wall_hazard.png",
        "rack_tex": "rack_metal.png",
    },
}


MAX_WORLD_SCALE: int = 5   # 5 x 5 csempe = a sablon 25-szörös területe


class SimulationEnvironment:
    """
    Optimalizált szimulációs környezet előre kiszámított taszító vektormezőkkel
    és 3D-s Greedy-Mesh geometria-exportálással.
    """

    def __init__(
        self,
        width: int = 88,
        height: int = 60,
        initial_dirt: int = 130,
        dirt_spawn_rate: int = 2,
        map_key: str = "COMPACT",
        mode: str = "CLASSIC",
        seed: int = 42,
        world_scale: int = 1,
    ):
        # A `width`/`height` a pálya-SABLON mérete; a világ ennek `world_scale` x `world_scale`-szoros
        # csempézése (lineáris skála; a terület world_scale^2-szeresére nő).
        self.world_scale: int = max(1, min(MAX_WORLD_SCALE, int(world_scale)))
        self.tile_w: int = width
        self.tile_h: int = height
        self.width: int = width * self.world_scale
        self.height: int = height * self.world_scale
        area_factor = self.world_scale ** 2
        self.initial_dirt: int = initial_dirt * area_factor
        self.dirt_spawn_rate: int = dirt_spawn_rate * area_factor if dirt_spawn_rate > 0 else 0
        self.map_key: str = map_key if map_key in MAP_CATALOG else "COMPACT"
        self.mode: str = mode
        self.seed: int = seed
        self.rng = np.random.default_rng(seed)

        self.dirt_grid: np.ndarray = np.zeros((self.width, self.height), dtype=np.int8)
        self.obstacle_grid: np.ndarray = np.zeros((self.width, self.height), dtype=np.int8)
        self.floor_grid: np.ndarray = np.zeros((self.width, self.height), dtype=np.int8)
        self.hazard_grid: np.ndarray = np.zeros((self.width, self.height), dtype=np.int8)

        self.wall_field: np.ndarray = np.zeros((self.width, self.height, 2), dtype=np.float32)
        self.hazard_field: np.ndarray = np.zeros((self.width, self.height, 2), dtype=np.float32)

        self.chargers: List[np.ndarray] = []
        self.base_pos: np.ndarray = np.array([10.0, self.height / 2.0], dtype=np.float32)
        self.goal_pos: np.ndarray = np.array([self.width - 10.0, self.height / 2.0], dtype=np.float32)

        # A rádió hatótávja a világgal együtt (mérsékelten) nő, hogy nagy pályán se legyenek teljesen elszigeteltek a robotok
        self.comm_bus: CommunicationBus = CommunicationBus(comm_range=min(80.0, 38.0 * (1.0 + 0.25 * (self.world_scale - 1))))
        self._template: Optional["SimulationEnvironment"] = None
        if self.world_scale > 1:
            self._template = SimulationEnvironment(
                width=self.tile_w, height=self.tile_h, initial_dirt=0, dirt_spawn_rate=0,
                map_key=self.map_key, mode=self.mode, seed=seed, world_scale=1,
            )

        self.combat_events: List[Tuple[np.ndarray, np.ndarray]] = []
        self.energy_share_events: List[Tuple[np.ndarray, np.ndarray]] = []
        self.comm_events: List[AgentMessage] = []
        self.charge_events: List[int] = []
        self.role_events: List[str] = []
        self._dead_notified: set = set()
        self.event_log: List[EventLogEntry] = []

        self.tick_count: int = 0
        self.step_kills: int = 0
        self.total_dirt_spawned: int = 0
        self.total_dirt_cleaned: int = 0
        self.total_energy_transfers: int = 0
        self.total_energy_shared_amount: float = 0.0
        self.total_messages: int = 0
        self.last_step_ai_decisions: int = 0

        self._build_map(self.map_key)

    @property
    def communication_enabled(self) -> bool:
        return self.comm_bus.enabled

    @communication_enabled.setter
    def communication_enabled(self, val: bool) -> None:
        self.comm_bus.enabled = val

    def add_log(self, tick: int, agent_name: str, category: str, message: str) -> None:
        self.event_log.append(EventLogEntry(tick=tick, agent_name=agent_name, category=category, message=message))
        if len(self.event_log) > 250:
            self.event_log.pop(0)

    @staticmethod
    def _repulsion_field(mask: np.ndarray, radius: int, power: float, min_dist: float) -> np.ndarray:
        """
        Taszító vektormező az `mask > 0` cellák körül (a cella-középpontok közti (diff / dist^power)
        összeg, normalizálva). Egyenértékű a korábbi cellánkénti Python-ciklussal, de a (2r+1)^2
        eltolás összegzésével vektorizált, ezért a 25-szörös terület is gyorsan számolható.
        """
        w, h = mask.shape
        occ = (mask > 0).astype(np.float32)
        padded = np.pad(occ, radius)
        fx = np.zeros((w, h), dtype=np.float32)
        fy = np.zeros((w, h), dtype=np.float32)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = max(math.hypot(dx, dy), min_dist)
                # a (x, y) cellához képest a (x+dx, y+dy) akadály: diff = pos - coords = (-dx, -dy)
                shifted = padded[radius + dx: radius + dx + w, radius + dy: radius + dy + h]
                k = 1.0 / (dist ** (power + 1.0)) if power else 1.0 / dist
                fx -= dx * k * shifted
                fy -= dy * k * shifted
        norm = np.sqrt(fx * fx + fy * fy)
        out = np.zeros((w, h, 2), dtype=np.float32)
        ok = norm > 1e-5
        out[..., 0][ok] = fx[ok] / norm[ok]
        out[..., 1][ok] = fy[ok] / norm[ok]
        return out

    def _precompute_vector_fields(self) -> None:
        # fal-taszítás: 7x7 ablak, 1/d^3 erő, min. távolság 0.3; veszélyzóna: 9x9 ablak, 1/d^2 erő, min. 0.5
        self.wall_field = self._repulsion_field(self.obstacle_grid, 3, 2.0, 0.3)
        self.hazard_field = self._repulsion_field(self.hazard_grid, 4, 1.0, 0.5)

    def _precompute_vector_fields_reference(self) -> None:
        """A korábbi, lassú, cellánkénti referencia-implementáció (csak tesztelésre)."""
        self.wall_field = np.zeros((self.width, self.height, 2), dtype=np.float32)
        self.hazard_field = np.zeros((self.width, self.height, 2), dtype=np.float32)
        for x in range(self.width):
            for y in range(self.height):
                pos = np.array([x + 0.5, y + 0.5], dtype=np.float32)
                x0, x1 = max(0, x - 3), min(self.width, x + 4)
                y0, y1 = max(0, y - 3), min(self.height, y + 4)
                sub_obs = self.obstacle_grid[x0:x1, y0:y1]
                obs_idx = np.argwhere(sub_obs > 0)
                if len(obs_idx) > 0:
                    coords = obs_idx + np.array([x0, y0], dtype=np.float32) + 0.5
                    diffs = pos - coords
                    dists = np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 0.3)
                    forces = (diffs / (dists ** 3)).sum(axis=0)
                    norm = float(np.linalg.norm(forces))
                    if norm > 1e-5:
                        self.wall_field[x, y] = forces / norm
                hx0, hx1 = max(0, x - 4), min(self.width, x + 5)
                hy0, hy1 = max(0, y - 4), min(self.height, y + 5)
                sub_hz = self.hazard_grid[hx0:hx1, hy0:hy1]
                hz_idx = np.argwhere(sub_hz > 0)
                if len(hz_idx) > 0:
                    coords = hz_idx + np.array([hx0, hy0], dtype=np.float32) + 0.5
                    diffs = pos - coords
                    dists = np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 0.5)
                    forces = (diffs / (dists ** 2)).sum(axis=0)
                    norm = float(np.linalg.norm(forces))
                    if norm > 1e-5:
                        self.hazard_field[x, y] = forces / norm

    def _build_map(self, map_key: str) -> None:
        if self.world_scale > 1 and self._template is not None:
            self._assemble_tiled_world()
            return
        self._build_template_map(map_key)

    # ------------------------------------------------------------------ nagy pályák (csempézés)
    @staticmethod
    def _tile_row(template_y: float, tile_h: int, k: int) -> int:
        """Melyik csempe-sorba kerül a bázis/cél: a sablonbeli függőleges helyzete szerint."""
        if template_y < tile_h * 0.35:
            return 0
        if template_y > tile_h * 0.65:
            return k - 1
        return k // 2

    def _assemble_tiled_world(self) -> None:
        """
        A `world_scale` x `world_scale` csempéből álló nagy világ építése a sablonpályából:
          * a csempék a szomszédos sorban/oszlopban tükrözve ismétlődnek (változatosabb elrendezés),
          * a csempék közti falakon ajtónyílások vannak (3 / csempe-oldal), így a világ összefüggő,
          * a bázis a bal oldali, a cél a jobb oldali csempében van, a töltők minden csempében,
          * a bázis-/cél-padló csak a saját csempéjében marad.
        """
        t = self._template
        assert t is not None
        k, tw, th = self.world_scale, self.tile_w, self.tile_h
        self.obstacle_grid.fill(0)
        self.floor_grid.fill(0)
        self.hazard_grid.fill(0)
        self.chargers = []

        base_row = self._tile_row(float(t.base_pos[1]), th, k)
        goal_row = self._tile_row(float(t.goal_pos[1]), th, k)

        for i in range(k):
            for j in range(k):
                is_base = (i == 0 and j == base_row)
                is_goal = (i == k - 1 and j == goal_row)
                flip_x = (i % 2 == 1) and not is_goal
                flip_y = (j % 2 == 1) and not (is_base or is_goal)
                obs, flo, haz = t.obstacle_grid, t.floor_grid.copy(), t.hazard_grid
                if flip_x:
                    obs, flo, haz = obs[::-1, :], flo[::-1, :], haz[::-1, :]
                if flip_y:
                    obs, flo, haz = obs[:, ::-1], flo[:, ::-1], haz[:, ::-1]
                flo = flo.copy()
                if not is_base:
                    flo[flo == 1] = 0
                if not is_goal:
                    flo[flo == 2] = 0
                x0, y0 = i * tw, j * th
                self.obstacle_grid[x0:x0 + tw, y0:y0 + th] = obs
                self.floor_grid[x0:x0 + tw, y0:y0 + th] = flo
                self.hazard_grid[x0:x0 + tw, y0:y0 + th] = haz
                for c in t.chargers:
                    cx = (tw - c[0]) if flip_x else c[0]
                    cy = (th - c[1]) if flip_y else c[1]
                    self.chargers.append(np.array([x0 + cx, y0 + cy], dtype=np.float32))

        # ajtónyílások a csempék közötti (dupla) falakon; az ajtó körüli 8x8-as zónát szabaddá tesszük
        for s in range(1, k):
            sx, sy = s * tw, s * th
            for t_idx in range(k):
                for frac in (0.25, 0.5, 0.75):
                    yc = int(t_idx * th + th * frac)
                    self.obstacle_grid[sx - 4: sx + 4, yc - 4: yc + 4] = 0
                    self.hazard_grid[sx - 4: sx + 4, yc - 4: yc + 4] = 0
                    xc = int(t_idx * tw + tw * frac)
                    self.obstacle_grid[xc - 4: xc + 4, sy - 4: sy + 4] = 0
                    self.hazard_grid[xc - 4: xc + 4, sy - 4: sy + 4] = 0

        # a külső keret mindig fal
        self.obstacle_grid[0, :] = 1
        self.obstacle_grid[-1, :] = 1
        self.obstacle_grid[:, 0] = 1
        self.obstacle_grid[:, -1] = 1

        self.base_pos = np.array([t.base_pos[0], base_row * th + t.base_pos[1]], dtype=np.float32)
        self.goal_pos = np.array([(k - 1) * tw + t.goal_pos[0], goal_row * th + t.goal_pos[1]], dtype=np.float32)
        if self.mode == "ANTARCTIC":
            self.chargers = []
        self._precompute_vector_fields()

    def _build_template_map(self, map_key: str) -> None:
        self.obstacle_grid.fill(0)
        self.floor_grid.fill(0)
        self.hazard_grid.fill(0)
        self.chargers.clear()

        w, h = self.width, self.height
        self.obstacle_grid[0, :] = 1
        self.obstacle_grid[-1, :] = 1
        self.obstacle_grid[:, 0] = 1
        self.obstacle_grid[:, -1] = 1

        if map_key == "COMPACT":
            pillars = [
                (w // 4, h // 4, 4, 8),
                (3 * w // 4 - 4, h // 4, 4, 8),
                (w // 4, 3 * h // 4 - 8, 4, 8),
                (3 * w // 4 - 4, 3 * h // 4 - 8, 4, 8),
                (w // 2 - 6, h // 2 - 2, 12, 4),
            ]
            for px, py, pw, ph in pillars:
                self.obstacle_grid[px : px + pw, py : py + ph] = 1

            self.base_pos = np.array([10.0, h / 2.0], dtype=np.float32)
            self.goal_pos = np.array([w - 10.0, h / 2.0], dtype=np.float32)
            self.floor_grid[4:16, h // 2 - 8 : h // 2 + 8] = 1
            self.chargers = [
                np.array([10.0, 10.0], dtype=np.float32),
                np.array([w - 10.0, h - 10.0], dtype=np.float32),
            ]

        elif map_key == "WAREHOUSE":
            for col_x in range(18, w - 16, 14):
                self.obstacle_grid[col_x : col_x + 3, 7 : h // 2 - 4] = 2
                self.obstacle_grid[col_x : col_x + 3, h // 2 + 4 : h - 7] = 2
            self.base_pos = np.array([8.0, h / 2.0], dtype=np.float32)
            self.goal_pos = np.array([w - 8.0, h / 2.0], dtype=np.float32)
            self.floor_grid[2:14, h // 2 - 10 : h // 2 + 10] = 1
            self.floor_grid[w - 14 : w - 2, h // 2 - 10 : h // 2 + 10] = 2
            self.chargers = [
                np.array([8.0, 8.0], dtype=np.float32),
                np.array([8.0, h - 8.0], dtype=np.float32),
            ]

        elif map_key == "APARTMENT":
            mid_x, mid_y = w // 2, h // 2
            self.obstacle_grid[mid_x - 1 : mid_x + 1, 1 : h - 1] = 1
            self.obstacle_grid[1 : w - 1, mid_y - 1 : mid_y + 1] = 1

            self.obstacle_grid[mid_x - 1 : mid_x + 1, mid_y - 13 : mid_y - 6] = 0
            self.obstacle_grid[mid_x - 1 : mid_x + 1, mid_y + 6 : mid_y + 13] = 0
            self.obstacle_grid[mid_x - 18 : mid_x - 11, mid_y - 1 : mid_y + 1] = 0
            self.obstacle_grid[mid_x + 11 : mid_x + 18, mid_y - 1 : mid_y + 1] = 0

            self.floor_grid[6 : mid_x - 4, 6 : mid_y - 4] = 3
            self.floor_grid[mid_x + 4 : w - 6, mid_y + 4 : h - 6] = 3
            self.obstacle_grid[14:22, 13:17] = 2
            self.obstacle_grid[w - 24 : w - 14, 13:18] = 2
            self.obstacle_grid[16:24, h - 17 : h - 12] = 2

            self.base_pos = np.array([9.0, h - 9.0], dtype=np.float32)
            self.goal_pos = np.array([w - 9.0, 9.0], dtype=np.float32)
            self.floor_grid[3:15, h - 15 : h - 3] = 1
            self.chargers = [np.array([9.0, h - 9.0], dtype=np.float32)]

        elif map_key == "ANTARCTIC":
            self.base_pos = np.array([9.0, h / 2.0], dtype=np.float32)
            self.goal_pos = np.array([w - 9.0, h / 2.0], dtype=np.float32)
            self.floor_grid[2:15, h // 2 - 10 : h // 2 + 10] = 1
            self.floor_grid[w - 15 : w - 2, h // 2 - 10 : h // 2 + 10] = 2

            self.obstacle_grid[24:27, 1 : h // 2 - 6] = 1
            self.obstacle_grid[24:27, h // 2 + 6 : h - 1] = 1

            self.obstacle_grid[44:47, 9 : h - 9] = 1
            self.obstacle_grid[44:47, h // 2 - 4 : h // 2 + 4] = 0

            self.obstacle_grid[64:67, 1 : h // 2 - 8] = 1
            self.obstacle_grid[64:67, h // 2 + 8 : h - 1] = 1

            self.hazard_grid[28:43, h // 2 - 9 : h // 2 + 9] = 1
            self.hazard_grid[48:63, h // 2 - 7 : h // 2 + 7] = 1
            self.hazard_grid[34:56, 3:8] = 1
            self.hazard_grid[34:56, h - 8 : h - 3] = 1
            self.chargers = []

        elif map_key == "INDUSTRIAL":
            for sx in [w // 3, 2 * w // 3]:
                self.obstacle_grid[sx - 1 : sx + 1, 5 : h - 5] = 1
                self.obstacle_grid[sx - 1 : sx + 1, h // 4 - 3 : h // 4 + 3] = 0
                self.obstacle_grid[sx - 1 : sx + 1, 3 * h // 4 - 3 : 3 * h // 4 + 3] = 0

            self.obstacle_grid[10 : w - 10, h // 2 - 1 : h // 2 + 1] = 2
            self.obstacle_grid[w // 6 - 3 : w // 6 + 3, h // 2 - 1 : h // 2 + 1] = 0
            self.obstacle_grid[w // 2 - 4 : w // 2 + 4, h // 2 - 1 : h // 2 + 1] = 0
            self.obstacle_grid[5 * w // 6 - 3 : 5 * w // 6 + 3, h // 2 - 1 : h // 2 + 1] = 0

            self.hazard_grid[w // 3 + 4 : 2 * w // 3 - 4, h // 2 - 6 : h // 2 + 6] = 1
            self.base_pos = np.array([10.0, h / 2.0], dtype=np.float32)
            self.goal_pos = np.array([w - 10.0, h / 2.0], dtype=np.float32)
            self.floor_grid[2:15, h // 2 - 9 : h // 2 + 9] = 1
            self.chargers = [np.array([10.0, h / 2.0], dtype=np.float32)]

        elif map_key == "ARENA":
            cx, cy = w // 2, h // 2
            self.obstacle_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = 2
            for dx, dy in [(-20, -13), (16, -13), (-20, 9), (16, 9)]:
                self.obstacle_grid[cx + dx : cx + dx + 5, cy + dy : cy + dy + 5] = 1

            self.base_pos = np.array([10.0, cy], dtype=np.float32)
            self.goal_pos = np.array([w - 10.0, cy], dtype=np.float32)
            self.floor_grid[2:14, cy - 11 : cy + 11] = 1
            self.floor_grid[w - 14 : w - 2, cy - 11 : cy + 11] = 2
            self.chargers = [
                np.array([cx, 8.0], dtype=np.float32),
                np.array([cx, h - 8.0], dtype=np.float32),
            ]

        if self.mode == "ANTARCTIC":
            self.chargers = []

        self._precompute_vector_fields()

    def reset(self, agents: List[RobotAgent]) -> None:
        self._build_map(self.map_key)
        self.dirt_grid.fill(0)
        self.combat_events.clear()
        self.energy_share_events.clear()
        self.comm_events.clear()
        self.charge_events.clear()
        self.role_events.clear()
        self._dead_notified.clear()
        self.event_log.clear()
        self.comm_bus.reset()

        self.tick_count = 0
        self.step_kills = 0
        self.total_dirt_spawned = 0
        self.total_dirt_cleaned = 0
        self.total_energy_transfers = 0
        self.total_energy_shared_amount = 0.0
        self.total_messages = 0
        self.last_step_ai_decisions = 0

        initial_dirt = self.initial_dirt
        if self.mode == "ANTARCTIC":
            initial_dirt = max(35, self.initial_dirt // 3)
        elif self.mode == "COOP":
            initial_dirt = max(155, int(self.initial_dirt * 1.3))

        self.spawn_dirt(initial_dirt)

        free_coords = np.argwhere(self.obstacle_grid == 0)
        for i, agent in enumerate(agents):
            agent.energy = agent.MAX_ENERGY
            agent.alive = True
            agent.steps_alive = 0
            agent.dirt_cleaned = 0
            agent.kills = 0
            agent.damage_dealt = 0.0
            agent.distance_traveled = 0.0
            agent.energy_used = 0.0
            agent.energy_given = 0.0
            agent.energy_received = 0.0
            agent.messages_sent = 0
            agent.messages_received = 0
            agent.assists = 0
            agent.ai_decisions_count = 0
            agent.reached_goal = False
            agent.returned_to_base = False
            agent.mission_aborted = False
            agent.death_reason = "Aktív"
            agent.state = "CLEANING"
            agent.trail.clear()
            agent.inbox.clear()
            agent.recent_comm_links.clear()
            agent.memory = type(agent.memory)()
            agent.known_partners.clear()
            agent.reset_multiagent_state()
            agent.assigned_sector = i % 6
            agent.decision_log = [f"[T:0] Küldetés indult ({self.mode})."]

            if self.mode in ("ANTARCTIC", "COOP"):
                start_zone = np.argwhere((self.obstacle_grid == 0) & (self.floor_grid == 1))
                if len(start_zone) > 0:
                    idx = self.rng.integers(0, len(start_zone))
                    agent.pos = start_zone[idx].astype(np.float32) + 0.5
                else:
                    agent.pos = self.base_pos + self.rng.uniform(-4.0, 4.0, size=2).astype(np.float32)
            elif self.mode == "TDM":
                if agent.team_id == 1:
                    zone = np.argwhere((self.obstacle_grid == 0) & (np.arange(self.width)[:, None] < 16))
                elif agent.team_id == 2:
                    zone = np.argwhere((self.obstacle_grid == 0) & (np.arange(self.width)[:, None] > self.width - 17))
                else:
                    zone = np.argwhere(
                        (self.obstacle_grid == 0)
                        & (np.arange(self.width)[:, None] > self.width // 3)
                        & (np.arange(self.width)[:, None] < 2 * self.width // 3)
                    )
                if len(zone) > 0:
                    idx = self.rng.integers(0, len(zone))
                    agent.pos = zone[idx].astype(np.float32) + 0.5
                else:
                    idx = self.rng.integers(0, len(free_coords))
                    agent.pos = free_coords[idx].astype(np.float32) + 0.5
            else:
                idx = self.rng.integers(0, len(free_coords))
                agent.pos = free_coords[idx].astype(np.float32) + 0.5

            agent.snap_render_state()

        coordination.assign_roles(agents, self.mode, tick=0)       # stabil szerepek a genomból + csapatösszetétel
        counts: Dict[str, int] = {}
        for a in agents:
            counts[a.role_id] = counts.get(a.role_id, 0) + 1
        self.add_log(0, "RENDSZER", "SZEREPEK", "Szerepek: " + ", ".join(
            f"{coordination.ROLE_LABEL[r]} x{n}" for r, n in sorted(counts.items())))
        self.add_log(0, "RENDSZER", "KÜLDETÉS", f"{self.mode} elindítva a(z) {MAP_CATALOG[self.map_key]['name']} 3D pályán.")

    def spawn_dirt(self, count: int) -> None:
        if count <= 0:
            return
        free_mask = (self.obstacle_grid == 0) & (self.dirt_grid == 0) & (self.floor_grid != 1)
        free_coords = np.argwhere(free_mask)
        if len(free_coords) == 0:
            return

        actual_count = min(count, len(free_coords))
        chosen_idx = self.rng.choice(len(free_coords), size=actual_count, replace=False)
        chosen = free_coords[chosen_idx]
        intensities = self.rng.choice([1, 2, 3], size=actual_count, p=[0.60, 0.30, 0.10]).astype(np.int8)
        self.dirt_grid[chosen[:, 0], chosen[:, 1]] = intensities
        self.total_dirt_spawned += int(intensities.sum())

    def _get_sector_info(self, sector_id: int) -> Tuple[np.ndarray, int]:
        col = sector_id % 3
        row = sector_id // 3
        sw = self.width // 3
        sh = self.height // 2
        x0, x1 = col * sw, min(self.width, (col + 1) * sw)
        y0, y1 = row * sh, min(self.height, (row + 1) * sh)
        center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
        dirt_count = int((self.dirt_grid[x0:x1, y0:y1] > 0).sum())
        return center, dirt_count

    def step(self, agents: List[RobotAgent]) -> Tuple[int, int]:
        self.tick_count += 1
        self.combat_events.clear()
        self.energy_share_events.clear()
        self.comm_events.clear()
        self.charge_events.clear()
        self.last_step_ai_decisions = 0

        for change in coordination.rebalance_roles(agents, self.mode, self.tick_count):   # ritka, indokolt szerepcsere
            self.add_log(self.tick_count, "KOORDINÁTOR", "SZEREP", change)

        living_agents = [a for a in agents if a.alive]
        num_alive = len(living_agents)

        if num_alive == 0:
            return 0, int((self.dirt_grid > 0).sum())

        delivered = self.comm_bus.process_deliveries(self.tick_count, living_agents, self.mode)
        self.comm_events.extend(delivered)

        positions = np.array([a.pos for a in living_agents], dtype=np.float32)
        diff_agents = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_agents = np.linalg.norm(diff_agents, axis=-1)
        np.fill_diagonal(dist_agents, np.inf)

        charger_arr = np.array(self.chargers, dtype=np.float32).reshape(-1, 2)
        dirt_coords = np.argwhere(self.dirt_grid > 0).astype(np.float32) + 0.5
        sector_cache = {s: self._get_sector_info(s) for s in range(6)}

        for i, agent in enumerate(living_agents):
            s_range = agent.SENSOR_RANGE
            visible_dirt_count = 0
            dirt_dist = 999.0
            dirt_vec = np.zeros(2, dtype=np.float32)

            if len(dirt_coords) > 0:
                d_vecs = dirt_coords - agent.pos
                d_dists = np.linalg.norm(d_vecs, axis=1)
                vis_mask = d_dists <= s_range
                visible_dirt_count = int(vis_mask.sum())
                if visible_dirt_count > 0:
                    min_d_idx = int(np.argmin(d_dists))
                    dirt_dist = float(d_dists[min_d_idx])
                    dirt_vec = d_vecs[min_d_idx] / max(dirt_dist, 1e-5)

            agent_dist, neighbor_energy = 999.0, 0.0
            agent_vec = np.zeros(2, dtype=np.float32)
            neighbor_id: Optional[int] = None
            neighbor_name: str = "TEAM"

            enemy_dist, enemy_energy = 999.0, 0.0
            enemy_vec = np.zeros(2, dtype=np.float32)

            ally_low_dist = 999.0
            ally_low_vec = np.zeros(2, dtype=np.float32)
            ally_low_name = "TEAM"
            ally_low_id: Optional[int] = None

            if num_alive > 1:
                min_a_idx = int(np.argmin(dist_agents[i]))
                raw_a_dist = float(dist_agents[i, min_a_idx])
                if raw_a_dist <= s_range:
                    agent_dist = raw_a_dist
                    other_nb = living_agents[min_a_idx]
                    agent_vec = (other_nb.pos - agent.pos) / max(agent_dist, 1e-5)
                    neighbor_energy = other_nb.energy
                    neighbor_id = other_nb.id
                    neighbor_name = other_nb.name

                for j, other in enumerate(living_agents):
                    if i == j:
                        continue
                    d_ij = float(dist_agents[i, j])
                    if d_ij > s_range:
                        continue

                    is_enemy = (self.mode != "TDM") or (other.team_id != agent.team_id)
                    if is_enemy and d_ij < enemy_dist:
                        enemy_dist = d_ij
                        enemy_vec = (other.pos - agent.pos) / max(d_ij, 1e-5)
                        enemy_energy = other.energy

                    is_ally = (self.mode in ("COOP", "ANTARCTIC")) or (
                        self.mode == "TDM" and other.team_id == agent.team_id
                    )
                    if is_ally and other.energy_ratio < 0.35 and not other.reached_goal and d_ij < ally_low_dist:
                        ally_low_dist = d_ij
                        ally_low_vec = (other.pos - agent.pos) / max(d_ij, 1e-5)
                        ally_low_name = other.name
                        ally_low_id = other.id

            charger_dist = 999.0
            charger_vec = np.zeros(2, dtype=np.float32)
            if len(self.chargers) > 0:
                c_dists = np.linalg.norm(charger_arr - agent.pos, axis=1)      # vektorizált (sok töltő nagy pályán)
                min_c = int(np.argmin(c_dists))
                charger_dist = float(c_dists[min_c])
                charger_vec = (charger_arr[min_c] - agent.pos) / max(charger_dist, 1e-5)

            sec_center, sec_dirt = sector_cache[agent.assigned_sector % 6]
            cx = int(np.clip(agent.pos[0], 0, self.width - 1))
            cy = int(np.clip(agent.pos[1], 0, self.height - 1))

            sensor_data: Dict[str, Any] = {
                "dirt_vec": dirt_vec,
                "dirt_dist": dirt_dist,
                "visible_dirt_count": visible_dirt_count,
                "agent_vec": agent_vec,
                "agent_dist": agent_dist,
                "agent_energy": neighbor_energy,
                "neighbor_id": neighbor_id,
                "neighbor_name": neighbor_name,
                "enemy_vec": enemy_vec,
                "enemy_dist": enemy_dist,
                "enemy_energy": enemy_energy,
                "ally_low_vec": ally_low_vec,
                "ally_low_dist": ally_low_dist,
                "ally_low_name": ally_low_name,
                "ally_low_id": ally_low_id,
                "charger_vec": charger_vec,
                "charger_dist": charger_dist,
                "base_pos": self.base_pos,
                "goal_pos": self.goal_pos,
                "sector_center": sec_center,
                "sector_dirt_count": sec_dirt,
                "wall_vec": self.wall_field[cx, cy],
                "hazard_vec": self.hazard_field[cx, cy],
            }

            was_alive = agent.alive
            was_at_goal = agent.reached_goal
            outgoing_specs = agent.decide_and_move(
                sensor_data,
                self.width,
                self.height,
                self.obstacle_grid,
                mode=self.mode,
                tick=self.tick_count,
                hazard_grid=self.hazard_grid,
            )
            self.last_step_ai_decisions += 1

            if was_alive and not agent.alive:
                self.add_log(self.tick_count, agent.name, "ENERGIA", f"Kiesett ({agent.death_reason}).")
            if not was_at_goal and agent.reached_goal:
                self.add_log(self.tick_count, agent.name, "KÜLDETÉS", "Elérte a B kutatóállomást!")

            if self.comm_bus.enabled:
                for spec in outgoing_specs:
                    if self.comm_bus.enqueue_from_spec(agent, spec, self.tick_count, self.rng) is not None:
                        self.total_messages += 1

        for agent in living_agents:
            if not agent.alive:
                continue
            cx, cy = int(agent.pos[0]), int(agent.pos[1])
            x0, x1 = max(1, cx - 1), min(self.width - 1, cx + 2)
            y0, y1 = max(1, cy - 1), min(self.height - 1, cy + 2)

            patch = self.dirt_grid[x0:x1, y0:y1]
            cleaned_amount = int(patch.sum())
            if cleaned_amount > 0:
                self.dirt_grid[x0:x1, y0:y1] = 0
                agent.dirt_cleaned += cleaned_amount
                self.total_dirt_cleaned += cleaned_amount
                energy_gain = cleaned_amount * (11.0 if self.mode != "COOP" else 8.5)
                agent.energy = min(agent.MAX_ENERGY, agent.energy + energy_gain)
                if cleaned_amount >= 4:
                    self.add_log(
                        self.tick_count,
                        agent.name,
                        "TAKARÍTÁS",
                        f"Sűrű szennyeződési gócot tisztított meg (+{cleaned_amount} egység).",
                    )

            if len(charger_arr) > 0 and agent.energy < agent.MAX_ENERGY:
                near = np.linalg.norm(charger_arr - agent.pos, axis=1) < 2.8
                for _ in range(int(near.sum())):          # minden közeli töltő külön tölt (eredeti viselkedés)
                    agent.energy = min(agent.MAX_ENERGY, agent.energy + 1.8)
                    if agent.id not in self.charge_events:
                        self.charge_events.append(agent.id)

        self._resolve_energy_sharing(living_agents)

        if self.mode in ("CLASSIC", "TDM"):
            self._resolve_combat(living_agents)

        # kiesett robotok: minden élő társ törli a rájuk mutató hivatkozásokat (nincs érvénytelen címzett / foglalás)
        for a in agents:
            if not a.alive and a.id not in self._dead_notified:
                self._dead_notified.add(a.id)
                for b in agents:
                    if b.alive:
                        b.forget_peer(a.id, self.tick_count)

        if self.mode == "CLASSIC":
            self.spawn_dirt(self.dirt_spawn_rate)
        elif self.mode == "TDM" and self.tick_count % 2 == 0:
            self.spawn_dirt(max(1, self.dirt_spawn_rate // 2))
        elif self.mode == "COOP" and self.tick_count % 8 == 0 and self.dirt_spawn_rate > 0:
            self.spawn_dirt(1)

        still_alive = sum(1 for a in agents if a.alive)
        return still_alive, int(self.dirt_grid.sum())

    def _resolve_energy_sharing(self, living_agents: List[RobotAgent]) -> None:
        n = len(living_agents)
        if n < 2:
            return

        share_range = 3.4
        for i in range(n):
            donor = living_agents[i]
            if not donor.alive or donor.energy_ratio < 0.55 or donor.genome.cooperation < 0.38:
                continue

            for j in range(n):
                if i == j:
                    continue
                receiver = living_agents[j]
                if not receiver.alive or receiver.energy_ratio > 0.38 or receiver.reached_goal:
                    continue
                if self.mode == "TDM" and donor.team_id != receiver.team_id:
                    continue

                dist = float(np.linalg.norm(donor.pos - receiver.pos))
                if dist <= share_range:
                    transfer = min(13.0, donor.energy - 40.0, receiver.MAX_ENERGY - receiver.energy)
                    if transfer > 2.0:
                        donor.energy -= transfer
                        received = transfer * 0.85
                        receiver.energy = min(receiver.MAX_ENERGY, receiver.energy + received)
                        donor.energy_given += transfer
                        receiver.energy_received += received
                        donor.assists += 1
                        self.total_energy_transfers += 1
                        self.total_energy_shared_amount += received
                        self.energy_share_events.append((donor.pos.copy(), receiver.pos.copy()))
                        self.add_log(
                            self.tick_count,
                            donor.name,
                            "ENERGIA",
                            f"Átadott {received:.1f} energiát {receiver.name} részére.",
                        )
                        break

    def _resolve_combat(self, living_agents: List[RobotAgent]) -> None:
        n = len(living_agents)
        if n < 2:
            return

        positions = np.array([a.pos for a in living_agents], dtype=np.float32)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        attack_range = 2.4
        for i in range(n):
            attacker = living_agents[i]
            if not attacker.alive or attacker.state != "HUNTING":
                continue

            targets_idx = np.where((dists[i] < attack_range) & (dists[i] > 0.0))[0]
            for j in targets_idx:
                victim = living_agents[j]
                if not victim.alive:
                    continue
                if self.mode == "TDM" and attacker.team_id == victim.team_id:
                    continue

                drain_amount = min(victim.energy, 13.0 + 16.0 * attacker.genome.aggressiveness)
                victim.energy -= drain_amount
                net_gain = max(0.0, drain_amount * 0.72 - 2.0)
                attacker.energy = min(attacker.MAX_ENERGY, attacker.energy + net_gain)
                attacker.damage_dealt += drain_amount

                self.combat_events.append((attacker.pos.copy(), victim.pos.copy()))
                victim.last_attacker_id = attacker.id            # a megtámadott tudja, ki támadta (veszélyjelzéshez)
                victim.last_attacked_tick = self.tick_count
                victim.last_attacker_pos = attacker.pos.copy()

                if victim.energy <= 0.0:
                    victim.energy = 0.0
                    victim.alive = False
                    victim.state = "DISABLED"
                    victim.death_reason = f"Harci treff ({attacker.name})"
                    attacker.kills += 1
                    self.step_kills += 1
                    self.add_log(
                        self.tick_count,
                        attacker.name,
                        "HARC",
                        f"Lemerítette és kiütötte {victim.name} egységet!",
                    )
                break
import csv
import os
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional
from agent import RobotAgent, Genome


class EvolutionaryAlgorithm:
    """
    Populáció-menedzsment, küldetésspecifikus fitnesz kiértékelés, szelekció és CSV exportálás.
    Minden játékmód külön statisztikai idősort kap.
    """

    def __init__(
        self,
        pop_size: int = 20,
        mutation_rate: float = 0.15,
        swap_rate: float = 0.06,
        elite_ratio: float = 0.16,
        survival_weight: float = 1.0,
        dirt_weight: float = 25.0,
        mode: str = "CLASSIC",
        num_teams: int = 2,
        start_energy: float = 100.0,
    ):
        self.pop_size: int = pop_size
        self.mutation_rate: float = mutation_rate
        self.swap_rate: float = swap_rate
        self.elite_ratio: float = elite_ratio
        self.survival_weight: float = survival_weight
        self.dirt_weight: float = dirt_weight
        self.mode: str = mode
        self.num_teams: int = num_teams
        self.start_energy: float = start_energy

        self.generation: int = 1
        self.agents: List[RobotAgent] = self._create_initial_agents()

        self.history: Dict[str, List[float]] = self._empty_history()
        self.histories_by_mode: Dict[str, Dict[str, List[float]]] = {
            "CLASSIC": self._empty_history(),
            "ANTARCTIC": self._empty_history(),
            "COOP": self._empty_history(),
            "TDM": self._empty_history(),
        }

    def _empty_history(self) -> Dict[str, List[float]]:
        return {
            "generation": [],
            "max_fitness": [],
            "avg_fitness": [],
            "survival_rate": [],
            "return_or_goal_rate": [],
            "avg_aggressiveness": [],
            "avg_risk_tolerance": [],
            "avg_cooperation": [],
            "avg_speed": [],
            "total_kills": [],
            "total_dirt": [],
            "total_energy_shared": [],
            "total_messages": [],
        }

    def _create_initial_agents(self, genomes: Optional[List[Genome]] = None) -> List[RobotAgent]:
        agents: List[RobotAgent] = []
        team_prefixes = {0: "R", 1: "ALP", 2: "BRV", 3: "GAM"}
        for i in range(self.pop_size):
            team_id = (i % self.num_teams) + 1 if self.mode == "TDM" else 0
            prefix = team_prefixes.get(team_id, "R")
            genome = genomes[i] if (genomes and i < len(genomes)) else Genome.random()
            agents.append(
                RobotAgent(
                    agent_id=i,
                    pos=np.zeros(2, dtype=np.float32),
                    genome=genome,
                    team_id=team_id,
                    name=f"{prefix}-{i + 1:02d}",
                    max_energy=self.start_energy,
                )
            )
        return agents

    def compute_fitness(
        self, agent: RobotAgent, goal_pos: Optional[np.ndarray] = None
    ) -> float:
        active_factor = min(1.0, agent.distance_traveled / 25.0)

        if self.mode == "ANTARCTIC":
            goal_bonus = 450.0 if agent.reached_goal else 0.0
            remaining_energy_bonus = agent.energy * 2.2 if agent.reached_goal else 0.0
            dist_penalty = 0.0
            if goal_pos is not None and not agent.reached_goal:
                dist_to_goal = float(np.linalg.norm(goal_pos - agent.pos))
                dist_penalty = max(0.0, 220.0 - dist_to_goal * 2.4)
            coop_bonus = agent.energy_given * 3.5 + agent.assists * 25.0
            survival_bonus = (60.0 if agent.alive else 0.0) * active_factor
            return float(goal_bonus + remaining_energy_bonus + dist_penalty + coop_bonus + survival_bonus)

        elif self.mode == "COOP":
            clean_score = agent.dirt_cleaned * 30.0
            base_bonus = 240.0 if (agent.alive and agent.returned_to_base and agent.dirt_cleaned > 0) else 0.0
            alive_bonus = 55.0 if agent.alive else 0.0
            coop_score = agent.energy_given * 3.2 + min(agent.messages_received + agent.messages_sent, 16) * 4.5
            return float(clean_score + base_bonus + alive_bonus + coop_score)

        elif self.mode == "TDM":
            combat_score = agent.damage_dealt * 1.6 + agent.kills * 95.0
            team_support = agent.energy_given * 2.8 + agent.dirt_cleaned * 12.0
            survival_score = agent.steps_alive * 0.7 * active_factor + (80.0 if agent.alive else 0.0)
            return float(combat_score + team_support + survival_score)

        else:
            return float(
                agent.steps_alive * self.survival_weight * max(0.3, active_factor)
                + agent.dirt_cleaned * self.dirt_weight
            )

    def _tournament_selection(
        self, agents: List[RobotAgent], fitnesses: np.ndarray, k: int = 3
    ) -> Genome:
        indices = np.random.choice(len(agents), size=min(k, len(agents)), replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return agents[best_idx].genome

    def evolve_next_generation(
        self, goal_pos: Optional[np.ndarray] = None
    ) -> List[RobotAgent]:
        fitnesses = np.array(
            [self.compute_fitness(a, goal_pos=goal_pos) for a in self.agents],
            dtype=np.float32,
        )

        n = max(1, len(self.agents))
        surv_rate = sum(1 for a in self.agents if a.alive) / n * 100.0
        if self.mode == "ANTARCTIC":
            goal_rate = sum(1 for a in self.agents if a.reached_goal) / n * 100.0
        elif self.mode == "COOP":
            goal_rate = sum(1 for a in self.agents if a.alive and a.returned_to_base) / n * 100.0
        else:
            goal_rate = surv_rate

        record = {
            "generation": self.generation,
            "max_fitness": float(np.max(fitnesses)),
            "avg_fitness": float(np.mean(fitnesses)),
            "survival_rate": float(surv_rate),
            "return_or_goal_rate": float(goal_rate),
            "avg_aggressiveness": float(np.mean([a.genome.aggressiveness for a in self.agents])),
            "avg_risk_tolerance": float(np.mean([a.genome.risk_tolerance for a in self.agents])),
            "avg_cooperation": float(np.mean([a.genome.cooperation for a in self.agents])),
            "avg_speed": float(np.mean([a.genome.speed for a in self.agents])),
            "total_kills": int(sum(a.kills for a in self.agents)),
            "total_dirt": int(sum(a.dirt_cleaned for a in self.agents)),
            "total_energy_shared": float(sum(a.energy_given for a in self.agents)),
            "total_messages": int(sum(a.messages_sent for a in self.agents)),
        }

        for key, val in record.items():
            self.history[key].append(val)
            self.histories_by_mode[self.mode][key].append(val)

        sorted_indices = np.argsort(fitnesses)[::-1]
        num_elites = max(2, int(self.pop_size * self.elite_ratio))

        new_genomes: List[Genome] = []
        for i in range(num_elites):
            new_genomes.append(Genome(genes=self.agents[sorted_indices[i]].genome.genes.copy()))

        while len(new_genomes) < self.pop_size:
            parent1 = self._tournament_selection(self.agents, fitnesses, k=3)
            parent2 = self._tournament_selection(self.agents, fitnesses, k=3)

            child_genome = Genome.crossover(parent1, parent2)
            child_genome.mutate_shift(mutation_rate=self.mutation_rate, sigma=0.12)
            child_genome.mutate_swap(swap_rate=self.swap_rate)
            new_genomes.append(child_genome)

        self.agents = self._create_initial_agents(genomes=new_genomes)
        self.generation += 1
        return self.agents

    def export_history_to_csv(self, filename: Optional[str] = None) -> str:
        if len(self.history["generation"]) == 0:
            return "Nincs menthető generációs adat (futtass le legalább 1 generációt)!"

        if filename is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stats_{self.mode.lower()}_{stamp}.csv"

        keys = list(self.history.keys())
        rows = zip(*(self.history[k] for k in keys))

        filepath = os.path.abspath(filename)
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)

        return f"Statisztika CSV mentve: {os.path.basename(filepath)}"
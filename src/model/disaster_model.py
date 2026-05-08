import mesa
import numpy as np

from src.agents.astar_drone import AStarDrone
from src.agents.pheromone_drone import PheromoneDrone
from src.agents.random_drone import RandomDrone
from src.environment.grid import CellType, DisasterGrid

HAZARD_RATES: dict[str, float] = {
    "slow": 0.05,
    "medium": 0.20,
    "fast": 0.40,
}

_STRATEGY_MAP: dict[str, type] = {
    "random": RandomDrone,
    "astar": AStarDrone,
    "pheromone": PheromoneDrone,
}


class DisasterModel(mesa.Model):
    """
    Central coordinator for the disaster simulation.

    Attributes:
        - strategy: Search strategy name ('random', 'astar', 'pheromone').
        - swarm_size: Number of drone agents.
        - hazard_rate: Named hazard level ('slow', 'medium', 'fast').
        - noise_prob: Probability each in-range survivor is missed (false negative).
        - disaster_grid: The DisasterGrid environment.
        - pheromone_grid: Shared pheromone values updated by PheromoneDrone.
        - coverage_grid: Per-cell visit counts across all agents.
        - agents_lost: Count of agents removed due to fire.
        - survivors_found_count: Count of survivors found so far.
        - timestep: Current simulation step number.
    """

    def __init__(
        self,
        strategy: str,
        swarm_size: int,
        hazard_rate: str,
        seed: int,
        width: int = 20,
        height: int = 20,
        noise_prob: float = 0.0,
    ) -> None:
        """
        Initialises the disaster model.

        Args:
            - strategy: Search strategy ('random', 'astar', 'pheromone').
            - swarm_size: Number of drone agents to deploy.
            - hazard_rate: Fire spread rate ('slow', 'medium', 'fast').
            - seed: Random seed for reproducibility.
            - width: Grid width in cells.
            - height: Grid height in cells.
            - noise_prob: Per-survivor miss probability in sensing (0.0 = perfect, 1.0 = blind).
        """
        super().__init__(rng=seed)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.strategy = strategy
        self.swarm_size = swarm_size
        self.hazard_rate = hazard_rate
        self.noise_prob = noise_prob
        self._hazard_p = HAZARD_RATES[hazard_rate]

        self.disaster_grid = DisasterGrid(width, height, seed)
        self.disaster_grid.place_obstacles()
        self.disaster_grid.place_survivors()
        self.disaster_grid.place_fire_seeds()

        self.pheromone_grid = np.zeros((width, height))
        self.coverage_grid = np.zeros((width, height), dtype=int)

        self.agents_lost: int = 0
        self.survivors_found_count: int = 0
        self.timestep: int = 0

        agent_class = _STRATEGY_MAP[strategy]
        for _ in range(swarm_size):
            agent = agent_class(self)
            self.disaster_grid.grid.place_agent(agent, (0, 0))

    @property
    def is_done(self) -> bool:
        """
        Returns True when the simulation run has reached a terminal state.

        Returns:
            - True if all survivors found, all agents dead, or timestep >= 200.
        """
        total_survivors = len(self.disaster_grid.survivors)
        all_found = self.survivors_found_count >= total_survivors
        all_dead = self.swarm_size > 0 and len(list(self.agents)) == 0
        timed_out = self.timestep >= 200
        return all_found or all_dead or timed_out

    @property
    def termination_reason(self) -> str:
        """
        Returns the reason the simulation ended, or empty string if running.

        Returns:
            - 'survivors', 'agents_dead', 'timeout', or '' if not done.
        """
        if not self.is_done:
            return ""
        total_survivors = len(self.disaster_grid.survivors)
        if self.survivors_found_count >= total_survivors:
            return "survivors"
        if self.swarm_size > 0 and len(list(self.agents)) == 0:
            return "agents_dead"
        return "timeout"

    def evaporate_pheromones(self) -> None:
        """
        Decays all pheromone values by the evaporation factor each step.
        """
        self.pheromone_grid *= 0.95

    def check_agent_deaths(self) -> None:
        """
        Remove agents on fire cells and increment agents_lost.
        """
        to_remove = [
            agent
            for agent in list(self.agents)
            if agent.pos is not None
            and self.disaster_grid.grid_state[agent.pos] == CellType.FIRE
        ]
        for agent in to_remove:
            self.disaster_grid.grid.remove_agent(agent)
            agent.remove()
            self.agents_lost += 1

    def step(self) -> None:
        """
        Advance the simulation by one timestep.
        """
        self.agents.shuffle_do("step")
        self.disaster_grid.spread_fire(self._hazard_p)
        self.check_agent_deaths()
        self.evaporate_pheromones()
        self.timestep += 1

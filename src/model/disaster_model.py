import mesa
import numpy as np

from src.environment.grid import CellType, DisasterGrid

HAZARD_RATES: dict[str, float] = {
    "slow": 0.05,
    "medium": 0.20,
    "fast": 0.40,
}


class DisasterModel(mesa.Model):
    """
    Central coordinator for the disaster simulation.

    Attributes:
        - strategy: Search strategy name ('random', 'astar', 'pheromone').
        - swarm_size: Number of drone agents.
        - hazard_rate: Named hazard level ('slow', 'medium', 'fast').
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
        """
        super().__init__(rng=seed)
        self.rng = np.random.default_rng(seed)
        self.strategy = strategy
        self.swarm_size = swarm_size
        self.hazard_rate = hazard_rate
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

    def check_agent_deaths(self) -> None:
        """
        Remove any agents whose current cell is on fire and increment agents_lost.
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
        self.timestep += 1

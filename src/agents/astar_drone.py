import mesa
import numpy as np

from src.agents.base_drone import DroneAgent
from src.environment.grid import CellType
from src.strategies.astar import astar, get_nearest_frontier


class AStarDrone(DroneAgent):
    """
    Drone that uses A* pathfinding to navigate to the nearest frontier.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
        - current_path: Planned path to current target.
        - current_target: Current frontier target position.
    """

    def __init__(self, model: mesa.Model) -> None:
        """
        Initialises the A* drone.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.current_path: list[tuple[int, int]] = []
        self.current_target: tuple[int, int] | None = None

    def _replan(self) -> None:
        grid_state = self.model.disaster_grid.grid_state
        passable_mask = grid_state != CellType.OBSTACLE.value
        fire_mask = grid_state == CellType.FIRE.value

        frontier = get_nearest_frontier(self.pos, self.visited_cells, grid_state)
        if frontier is None:
            self.current_path = []
            self.current_target = None
            return

        path = astar(passable_mask, fire_mask, self.pos, frontier)
        if path is None:
            self.current_path = []
            self.current_target = None
            return

        self.current_path = path
        self.current_target = frontier

    def step(self) -> None:
        """
        Navigates to the nearest frontier via A*; falls back to random if no frontier exists.
        """
        grid_state = self.model.disaster_grid.grid_state

        if not self.current_path or grid_state[self.current_path[0]] == CellType.FIRE.value:
            self._replan()

        if self.current_path:
            next_pos = self.current_path.pop(0)
            self.move_to(next_pos)
        else:
            neighbours = self.get_passable_neighbours(self.pos)
            if neighbours:
                new_pos = tuple(int(c) for c in self.model.rng.choice(neighbours))
                self.move_to(new_pos)

        self.detect_survivors()

import math

from src.agents.base_drone import DroneAgent
from src.environment.grid import CellType, Survivor
from src.strategies.astar import astar


class MedicDrone(DroneAgent):
    """
    MedicDrone: A drone specialised in rescuing survivors by navigating to them and marking them as found.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
        - sensing_radius: 2.
        - target: The Survivor exclusively assigned to this medic, or None.
        - _current_path: A* path steps remaining to the current target.
        - _idle_step: Counts idle steps for pheromone fallback cadence.
    """

    def __init__(self, model) -> None:
        """
        Initialises the medic drone with sensing_radius=2.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.sensing_radius = 2
        self.target: Survivor | None = None
        self._current_path: list[tuple[int, int]] = []
        self._idle_step: int = 0

    def _assigned_survivor_ids(self) -> set[int]:
        """
        Return id() of survivors currently assigned to other medics.
        """
        assigned: set[int] = set()
        for agent in self.model.agents:
            if (
                agent is not self
                and isinstance(agent, MedicDrone)
                and agent.target is not None
            ):
                assigned.add(id(agent.target))
        return assigned

    def _assign_nearest(self) -> None:
        """
        Assign self to the nearest unassigned survivor in rescue_queue.
        """
        assigned_ids = self._assigned_survivor_ids()
        best: Survivor | None = None
        best_dist = math.inf
        x, y = self.pos
        for survivor in self.model.rescue_queue:
            if id(survivor) in assigned_ids:
                continue
            sx, sy = survivor.pos
            dist = abs(x - sx) + abs(y - sy)
            if dist < best_dist:
                best_dist = dist
                best = survivor
        if best is not None:
            self.target = best
            self._replan()

    def _replan(self) -> None:
        """
        Compute A* path to the current target, treating fire as impassable.
        """
        if self.target is None:
            self._current_path = []
            return
        grid_state = self.model.disaster_grid.grid_state
        passable = grid_state != CellType.OBSTACLE
        fire = grid_state == CellType.FIRE
        path = astar(passable, fire, self.pos, self.target.pos)
        self._current_path = path if path is not None else []

    def detect_survivors(self) -> None:
        """
        Detects survivors within sensing_radius and marks them as found.
        """
        noise = self.model.survivor_detection_noise
        x, y = self.pos
        grid_state = self.model.disaster_grid.grid_state
        for survivor in self.model.disaster_grid.survivors:
            if survivor.found:
                continue
            sx, sy = survivor.pos
            if grid_state[sx, sy] == CellType.FIRE:
                continue
            if abs(x - sx) + abs(y - sy) <= self.sensing_radius:
                if noise > 0.0 and self.model.rng.random() < noise:
                    continue
                survivor.found = True
                self.model.survivors_found_count += 1
                if survivor in self.model.rescue_queue:
                    self.model.rescue_queue.remove(survivor)
                if self.target is survivor:
                    self.target = None
                    self._current_path = []

    def _pheromone_move(self) -> None:
        """
        Execute one pheromone-based movement step.
        """
        self.model.pheromone_grid[self.pos] += 1.0
        neighbours = self.get_perceived_neighbours(self.pos)
        if neighbours:
            min_val = min(self.model.pheromone_grid[n] for n in neighbours)
            candidates = [
                n for n in neighbours if self.model.pheromone_grid[n] == min_val
            ]
            chosen = candidates[self.model.rng.integers(0, len(candidates))]
            self.move_to(chosen)

    def step(self) -> None:
        """
        Executes one simulation step for the medic drone.
        """
        if (
            self.target is not None
            and self.target not in self.model.rescue_queue
        ):
            self.target = None
            self._current_path = []

        if self.target is None and self.model.rescue_queue:
            self._assign_nearest()

        if self.target is not None:
            if self.pos != self.target.pos:
                if not self._current_path or (
                    self.model.disaster_grid.grid_state[self._current_path[0]]
                    == CellType.FIRE
                ):
                    self._replan()

                if self._current_path:
                    next_step = self._current_path.pop(0)
                    self.move_to(next_step)
                else:
                    self.target = None
        else:
            self._idle_step += 1
            if self._idle_step % 2 == 0:
                self._pheromone_move()

        self.detect_survivors()

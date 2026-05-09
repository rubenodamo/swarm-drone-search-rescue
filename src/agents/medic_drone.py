import math

from src.agents.base_drone import DroneAgent
from src.environment.grid import CellType, Survivor
from src.strategies.astar import astar


class MedicDrone(DroneAgent):
    """
    Rescue specialist that navigates to detected survivors and rescues them.

    Assigned exclusively to the nearest unassigned queued survivor via A*.
    Falls back to pheromone movement every 2 steps when the rescue queue
    is empty.

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
        """Return id() of survivors currently assigned to other medics."""
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

    def _rescue_target(self) -> None:
        """
        Rescue the assigned survivor: mark found, increment count, clear queue entry.
        """
        if self.target is None:
            return
        self.target.found = True
        self.model.survivors_found_count += 1
        if self.target in self.model.rescue_queue:
            self.model.rescue_queue.remove(self.target)
        self.target = None
        self._current_path = []

    def _pheromone_move(self) -> None:
        """Execute one pheromone-based movement step."""
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
        Executes one simulation step.

        Priority order:
        1. If assigned target left queue (fire), clear assignment.
        2. Assign to nearest queued survivor if idle.
        3. Navigate to target; rescue on arrival; reassign if unreachable.
        4. Fallback: pheromone move every 2 idle steps.
        """
        if self.target is not None and self.target not in self.model.rescue_queue:
            self.target = None
            self._current_path = []

        if self.target is None and self.model.rescue_queue:
            self._assign_nearest()

        if self.target is not None:
            if self.pos == self.target.pos:
                self._rescue_target()
                return

            if not self._current_path or (
                self.model.disaster_grid.grid_state[self._current_path[0]]
                == CellType.FIRE
            ):
                self._replan()

            if self._current_path:
                next_step = self._current_path.pop(0)
                self.move_to(next_step)
                if self.target is not None and self.pos == self.target.pos:
                    self._rescue_target()
            else:
                self.target = None
        else:
            self._idle_step += 1
            if self._idle_step % 2 == 0:
                self._pheromone_move()

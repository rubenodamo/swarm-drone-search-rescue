import mesa

from src.environment.grid import CellType


class DroneAgent(mesa.Agent):
    """
    Base class for all drone agents in the swarm.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
        - sensing_radius: Manhattan distance radius for survivor detection and observation.
        - _survivor_bonus: Count added to survivors_found_count per survivor found.
    """

    def __init__(self, model: mesa.Model) -> None:
        """
        Initialises the drone agent.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.alive: bool = True
        self.visited_cells: set[tuple[int, int]] = set()
        self.sensing_radius: int = 2
        self._survivor_bonus: int = 1

    def get_passable_neighbours(
        self, pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Returns all passable, non-fire cardinal neighbours of a position.

        Args:
            - pos: The (x, y) position to query neighbours for.

        Returns:
            - List of (x, y) positions that are PASSABLE.
        """
        return self.model.disaster_grid.passable_neighbours(pos)

    def get_local_observation(self) -> dict[tuple[int, int], int]:
        """
        Returns cell states within sensing_radius of current position.

        Returns:
            - Dict mapping (x, y) positions to their CellType integer values.
        """
        x, y = self.pos
        width = self.model.disaster_grid.width
        height = self.model.disaster_grid.height
        r = self.sensing_radius
        observation: dict[tuple[int, int], int] = {}
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        observation[(nx, ny)] = int(
                            self.model.disaster_grid.grid_state[nx, ny]
                        )
        return observation

    def mark_visited(self, pos: tuple[int, int]) -> None:
        """
        Records a cell as visited and updates the shared coverage grid.

        Args:
            - pos: The (x, y) position to mark as visited.
        """
        self.visited_cells.add(pos)
        self.model.coverage_grid[pos] += 1

    def get_perceived_neighbours(
        self, pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Returns neighbours the drone believes are safe to move to.

        Args:
            - pos: The (x, y) position to query neighbours for.

        Returns:
            - List of (x, y) positions perceived as passable.
        """
        result = list(self.get_passable_neighbours(pos))
        hazard_noise = self.model.hazard_detection_noise
        if hazard_noise == 0.0:
            return result

        x, y = pos
        width = self.model.disaster_grid.width
        height = self.model.disaster_grid.height
        grid_state = self.model.disaster_grid.grid_state
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid_state[nx, ny] == CellType.FIRE:
                    if self.model.rng.random() < hazard_noise:
                        result.append((nx, ny))
        return result

    def detect_survivors(self) -> None:
        """
        Finds and marks as found any survivors within sensing_radius.
        """
        noise = self.model.survivor_detection_noise
        x, y = self.pos
        for survivor in self.model.disaster_grid.survivors:
            if survivor.found:
                continue
            sx, sy = survivor.pos
            if abs(x - sx) + abs(y - sy) <= self.sensing_radius:
                if noise > 0.0 and self.model.rng.random() < noise:
                    continue
                survivor.found = True
                self.model.survivors_found_count += self._survivor_bonus

    def move_to(self, new_pos: tuple[int, int]) -> None:
        """
        Moves the agent to a new position, raising only if the target is an obstacle.

        Args:
            - new_pos: The target (x, y) position.
        """
        if self.model.disaster_grid.grid_state[new_pos] == CellType.OBSTACLE:
            raise ValueError(f"Cannot move to {new_pos}: cell is an obstacle")
        self.model.disaster_grid.grid.move_agent(self, new_pos)
        self.mark_visited(new_pos)

    def step(self) -> None:
        """
        Executes one simulation step. Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

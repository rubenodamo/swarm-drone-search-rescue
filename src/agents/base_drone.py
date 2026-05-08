import mesa

from src.environment.grid import CellType


class DroneAgent(mesa.Agent):
    """
    Base class for all drone agents in the swarm.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
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
        Returns cell states within Manhattan distance 2 of current position.

        Returns:
            - Dict mapping (x, y) positions to their CellType integer values.
        """
        x, y = self.pos
        width = self.model.disaster_grid.width
        height = self.model.disaster_grid.height
        observation: dict[tuple[int, int], int] = {}
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
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

    def detect_survivors(self) -> None:
        """
        Finds and marks as found any survivors within Manhattan distance 2.

        Each in-range survivor is missed with probability model.noise_prob
        (false negative). At noise_prob=0.0 every in-range survivor is found;
        at noise_prob=1.0 none are ever detected.
        """
        noise_prob = self.model.noise_prob
        x, y = self.pos
        for survivor in self.model.disaster_grid.survivors:
            if survivor.found:
                continue
            sx, sy = survivor.pos
            if abs(x - sx) + abs(y - sy) <= 2:
                if noise_prob > 0.0 and self.model.rng.random() < noise_prob:
                    continue
                survivor.found = True
                self.model.survivors_found_count += 1

    def move_to(self, new_pos: tuple[int, int]) -> None:
        """
        Moves the agent to a new position after validating it is passable.

        Args:
            - new_pos: The target (x, y) position.
        """
        cell_type = self.model.disaster_grid.grid_state[new_pos]
        if cell_type in (CellType.OBSTACLE, CellType.FIRE):
            raise ValueError(
                f"Cannot move to {new_pos}: cell type is "
                f"{CellType(cell_type).name}"
            )
        self.model.disaster_grid.grid.move_agent(self, new_pos)
        self.mark_visited(new_pos)

    def step(self) -> None:
        """
        Executes one simulation step. Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

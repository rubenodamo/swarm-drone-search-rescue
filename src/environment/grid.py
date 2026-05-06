from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from mesa.space import MultiGrid


class CellType(IntEnum):
    """
    Integer labels used to represent the state of each cell in the grid:

    Attributes:
        - PASSABLE: Open cell that agents can move through.
        - OBSTACLE: Impassable cell representing walls or debris.
        - FIRE: Cell currently on fire, which agents must avoid.
    """

    PASSABLE = 0
    OBSTACLE = 1
    FIRE = 2


@dataclass
class Survivor:
    """
    Represents a survivor in the disaster scenario.

    Attributes:
        - pos: The (x, y) coordinates of the survivor on the grid.
        - found: Whether the survivor has been located by an agent.
    """

    pos: tuple[int, int]
    found: bool = False


class DisasterGrid:
    """
    Represents the environment grid for the disaster scenario.

    Attributes:
        - width: The width of the grid.
        - height: The height of the grid.
        - grid: The Mesa MultiGrid representing the spatial environment.
        - grid_state: A 2D numpy array storing each cell's CellType.
        - survivors: A list of Survivor objects placed in the environment.
    """

    MAX_FIRE_FRACTION = 0.60

    def __init__(self, width: int, height: int, seed: int) -> None:
        """
        Initialises a new disaster grid.

        Args:
            - width: The width of the grid.
            - height: The height of the grid.
            - seed: Random seed for reproducible obstacle/survivor/fire
              placement.
        """

        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.grid = MultiGrid(width, height, torus=False)
        self.grid_state = np.zeros((width, height), dtype=int)
        self.survivors: list[Survivor] = []

    def place_obstacles(self, n: int = 60) -> None:
        """
        Randomly place obstacle cells in the environment.

        Args:
            - n: The number of obstacle cells to place.
        """
        all_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) != (0, 0)
        ]

        indices = self.rng.choice(len(all_cells), size=n, replace=False)

        for i in indices:
            x, y = all_cells[i]
            self.grid_state[x, y] = CellType.OBSTACLE

    def place_survivors(self, n: int = 10) -> None:
        """
        Randomly place survivors on passable cells.

        Args:
            - n: The number of survivors to place.
        """
        passable = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid_state[x, y] == CellType.PASSABLE and (x, y) != (0, 0)
        ]

        indices = self.rng.choice(len(passable), size=n, replace=False)

        self.survivors = [Survivor(pos=passable[i]) for i in indices]

    def place_fire_seeds(self, n: int = 3) -> None:
        """
        Randomly place initial fire cells in the environment.

        Args:
            - n: Number of initial fire cells to place.
        """
        candidates = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid_state[x, y] == CellType.PASSABLE
            and abs(x) + abs(y) >= 4
        ]

        indices = self.rng.choice(len(candidates), size=n, replace=False)

        for i in indices:
            x, y = candidates[i]
            self.grid_state[x, y] = CellType.FIRE

    def spread_fire(self, p: float) -> None:
        """
        Spread fire to adjacent passable cells with probability p.

        Args:
            - p: Probability that fire spreads to each adjacent passable
              cell.
        """
        passable_count = (self.grid_state != CellType.OBSTACLE).sum()
        fire_count = (self.grid_state == CellType.FIRE).sum()
        if (
            passable_count > 0
            and fire_count / passable_count >= self.MAX_FIRE_FRACTION
        ):
            return

        fire_positions = list(zip(*np.where(self.grid_state == CellType.FIRE)))
        new_fires: list[tuple[int, int]] = []

        for fx, fy in fire_positions:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid_state[nx, ny] == CellType.PASSABLE:
                        if self.rng.random() < p:
                            new_fires.append((nx, ny))

        for x, y in new_fires:
            self.grid_state[x, y] = CellType.FIRE

    def passable_neighbours(
        self, pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Return all passable (non-fire, non-obstacle) cardinal neighbours.

        Args:
            - pos: The (x, y) grid coordinates to query from.

        Returns:
            - List of (x, y) coordinates of passable neighbouring cells.
        """
        x, y = pos
        result = []

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid_state[nx, ny] == CellType.PASSABLE:
                    result.append((nx, ny))

        return result

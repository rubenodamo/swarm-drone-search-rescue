import numpy as np
import pytest

from src.strategies.astar import astar, get_nearest_frontier


def _make_masks(
    width: int, height: int, fire_cells: list[tuple[int, int]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    passable = np.ones((width, height), dtype=bool)
    fire = np.zeros((width, height), dtype=bool)
    if fire_cells:
        for x, y in fire_cells:
            fire[x, y] = True
            passable[x, y] = False
    return passable, fire


class TestAstar:
    """Tests for astar()."""

    def test_finds_path_on_open_grid(self):
        passable, fire = _make_masks(5, 5)
        path = astar(passable, fire, (0, 0), (4, 4))
        assert path is not None
        assert path[-1] == (4, 4)
        assert (0, 0) not in path

    def test_returns_none_when_no_path(self):
        passable, fire = _make_masks(5, 5)
        for x in range(5):
            passable[x, 2] = False
        path = astar(passable, fire, (0, 0), (4, 4))
        assert path is None

    def test_avoids_fire_cells(self):
        passable = np.ones((1, 5), dtype=bool)
        fire = np.zeros((1, 5), dtype=bool)
        fire[0, 2] = True
        passable[0, 2] = False
        path = astar(passable, fire, (0, 0), (0, 4))
        assert path is None

    def test_shortest_path_on_small_grid(self):
        passable = np.ones((3, 1), dtype=bool)
        fire = np.zeros((3, 1), dtype=bool)
        path = astar(passable, fire, (0, 0), (2, 0))
        assert path == [(1, 0), (2, 0)]

    def test_same_start_and_goal_returns_empty(self):
        passable, fire = _make_masks(5, 5)
        path = astar(passable, fire, (2, 2), (2, 2))
        assert path == []

    def test_goal_on_fire_returns_none(self):
        passable, fire = _make_masks(5, 5)
        fire[4, 4] = True
        passable[4, 4] = False
        path = astar(passable, fire, (0, 0), (4, 4))
        assert path is None


class TestGetNearestFrontier:
    """Tests for get_nearest_frontier()."""

    def _make_grid_state(
        self, width: int, height: int, obstacles: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
        from src.environment.grid import CellType

        gs = np.full((width, height), CellType.PASSABLE.value, dtype=int)
        if obstacles:
            for x, y in obstacles:
                gs[x, y] = CellType.OBSTACLE.value
        return gs

    def test_returns_closest_unvisited_cell(self):
        gs = self._make_grid_state(5, 5)
        visited = {(0, 0), (1, 0), (0, 1)}
        frontier = get_nearest_frontier((0, 0), visited, gs)
        assert frontier is not None
        assert frontier not in visited

    def test_returns_none_when_all_passable_cells_visited(self):
        gs = self._make_grid_state(3, 3)
        visited = {(x, y) for x in range(3) for y in range(3)}
        frontier = get_nearest_frontier((0, 0), visited, gs)
        assert frontier is None

    def test_does_not_return_fire_cell(self):
        from src.environment.grid import CellType

        gs = self._make_grid_state(3, 1)
        gs[1, 0] = CellType.FIRE.value
        gs[2, 0] = CellType.FIRE.value
        visited = {(0, 0)}
        frontier = get_nearest_frontier((0, 0), visited, gs)
        assert frontier is None

    def test_does_not_cross_obstacles_to_reach_frontier(self):
        from src.environment.grid import CellType

        gs = self._make_grid_state(5, 1)
        gs[2, 0] = CellType.OBSTACLE.value
        visited = {(0, 0), (1, 0)}
        frontier = get_nearest_frontier((0, 0), visited, gs)
        assert frontier is None

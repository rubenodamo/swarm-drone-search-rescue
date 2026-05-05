import numpy as np
import pytest

from src.environment.grid import CellType, DisasterGrid


@pytest.fixture
def grid() -> DisasterGrid:
    g = DisasterGrid(width=20, height=20, seed=0)
    g.place_obstacles()
    g.place_fire_seeds()
    g.place_survivors()
    return g


class TestDisasterGridInit:
    """Tests for DisasterGrid initialisation."""

    def test_dimensions(self) -> None:
        g = DisasterGrid(20, 20, seed=0)
        assert g.grid_state.shape == (20, 20)

    def test_cell_type_values_distinct(self) -> None:
        assert len({CellType.PASSABLE, CellType.OBSTACLE, CellType.FIRE}) == 3

    def test_initial_state_all_passable(self) -> None:
        g = DisasterGrid(20, 20, seed=0)
        assert (g.grid_state == CellType.PASSABLE).all()


class TestObstaclePlacement:
    """Tests for DisasterGrid.place_obstacles()."""

    def test_obstacle_count(self, grid: DisasterGrid) -> None:
        assert (grid.grid_state == CellType.OBSTACLE).sum() == 60

    def test_origin_not_obstacle(self, grid: DisasterGrid) -> None:
        assert grid.grid_state[0, 0] != CellType.OBSTACLE

    def test_different_seeds_give_different_layouts(self) -> None:
        g1 = DisasterGrid(20, 20, seed=1)
        g1.place_obstacles()
        g2 = DisasterGrid(20, 20, seed=2)
        g2.place_obstacles()
        assert not np.array_equal(g1.grid_state, g2.grid_state)


class TestSurvivorPlacement:
    """Tests for DisasterGrid.place_survivors()."""

    def test_survivor_count(self, grid: DisasterGrid) -> None:
        assert len(grid.survivors) == 10

    def test_no_survivor_on_obstacle(self, grid: DisasterGrid) -> None:
        for s in grid.survivors:
            x, y = s.pos
            assert grid.grid_state[x, y] != CellType.OBSTACLE

    def test_no_survivor_on_fire(self, grid: DisasterGrid) -> None:
        for s in grid.survivors:
            x, y = s.pos
            assert grid.grid_state[x, y] != CellType.FIRE


class TestFireSeeds:
    """Tests for DisasterGrid.place_fire_seeds()."""

    def test_fire_count(self, grid: DisasterGrid) -> None:
        assert (grid.grid_state == CellType.FIRE).sum() == 3

    def test_no_fire_near_origin(self, grid: DisasterGrid) -> None:
        fire_positions = list(zip(*np.where(grid.grid_state == CellType.FIRE)))
        for x, y in fire_positions:
            assert (
                abs(x) + abs(y) >= 4
            ), f"Fire at ({x},{y}) is too close to origin"


class TestFireSpread:
    """Tests for DisasterGrid.spread_fire()."""

    def test_spread_at_p1_covers_neighbours(self) -> None:
        g = DisasterGrid(5, 5, seed=0)
        g.grid_state[2, 2] = CellType.FIRE
        g.spread_fire(p=1.0)
        for nx, ny in [(2, 3), (2, 1), (3, 2), (1, 2)]:
            assert g.grid_state[nx, ny] == CellType.FIRE

    def test_spread_at_p0_no_change(self) -> None:
        g = DisasterGrid(5, 5, seed=0)
        g.grid_state[2, 2] = CellType.FIRE
        before = g.grid_state.copy()
        g.spread_fire(p=0.0)
        assert np.array_equal(g.grid_state, before)

    def test_obstacle_never_catches_fire(self) -> None:
        g = DisasterGrid(5, 5, seed=0)
        g.grid_state[2, 2] = CellType.FIRE
        g.grid_state[2, 3] = CellType.OBSTACLE
        g.spread_fire(p=1.0)
        assert g.grid_state[2, 3] == CellType.OBSTACLE

    def test_new_fire_does_not_spread_same_step(self) -> None:
        g = DisasterGrid(3, 1, seed=0)
        g.grid_state[0, 0] = CellType.FIRE
        g.spread_fire(p=1.0)
        assert g.grid_state[1, 0] == CellType.FIRE
        assert g.grid_state[2, 0] == CellType.PASSABLE

    def test_fire_cap(self) -> None:
        g = DisasterGrid(5, 5, seed=0)
        count = 0
        for x in range(5):
            for y in range(5):
                if count < 15:
                    g.grid_state[x, y] = CellType.FIRE
                    count += 1
        g.spread_fire(p=1.0)
        passable = (g.grid_state != CellType.OBSTACLE).sum()
        fire_after = (g.grid_state == CellType.FIRE).sum()
        assert fire_after / passable <= DisasterGrid.MAX_FIRE_FRACTION + 0.01

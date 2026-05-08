import pytest

from src.agents.astar_drone import AStarDrone
from src.agents.base_drone import DroneAgent
from src.agents.pheromone_drone import PheromoneDrone
from src.agents.random_drone import RandomDrone
from src.environment.grid import CellType, Survivor
from src.model.disaster_model import DisasterModel


class _ConcreteDrone(DroneAgent):
    """Concrete drone subclass for testing base class methods."""

    def step(self) -> None:
        self.detect_survivors()


def _make_model(seed: int = 42) -> DisasterModel:
    return DisasterModel(
        strategy="random", swarm_size=0, hazard_rate="medium", seed=seed
    )


def _place_drone(model: DisasterModel, pos: tuple[int, int]) -> _ConcreteDrone:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    drone = _ConcreteDrone(model)
    model.disaster_grid.grid.place_agent(drone, pos)
    return drone


class TestGetLocalObservation:
    """Tests for DroneAgent.get_local_observation()."""

    def test_returns_cells_within_manhattan_distance_2(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        obs = drone.get_local_observation()
        assert (5, 5) in obs
        assert (5, 7) in obs
        assert (7, 5) in obs
        assert (6, 6) in obs

    def test_excludes_cells_outside_radius(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        obs = drone.get_local_observation()
        assert (5, 8) not in obs
        assert (8, 5) not in obs
        assert (7, 7) not in obs

    def test_observation_count_at_interior_position(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        obs = drone.get_local_observation()
        assert len(obs) == 13


class TestSurvivorDetection:
    """Tests for DroneAgent.detect_survivors()."""

    def test_survivor_within_radius_is_found(self):
        model = _make_model()
        model.disaster_grid.survivors = [Survivor(pos=(7, 5))]
        drone = _place_drone(model, (5, 5))
        drone.step()
        assert model.disaster_grid.survivors[0].found is True
        assert model.survivors_found_count == 1

    def test_survivor_outside_radius_not_found(self):
        model = _make_model()
        model.disaster_grid.survivors = [Survivor(pos=(10, 5))]
        drone = _place_drone(model, (5, 5))
        drone.step()
        assert model.disaster_grid.survivors[0].found is False
        assert model.survivors_found_count == 0

    def test_already_found_survivor_not_double_counted(self):
        model = _make_model()
        model.disaster_grid.survivors = [Survivor(pos=(5, 5), found=True)]
        drone = _place_drone(model, (5, 5))
        drone.step()
        assert model.survivors_found_count == 0


class TestSurvivorDetectionNoise:
    """Tests for DroneAgent.detect_survivors() with noise_prob."""

    def test_noise_prob_1_never_detects_survivors(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            noise_prob=1.0,
        )
        model.disaster_grid.survivors = [Survivor(pos=(5, 5))]
        drone = _place_drone(model, (5, 5))
        for _ in range(30):
            drone.step()
        assert model.disaster_grid.survivors[0].found is False
        assert model.survivors_found_count == 0

    def test_noise_prob_0_detection_unchanged(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            noise_prob=0.0,
        )
        model.disaster_grid.survivors = [Survivor(pos=(7, 5))]
        drone = _place_drone(model, (5, 5))
        drone.step()
        assert model.disaster_grid.survivors[0].found is True
        assert model.survivors_found_count == 1


class TestMoveTo:
    """Tests for DroneAgent.move_to()."""

    def test_move_to_updates_position(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        model.disaster_grid.grid_state[5, 6] = CellType.PASSABLE
        drone.move_to((5, 6))
        assert drone.pos == (5, 6)

    def test_move_to_marks_cell_visited(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        model.disaster_grid.grid_state[5, 6] = CellType.PASSABLE
        drone.move_to((5, 6))
        assert (5, 6) in drone.visited_cells
        assert model.coverage_grid[5, 6] == 1

    def test_move_to_obstacle_raises(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        model.disaster_grid.grid_state[5, 6] = CellType.OBSTACLE
        with pytest.raises(ValueError):
            drone.move_to((5, 6))

    def test_move_to_fire_raises(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        model.disaster_grid.grid_state[5, 6] = CellType.FIRE
        with pytest.raises(ValueError):
            drone.move_to((5, 6))


class TestRandomDrone:
    """Tests for RandomDrone.step()."""

    def test_trajectory_is_reproducible_with_same_seed(self):
        def _run(seed: int) -> list[tuple[int, int]]:
            model = DisasterModel(
                strategy="random", swarm_size=1, hazard_rate="slow", seed=seed
            )
            positions = []
            for _ in range(10):
                model.step()
                agents = list(model.agents)
                if agents:
                    positions.append(agents[0].pos)
            return positions

        assert _run(0) == _run(0)

    def test_agent_never_moves_to_obstacle_or_fire(self):
        model = DisasterModel(
            strategy="random", swarm_size=1, hazard_rate="slow", seed=0
        )
        for _ in range(20):
            model.step()
            for agent in model.agents:
                cell_type = model.disaster_grid.grid_state[agent.pos]
                assert cell_type not in (CellType.OBSTACLE, CellType.FIRE)


def _place_astar_drone(
    model: DisasterModel, pos: tuple[int, int]
) -> AStarDrone:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    drone = AStarDrone(model)
    model.disaster_grid.grid.place_agent(drone, pos)
    return drone


class TestAStarDrone:
    """Tests for AStarDrone.step()."""

    def test_agent_moves_to_frontier_in_open_grid(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        drone = _place_astar_drone(model, (0, 0))
        drone.visited_cells = {(0, 0)}

        drone.step()

        assert drone.pos != (0, 0)
        assert model.disaster_grid.grid_state[drone.pos] == CellType.PASSABLE

    def test_agent_replans_when_fire_appears_on_path(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        drone = _place_astar_drone(model, (0, 0))
        drone.visited_cells = {(0, 0)}

        drone._replan()
        assert len(drone.current_path) > 0

        next_step = drone.current_path[0]
        model.disaster_grid.grid_state[next_step] = CellType.FIRE

        drone.step()

        assert drone.pos != next_step
        assert model.disaster_grid.grid_state[drone.pos] != CellType.FIRE

    def test_agent_never_moves_to_obstacle_or_fire(self):
        model = DisasterModel(
            strategy="astar", swarm_size=1, hazard_rate="slow", seed=0
        )
        for _ in range(20):
            model.step()
            for agent in model.agents:
                cell_type = model.disaster_grid.grid_state[agent.pos]
                assert cell_type not in (CellType.OBSTACLE, CellType.FIRE)

    def test_trajectory_is_reproducible_with_same_seed(self):
        def _run(seed: int) -> list[tuple[int, int]]:
            model = DisasterModel(
                strategy="astar", swarm_size=1, hazard_rate="slow", seed=seed
            )
            positions = []
            for _ in range(10):
                model.step()
                agents = list(model.agents)
                if agents:
                    positions.append(agents[0].pos)
            return positions

        assert _run(0) == _run(0)


def _place_pheromone_drone(
    model: DisasterModel, pos: tuple[int, int]
) -> PheromoneDrone:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    drone = PheromoneDrone(model)
    model.disaster_grid.grid.place_agent(drone, pos)
    return drone


class TestPheromoneDrone:
    """Tests for PheromoneDrone.step()."""

    def test_agent_avoids_high_pheromone_neighbour(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        drone = _place_pheromone_drone(model, (5, 5))

        model.pheromone_grid[5, 6] = 5.0

        for _ in range(10):
            drone.step()
            assert drone.pos != (5, 6), "drone moved to high-pheromone cell"

    def test_agent_deposits_pheromone_on_current_cell(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        drone = _place_pheromone_drone(model, (5, 5))

        drone.step()

        assert model.pheromone_grid[5, 5] >= 1.0

    def test_trajectory_is_reproducible_with_same_seed(self):
        def _run(seed: int) -> list[tuple[int, int]]:
            model = DisasterModel(
                strategy="pheromone",
                swarm_size=1,
                hazard_rate="slow",
                seed=seed,
            )
            positions = []
            for _ in range(10):
                model.step()
                agents = list(model.agents)
                if agents:
                    positions.append(agents[0].pos)
            return positions

        assert _run(0) == _run(0)

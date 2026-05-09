import pytest

from src.agents.astar_drone import AStarDrone
from src.agents.base_drone import DroneAgent
from src.agents.medic_drone import MedicDrone
from src.agents.pheromone_drone import PheromoneDrone
from src.agents.random_drone import RandomDrone
from src.agents.scout_drone import ScoutDrone
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
    """Tests for DroneAgent.detect_survivors() with survivor_detection_noise."""

    def test_survivor_noise_1_never_detects(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            survivor_detection_noise=1.0,
        )
        model.disaster_grid.survivors = [Survivor(pos=(5, 5))]
        drone = _place_drone(model, (5, 5))
        for _ in range(30):
            drone.step()
        assert model.disaster_grid.survivors[0].found is False
        assert model.survivors_found_count == 0

    def test_survivor_noise_0_detection_unchanged(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            survivor_detection_noise=0.0,
        )
        model.disaster_grid.survivors = [Survivor(pos=(7, 5))]
        drone = _place_drone(model, (5, 5))
        drone.step()
        assert model.disaster_grid.survivors[0].found is True
        assert model.survivors_found_count == 1


class TestHazardDetectionNoise:
    """Tests for DroneAgent.get_perceived_neighbours() with hazard_detection_noise."""

    def test_hazard_noise_0_excludes_fire_neighbours(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            hazard_detection_noise=0.0,
        )
        model.disaster_grid.grid_state[5, 6] = CellType.FIRE
        drone = _place_drone(model, (5, 5))
        assert (5, 6) not in drone.get_perceived_neighbours((5, 5))

    def test_hazard_noise_1_includes_fire_neighbours(self):
        model = DisasterModel(
            strategy="random",
            swarm_size=0,
            hazard_rate="medium",
            seed=42,
            hazard_detection_noise=1.0,
        )
        model.disaster_grid.grid_state[5, 6] = CellType.FIRE
        drone = _place_drone(model, (5, 5))
        assert (5, 6) in drone.get_perceived_neighbours((5, 5))


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

    def test_move_to_fire_is_allowed(self):
        model = _make_model()
        drone = _place_drone(model, (5, 5))
        model.disaster_grid.grid_state[5, 6] = CellType.FIRE
        drone.move_to((5, 6))
        assert drone.pos == (5, 6)


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

def _make_hetero_model(seed: int = 42) -> DisasterModel:
    return DisasterModel(
        strategy="heterogeneous", swarm_size=6, hazard_rate="medium", seed=seed
    )


def _place_scout(model: DisasterModel, pos: tuple[int, int]) -> ScoutDrone:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    scout = ScoutDrone(model)
    model.disaster_grid.grid.place_agent(scout, pos)
    return scout


def _place_medic(model: DisasterModel, pos: tuple[int, int]) -> MedicDrone:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    medic = MedicDrone(model)
    model.disaster_grid.grid.place_agent(medic, pos)
    return medic


class TestScoutDrone:
    """Tests for ScoutDrone: detect survivors into rescue_queue."""

    def test_sensing_radius_is_3(self):
        model = _make_model()
        scout = _place_scout(model, (5, 5))
        assert scout.sensing_radius == 3

    def test_scout_adds_to_rescue_queue_on_detection(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(7, 5))
        model.disaster_grid.survivors = [survivor]

        scout = _place_scout(model, (5, 5))
        scout._detect_and_queue_survivors()

        assert survivor in model.rescue_queue
        assert survivor.detected is True

    def test_scout_does_not_increment_survivors_found_count(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(6, 5))
        model.disaster_grid.survivors = [survivor]

        scout = _place_scout(model, (5, 5))
        scout.step()

        assert model.survivors_found_count == 0
        assert not survivor.found

    def test_scout_does_not_double_queue_already_detected_survivor(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(6, 5), detected=True)
        model.disaster_grid.survivors = [survivor]
        model.rescue_queue = [survivor]

        scout = _place_scout(model, (5, 5))
        scout._detect_and_queue_survivors()

        assert model.rescue_queue.count(survivor) == 1


class TestMedicDrone:
    """Tests for MedicDrone: A* navigation, rescue, fallback pheromone."""

    def test_sensing_radius_is_2(self):
        model = _make_model()
        medic = _place_medic(model, (5, 5))
        assert medic.sensing_radius == 2

    def test_medic_rescues_survivor_marks_found_and_increments_count(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(5, 5), detected=True)
        model.disaster_grid.survivors = [survivor]
        model.rescue_queue = [survivor]

        medic = _place_medic(model, (5, 5))
        medic.target = survivor

        medic.step()

        assert survivor.found is True
        assert model.survivors_found_count == 1

    def test_medic_clears_survivor_from_queue_after_rescue(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(5, 5), detected=True)
        model.disaster_grid.survivors = [survivor]
        model.rescue_queue = [survivor]

        medic = _place_medic(model, (5, 5))
        medic.target = survivor
        medic.step()

        assert survivor not in model.rescue_queue
        assert medic.target is None

    def test_fire_on_detected_survivor_removes_from_queue_and_frees_medic(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(10, 10), detected=True)
        model.disaster_grid.survivors = [survivor]
        model.rescue_queue = [survivor]

        medic = _place_medic(model, (5, 5))
        medic.target = survivor

        model.disaster_grid.grid_state[10, 10] = CellType.FIRE
        model.check_survivor_losses()

        assert survivor not in model.rescue_queue
        assert medic.target is None

    def test_medic_rescues_nearby_unqueued_survivor(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(6, 5))
        model.disaster_grid.survivors = [survivor]
        model.rescue_queue = []

        medic = _place_medic(model, (5, 5))
        medic.step()

        assert survivor.found is True
        assert model.survivors_found_count == 1

    def test_medic_does_not_rescue_fire_killed_survivor(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        survivor = Survivor(pos=(6, 5))
        model.disaster_grid.survivors = [survivor]
        model.disaster_grid.grid_state[6, 5] = CellType.FIRE
        model.rescue_queue = []

        medic = _place_medic(model, (5, 5))
        medic.step()

        assert survivor.found is False
        assert model.survivors_found_count == 0

    def test_medic_pheromone_move_every_2_idle_steps(self):
        model = _make_model()
        model.disaster_grid.grid_state[:] = CellType.PASSABLE
        model.rescue_queue = []

        medic = _place_medic(model, (5, 5))

        initial_pos = medic.pos
        medic.step()
        pos_after_step1 = medic.pos

        medic.step()
        pos_after_step2 = medic.pos

        assert pos_after_step1 == initial_pos
        assert pos_after_step2 != initial_pos


class TestHeterogeneousModel:
    """Tests for DisasterModel with strategy='heterogeneous'."""

    def test_swarm_size_6_gives_4_scouts_and_2_medics(self):
        model = _make_hetero_model()
        scouts = [a for a in model.agents if isinstance(a, ScoutDrone)]
        medics = [a for a in model.agents if isinstance(a, MedicDrone)]
        assert len(scouts) == 4
        assert len(medics) == 2

    def test_swarm_size_3_gives_2_scouts_and_1_medic(self):
        model = DisasterModel(
            strategy="heterogeneous", swarm_size=3, hazard_rate="medium", seed=0
        )
        scouts = [a for a in model.agents if isinstance(a, ScoutDrone)]
        medics = [a for a in model.agents if isinstance(a, MedicDrone)]
        assert len(scouts) == 2
        assert len(medics) == 1

    def test_hetero_run_completes_without_error(self):
        model = _make_hetero_model()
        while not model.is_done:
            model.step()
        assert model.timestep > 0

    def test_hetero_run_survivors_found_count_greater_than_zero(self):
        model = _make_hetero_model(seed=0)
        while not model.is_done:
            model.step()
        assert model.survivors_found_count > 0

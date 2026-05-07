import time

import mesa
import pytest

from src.environment.grid import CellType
from src.model.disaster_model import DisasterModel


class _StubAgent(mesa.Agent):
    """Minimal agent for use in model tests."""

    def step(self) -> None:
        pass


def _make_model(seed: int = 42) -> DisasterModel:
    return DisasterModel(
        strategy="random", swarm_size=0, hazard_rate="medium", seed=seed
    )


def _place_stub(model: DisasterModel, pos: tuple[int, int]) -> _StubAgent:
    model.disaster_grid.grid_state[pos] = CellType.PASSABLE
    agent = _StubAgent(model)
    model.disaster_grid.grid.place_agent(agent, pos)
    return agent


class TestAgentDeaths:
    """Tests for DisasterModel.check_agent_deaths()."""

    def test_agent_on_fire_cell_is_removed(self):
        model = _make_model()
        agent = _place_stub(model, (3, 3))
        model.disaster_grid.grid_state[3, 3] = CellType.FIRE

        model.check_agent_deaths()

        assert model.agents_lost == 1
        assert agent not in list(model.agents)

    def test_agent_on_passable_cell_survives(self):
        model = _make_model()
        agent = _place_stub(model, (3, 3))

        model.check_agent_deaths()

        assert model.agents_lost == 0
        assert agent in list(model.agents)

    def test_agent_removed_after_step_when_on_fire(self):
        model = _make_model()
        agent = _place_stub(model, (3, 3))
        model.disaster_grid.grid_state[3, 3] = CellType.FIRE

        model.step()

        assert model.agents_lost == 1
        assert agent not in list(model.agents)

    def test_multiple_agents_only_fire_ones_removed(self):
        model = _make_model()
        fire_agent = _place_stub(model, (3, 3))
        safe_agent = _place_stub(model, (7, 7))
        model.disaster_grid.grid_state[3, 3] = CellType.FIRE

        model.check_agent_deaths()

        assert model.agents_lost == 1
        assert fire_agent not in list(model.agents)
        assert safe_agent in list(model.agents)

    def test_timestep_increments_after_step(self):
        model = _make_model()
        assert model.timestep == 0
        model.step()
        assert model.timestep == 1


class TestDisasterModelInit:
    """Tests for DisasterModel.__init__() agent placement."""

    def test_agent_count_matches_swarm_size(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        assert len(list(model.agents)) == 6

    def test_step_runs_without_error(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        for _ in range(10):
            model.step()

    def test_agents_placed_at_origin(self):
        model = DisasterModel(
            strategy="random", swarm_size=3, hazard_rate="slow", seed=1
        )
        for agent in model.agents:
            assert agent.pos == (0, 0)

    def test_pheromone_evaporation_called_in_step(self):
        model = DisasterModel(
            strategy="pheromone", swarm_size=1, hazard_rate="slow", seed=0
        )
        model.pheromone_grid[5, 5] = 1.0
        model.step()
        assert model.pheromone_grid[5, 5] == pytest.approx(0.95)


class TestTermination:
    """Tests for DisasterModel.is_done and termination_reason."""

    def test_not_done_at_start(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        assert model.is_done is False
        assert model.termination_reason == ""

    def test_done_when_all_survivors_found(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        model.survivors_found_count = len(model.disaster_grid.survivors)
        assert model.is_done is True
        assert model.termination_reason == "survivors"

    def test_done_when_all_agents_dead(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        for agent in list(model.agents):
            model.disaster_grid.grid.remove_agent(agent)
            agent.remove()
        assert model.is_done is True
        assert model.termination_reason == "agents_dead"

    def test_done_when_timestep_200(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        model.timestep = 200
        assert model.is_done is True
        assert model.termination_reason == "timeout"

    def test_survivors_reason_takes_priority_over_timeout(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        model.timestep = 200
        model.survivors_found_count = len(model.disaster_grid.survivors)
        assert model.termination_reason == "survivors"


class TestPheromoneGrid:
    """Tests for DisasterModel pheromone grid initialisation and evaporation."""

    def test_pheromone_grid_initialised_to_zeros(self):
        model = DisasterModel(
            strategy="random", swarm_size=0, hazard_rate="medium", seed=0
        )
        import numpy as np

        assert model.pheromone_grid.shape == (20, 20)
        assert np.all(model.pheromone_grid == 0.0)

    def test_pheromone_decays_to_approx_point36_after_20_steps(self):
        model = DisasterModel(
            strategy="random", swarm_size=0, hazard_rate="slow", seed=0
        )
        model.pheromone_grid[10, 10] = 1.0
        for _ in range(20):
            model.evaporate_pheromones()
        assert model.pheromone_grid[10, 10] == pytest.approx(0.95**20, abs=1e-6)

    def test_pheromone_values_never_go_negative(self):
        model = DisasterModel(
            strategy="random", swarm_size=0, hazard_rate="slow", seed=0
        )
        model.pheromone_grid[5, 5] = 0.001
        for _ in range(100):
            model.evaporate_pheromones()
        assert model.pheromone_grid[5, 5] >= 0.0


class TestPheromoneStrategySmoke:
    """Tests for DisasterModel full run with pheromone strategy."""

    def test_run_completes_without_error(self):
        model = DisasterModel(
            strategy="pheromone", swarm_size=6, hazard_rate="medium", seed=0
        )
        while not model.is_done:
            model.step()
        assert 0 <= model.survivors_found_count <= 10
        assert 0 <= model.agents_lost <= 6

    def test_pheromone_grid_has_nonzero_values_after_run(self):
        model = DisasterModel(
            strategy="pheromone", swarm_size=6, hazard_rate="medium", seed=0
        )
        while not model.is_done:
            model.step()
        assert model.pheromone_grid.max() > 0.0

    def test_agents_visited_distinct_cells(self):
        model = DisasterModel(
            strategy="pheromone", swarm_size=6, hazard_rate="medium", seed=0
        )
        while not model.is_done:
            model.step()
        assert model.coverage_grid.sum() > 6


class TestSingleRunTiming:
    """Tests for single worst-case run timing benchmark."""

    def test_worst_case_run_under_3_seconds(self):
        model = DisasterModel(
            strategy="astar", swarm_size=12, hazard_rate="fast", seed=0
        )
        start = time.perf_counter()
        while not model.is_done:
            model.step()
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"Run took {elapsed:.2f}s, expected < 3s"


class TestRandomStrategySmoke:
    """Tests for DisasterModel full run with random strategy."""

    def test_run_completes_and_metrics_in_valid_range(self):
        model = DisasterModel(
            strategy="random", swarm_size=6, hazard_rate="medium", seed=0
        )
        while not model.is_done:
            model.step()
        print(
            f"Seed=0 | Strategy=random | "
            f"Survivors={model.survivors_found_count}/10 | "
            f"Agents lost={model.agents_lost} | Steps={model.timestep}"
        )
        assert 0 <= model.survivors_found_count <= 10
        assert 0 <= model.agents_lost <= 6


class TestAStarStrategySmoke:
    """Tests for DisasterModel full run with astar strategy."""

    def test_run_completes_and_metrics_in_valid_range(self):
        model = DisasterModel(
            strategy="astar", swarm_size=6, hazard_rate="medium", seed=0
        )
        while not model.is_done:
            model.step()
        print(
            f"Seed=0 | Strategy=astar | "
            f"Survivors={model.survivors_found_count}/10 | "
            f"Agents lost={model.agents_lost} | Steps={model.timestep}"
        )
        assert 0 <= model.survivors_found_count <= 10
        assert 0 <= model.agents_lost <= 6

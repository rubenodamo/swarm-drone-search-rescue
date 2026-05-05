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

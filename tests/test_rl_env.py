"""
Tests for DroneSearchEnv and RLDrone.
"""

import numpy as np
import pytest

from src.agents.rl_drone import RLDrone
from src.environment.drone_search_env import DroneSearchEnv
from src.model.disaster_model import DisasterModel


class TestObservationAndActionSpaces:
    """
    Tests for DroneSearchEnv observation and action space definitions.
    """

    def setup_method(self):
        self.env = DroneSearchEnv(hazard_rate="medium")

    def test_observation_space_shape(self):
        assert self.env.observation_space.shape == (100,)

    def test_observation_space_dtype(self):
        assert self.env.observation_space.dtype == np.float32

    def test_action_space_size(self):
        assert self.env.action_space.n == 5

    def test_observation_space_bounds(self):
        assert self.env.observation_space.low.min() == 0.0
        assert self.env.observation_space.high.max() == 1.0


class TestReset:
    """
    Tests for DroneSearchEnv.reset().
    """

    def setup_method(self):
        self.env = DroneSearchEnv(hazard_rate="medium")

    def test_reset_returns_obs_of_correct_shape(self):
        obs, info = self.env.reset(seed=0)
        assert obs.shape == (100,)

    def test_reset_obs_values_in_range(self):
        obs, info = self.env.reset(seed=0)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_reset_returns_dict_info(self):
        _, info = self.env.reset(seed=0)
        assert isinstance(info, dict)

    def test_reset_creates_single_rl_drone(self):
        self.env.reset(seed=0)
        agents = list(self.env.model.agents)
        assert len(agents) == 1
        assert isinstance(agents[0], RLDrone)

    def test_reset_with_explicit_seed_reproducible(self):
        obs1, _ = self.env.reset(seed=5)
        obs2, _ = self.env.reset(seed=5)
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    """
    Tests for DroneSearchEnv.step().
    """

    def setup_method(self):
        self.env = DroneSearchEnv(hazard_rate="slow")
        self.env.reset(seed=0)

    def test_step_returns_five_tuple(self):
        result = self.env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self):
        obs, *_ = self.env.step(0)
        assert obs.shape == (100,)

    def test_step_reward_is_float(self):
        _, reward, *_ = self.env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self):
        _, _, terminated, truncated, _ = self.env.step(0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_has_expected_keys(self):
        _, _, _, _, info = self.env.step(0)
        assert "survivors_found" in info
        assert "timestep" in info

    def test_stay_action_does_not_crash(self):
        for _ in range(5):
            obs, reward, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                break

    def test_all_actions_do_not_crash(self):
        for action in range(5):
            self.env.reset(seed=action)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            assert obs.shape == (100,)


class TestRewardStructure:
    """
    Tests for DroneSearchEnv reward signals.
    """

    def test_new_cell_visited_gives_small_positive_reward(self):
        env = DroneSearchEnv(hazard_rate="slow")
        env.reset(seed=0)
        drone = env._drone
        prev_visited = len(drone.visited_cells)
        for action in range(1, 5):
            env.reset(seed=0)
            _, reward, terminated, truncated, _ = env.step(action)
            if not (terminated or truncated):
                if len(env._drone.visited_cells) > prev_visited:
                    assert reward >= 0.01
                    return

    def test_agent_death_gives_negative_reward(self):
        env = DroneSearchEnv(hazard_rate="fast")
        env.reset(seed=0)
        death_reward = None
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated and info["survivors_found"] == 0:
                death_reward = reward
                break
            if terminated or truncated:
                break
        if death_reward is not None:
            assert death_reward <= -0.99


class TestEpisodeTermination:
    """
    Tests for DroneSearchEnv episode termination conditions.
    """

    def test_episode_terminates_eventually(self):
        env = DroneSearchEnv(hazard_rate="medium")
        env.reset(seed=0)
        for _ in range(250):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                return
        pytest.fail("Episode did not terminate within 250 steps")

    def test_truncated_at_200_steps(self):
        env = DroneSearchEnv(hazard_rate="slow")
        env.reset(seed=0)
        terminated = False
        truncated = False
        for _ in range(250):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        assert info["timestep"] <= 200


class TestRLDroneIntegration:
    """
    Tests for RLDrone within DisasterModel.
    """

    def test_rl_strategy_creates_rl_drone(self):
        model = DisasterModel(
            strategy="rl", swarm_size=1, hazard_rate="medium", seed=0
        )
        agents = list(model.agents)
        assert len(agents) == 1
        assert isinstance(agents[0], RLDrone)

    def test_rl_drone_pending_action_defaults_to_stay(self):
        model = DisasterModel(
            strategy="rl", swarm_size=1, hazard_rate="medium", seed=0
        )
        drone = next(iter(model.agents))
        assert drone.pending_action == 0

    def test_rl_drone_step_with_stay_keeps_position(self):
        model = DisasterModel(
            strategy="rl", swarm_size=1, hazard_rate="slow", seed=0
        )
        drone = next(iter(model.agents))
        drone.pending_action = 0
        start_pos = drone.pos
        drone.step()
        assert drone.pos == start_pos

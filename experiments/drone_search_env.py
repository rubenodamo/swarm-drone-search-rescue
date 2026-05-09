from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.model.disaster_model import DisasterModel

# Steady-state pheromone ceiling: 1 / (1 - evaporation_rate) = 1 / 0.05 = 20
_PHEROMONE_MAX: float = 20.0


class DroneSearchEnv(gym.Env):
    """
    Gymnasium environment wrapping a single-drone DisasterModel.

    Attributes:
        - hazard_rate: Named hazard level passed to DisasterModel.
        - observation_space: Box(0, 1, shape=(75,), float32).
        - action_space: Discrete(5).
    """

    metadata: dict = {"render_modes": []}

    _ACTIONS: dict[int, tuple[int, int]] = {
        0: (0, 0),
        1: (0, 1),
        2: (0, -1),
        3: (1, 0),
        4: (-1, 0),
    }

    def __init__(self, hazard_rate: str = "medium") -> None:
        """
        Initialises the environment.

        Args:
            - hazard_rate: Fire spread rate ('slow', 'medium', 'fast').
        """
        super().__init__()
        self.hazard_rate = hazard_rate

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(75,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.model: DisasterModel | None = None
        self._drone = None
        self._episode_seed: int = -1

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to a new episode.

        Args:
            - seed: Optional seed for this episode; auto-incremented if None.
            - options: Unused; present for gymnasium API compliance.

        Returns:
            - Tuple of (observation array of shape (75,), info dict).
        """
        super().reset(seed=seed)
        self._episode_seed = seed if seed is not None else self._episode_seed + 1

        self.model = DisasterModel(
            strategy="rl",
            swarm_size=1,
            hazard_rate=self.hazard_rate,
            seed=self._episode_seed,
        )
        self._drone = next(iter(self.model.agents))

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advances the simulation by one timestep with the given action.

        Args:
            - action: Integer in [0, 4] (Stay/N/S/E/W).

        Returns:
            - observation: np.ndarray of shape (75,).
            - reward: Float reward for this step.
            - terminated: True if episode ended naturally (survivor found or death).
            - truncated: True if episode ended due to 200-step timeout.
            - info: Dict with 'survivors_found' and 'timestep'.
        """
        self._drone.pending_action = int(action)

        prev_found = self.model.survivors_found_count
        prev_visited = frozenset(self._drone.visited_cells)
        drone_agents = list(self.model.agents)
        drone_was_alive = self._drone in drone_agents

        self.model.step()

        drone_alive = self._drone in list(self.model.agents)
        agent_died = drone_was_alive and not drone_alive

        new_survivors = self.model.survivors_found_count - prev_found
        new_cells = (
            len(self._drone.visited_cells - prev_visited) if drone_alive else 0
        )

        reward = float(
            new_survivors * 1.0
            + new_cells * 0.01
            - (1.0 if agent_died else 0.0)
        )

        all_found = self.model.survivors_found_count >= len(
            self.model.disaster_grid.survivors
        )
        terminated = agent_died or all_found
        truncated = (not terminated) and self.model.timestep >= 200

        obs = (
            self._get_obs()
            if drone_alive
            else np.zeros(75, dtype=np.float32)
        )
        info = {
            "survivors_found": self.model.survivors_found_count,
            "timestep": self.model.timestep,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        cx, cy = self._drone.pos
        grid_state = self.model.disaster_grid.grid_state
        pheromone = self.model.pheromone_grid
        width = self.model.disaster_grid.width
        height = self.model.disaster_grid.height

        survivor_positions = {
            s.pos for s in self.model.disaster_grid.survivors if not s.found
        }

        obs = np.zeros((3, 5, 5), dtype=np.float32)
        for di, dx in enumerate(range(-2, 3)):
            for dj, dy in enumerate(range(-2, 3)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    obs[0, di, dj] = grid_state[nx, ny] / 2.0
                    obs[1, di, dj] = min(
                        pheromone[nx, ny] / _PHEROMONE_MAX, 1.0
                    )
                    obs[2, di, dj] = (
                        1.0 if (nx, ny) in survivor_positions else 0.0
                    )
                else:
                    obs[0, di, dj] = 0.5

        return obs.flatten()

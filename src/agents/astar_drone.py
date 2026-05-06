import mesa

from src.agents.base_drone import DroneAgent


class AStarDrone(DroneAgent):
    """
    Drone that uses A* pathfinding to navigate to the nearest frontier.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
        - current_path: Planned path to current target.
        - current_target: Current frontier target position.
    """

    def __init__(self, model: mesa.Model) -> None:
        """
        Initialises the A* drone.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.current_path: list[tuple[int, int]] = []
        self.current_target: tuple[int, int] | None = None

    def step(self) -> None:
        """
        Executes one simulation step (stub — implemented in Phase 6).
        """

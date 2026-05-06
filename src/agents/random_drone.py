from src.agents.base_drone import DroneAgent


class RandomDrone(DroneAgent):
    """
    Drone that moves to a random passable neighbour each step.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
    """

    def step(self) -> None:
        """
        Executes one simulation step (stub — implemented in Phase 5).
        """

from src.agents.base_drone import DroneAgent


class PheromoneDrone(DroneAgent):
    """
    Drone that uses pheromone stigmergy to coordinate with the swarm.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
    """

    def step(self) -> None:
        """
        Executes one simulation step (stub — implemented in Phase 7).
        """

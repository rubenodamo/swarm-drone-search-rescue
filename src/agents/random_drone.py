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
        Moves to a random passable, non-fire neighbour and detects survivors.
        """
        neighbours = self.get_passable_neighbours(self.pos)
        if neighbours:
            new_pos = tuple(int(c) for c in self.model.rng.choice(neighbours))
            self.move_to(new_pos)
        self.detect_survivors()

from src.agents.base_drone import DroneAgent


class RandomDrone(DroneAgent):
    """
    Drone that moves to a random passable neighbour each step.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
    """

    @property
    def portrayal(self) -> dict:
        """
        Returns the visual portrayal dict for this drone.

        Returns:
            - Dict with Color, Shape, Filled, Layer, r keys.
        """
        return {
            "Color": "#1f77b4",
            "Shape": "circle",
            "Filled": True,
            "Layer": 2,
            "r": 0.45,
        }

    def step(self) -> None:
        """
        Moves to a random passable, non-fire neighbour and detects survivors.
        """
        neighbours = self.get_perceived_neighbours(self.pos)
        if neighbours:
            new_pos = tuple(int(c) for c in self.model.rng.choice(neighbours))
            self.move_to(new_pos)
        self.detect_survivors()

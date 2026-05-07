from src.agents.base_drone import DroneAgent


class PheromoneDrone(DroneAgent):
    """
    Drone that uses pheromone stigmergy to coordinate with the swarm.

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
            "Color": "#2ca02c",
            "Shape": "circle",
            "Filled": True,
            "Layer": 2,
            "r": 0.45,
        }

    def step(self) -> None:
        """
        Executes one simulation step for the pheromone drone.
        """
        self.model.pheromone_grid[self.pos] += 1.0

        neighbours = self.get_passable_neighbours(self.pos)
        if neighbours:
            min_val = min(self.model.pheromone_grid[n] for n in neighbours)
            candidates = [
                n for n in neighbours if self.model.pheromone_grid[n] == min_val
            ]
            chosen = candidates[self.model.rng.integers(0, len(candidates))]
            self.move_to(chosen)

        self.detect_survivors()

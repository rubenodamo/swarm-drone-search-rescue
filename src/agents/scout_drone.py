from src.agents.base_drone import DroneAgent


class ScoutDrone(DroneAgent):
    """
    A drone specialised in exploring the environment and detecting survivors.

    Attributes:
        - alive: Whether the agent is currently active.
        - visited_cells: Set of (x, y) positions visited by this agent.
        - sensing_radius: 3 (wide detection net).
    """

    def __init__(self, model) -> None:
        """
        Initialises the scout drone with sensing_radius=3.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.sensing_radius = 3

    def _detect_and_queue_survivors(self) -> None:
        """
        Detect survivors within sensing radius and add new ones to rescue_queue.
        """
        noise = self.model.survivor_detection_noise
        x, y = self.pos
        for survivor in self.model.disaster_grid.survivors:
            if survivor.found or survivor.detected:
                continue
            sx, sy = survivor.pos
            if abs(x - sx) + abs(y - sy) <= self.sensing_radius:
                if noise > 0.0 and self.model.rng.random() < noise:
                    continue
                survivor.detected = True
                self.model.rescue_queue.append(survivor)

    def step(self) -> None:
        """
        Executes one simulation step using pheromone stigmergy.
        """
        self.model.pheromone_grid[self.pos] += 1.0

        neighbours = self.get_perceived_neighbours(self.pos)
        if neighbours:
            min_val = min(self.model.pheromone_grid[n] for n in neighbours)
            candidates = [
                n for n in neighbours if self.model.pheromone_grid[n] == min_val
            ]
            chosen = candidates[self.model.rng.integers(0, len(candidates))]
            self.move_to(chosen)

        self._detect_and_queue_survivors()

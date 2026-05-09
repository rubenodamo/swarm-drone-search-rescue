import mesa

from src.agents.base_drone import DroneAgent


class RLDrone(DroneAgent):
    """
    Drone agent controlled by an external RL policy via pending_action.

    Attributes:
        - pending_action: Integer action set by DroneSearchEnv before each step.
    """

    # Action mapping: 0=Stay, 1=Up, 2=Down, 3=Right, 4=Left
    ACTIONS: dict[int, tuple[int, int]] = {
        0: (0, 0),
        1: (0, 1),
        2: (0, -1),
        3: (1, 0),
        4: (-1, 0),
    }

    def __init__(self, model: mesa.Model) -> None:
        """
        Initialises the RL drone.

        Args:
            - model: The DisasterModel instance managing this agent.
        """
        super().__init__(model)
        self.pending_action: int = 0

    def step(self) -> None:
        """
        Executes the action set in pending_action, then detects survivors.
        """
        dx, dy = self.ACTIONS[self.pending_action]
        if dx != 0 or dy != 0:
            x, y = self.pos
            new_pos = (x + dx, y + dy)
            if new_pos in self.get_perceived_neighbours(self.pos):
                self.move_to(new_pos)
        self.detect_survivors()

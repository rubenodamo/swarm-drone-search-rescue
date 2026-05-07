import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.grid import CellType
from src.model.disaster_model import DisasterModel

CELL_SIZE = 28
PADDING = 10
DRONE_RADIUS = CELL_SIZE * 0.35

_CELL_COLOURS: dict[int, str] = {
    CellType.PASSABLE: "#F0F0F0",
    CellType.OBSTACLE: "#555555",
    CellType.FIRE: "#FF4500",
}

STRATEGY_COLOURS: dict[str, str] = {
    "random": "#1f77b4",
    "astar": "#ff7f0e",
    "pheromone": "#2ca02c",
}


class PlaygroundApp:
    """
    Tkinter animated playground for the swarm drone simulation.

    Attributes:
        - root: The root Tk window.
        - model: The active DisasterModel instance.
        - canvas: The Canvas widget showing the grid.
        - running: Whether the simulation loop is actively stepping.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialise the playground with a default model.

        Args:
            - root: The root Tk window.
        """
        self.root = root
        self.root.title("Swarm Drone Search & Rescue — Playground")
        self.root.resizable(False, False)

        self.running: bool = False
        self._speed: int = 1
        self._step_duration_ms: float = 1000.0 / self._speed

        self._survivor_items: dict[tuple[int, int], int] = {}
        self._drone_ovals: dict[int, int] = {}
        
        self._drone_interp: dict[int, tuple[float, float, float, float, float]] = {}

        self.model = DisasterModel(
            strategy="random",
            swarm_size=6,
            hazard_rate="medium",
            seed=0,
        )

        self._build_canvas()
        self._init_drones()
        self.update_cells()

        self.running = True
        self._schedule_sim()
        self._render_loop()

        self.root.bind("<space>", lambda _: self._toggle_running())

    def _build_canvas(self) -> None:
        """Create the canvas widget and draw the initial grid and survivors."""
        grid = self.model.disaster_grid
        canvas_w = grid.width * CELL_SIZE + 2 * PADDING
        canvas_h = grid.height * CELL_SIZE + 2 * PADDING

        self.canvas = tk.Canvas(
            self.root,
            width=canvas_w,
            height=canvas_h,
            bg="#AAAAAA",
            highlightthickness=0,
        )
        self.canvas.pack()

        for x in range(grid.width):
            for y in range(grid.height):
                cx, cy = self._cell_topleft(x, y)
                colour = _CELL_COLOURS[int(grid.grid_state[x, y])]
                self.canvas.create_rectangle(
                    cx,
                    cy,
                    cx + CELL_SIZE,
                    cy + CELL_SIZE,
                    fill=colour,
                    outline="#AAAAAA",
                    width=1,
                    tags=f"cell_{x}_{y}",
                )

        self._survivor_items = {}
        for survivor in grid.survivors:
            self._draw_survivor_star(survivor.pos)

    def _init_drones(self) -> None:
        """Create oval canvas items for all agents at their starting positions."""
        colour = STRATEGY_COLOURS[self.model.strategy]
        r = DRONE_RADIUS
        now_ms = time.monotonic() * 1000

        for agent in self.model.agents:
            px, py = self._cell_centre(*agent.pos)
            oval_id = self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=colour,
                outline="white",
                width=1,
            )
            self._drone_ovals[agent.unique_id] = oval_id
            self._drone_interp[agent.unique_id] = (px, py, px, py, now_ms)

    def _cell_topleft(self, x: int, y: int) -> tuple[int, int]:
        """
        Return the canvas top-left pixel coordinates for grid cell (x, y).

        Args:
            - x: Grid x coordinate.
            - y: Grid y coordinate.

        Returns:
            - (canvas_x, canvas_y) of the top-left corner of the cell.
        """
        height = self.model.disaster_grid.height
        canvas_x = PADDING + x * CELL_SIZE
        canvas_y = PADDING + (height - 1 - y) * CELL_SIZE
        return canvas_x, canvas_y

    def _cell_centre(self, x: int, y: int) -> tuple[float, float]:
        """
        Return the canvas pixel coordinates of the centre of cell (x, y).

        Args:
            - x: Grid x coordinate.
            - y: Grid y coordinate.

        Returns:
            - (px, py) pixel centre of the cell.
        """
        cx, cy = self._cell_topleft(x, y)
        return cx + CELL_SIZE / 2, cy + CELL_SIZE / 2

    def _draw_survivor_star(self, pos: tuple[int, int]) -> None:
        """
        Place a gold star text item at the centre of cell pos.

        Args:
            - pos: Grid (x, y) of the survivor.
        """
        cx, cy = self._cell_topleft(*pos)
        item_id = self.canvas.create_text(
            cx + CELL_SIZE // 2,
            cy + CELL_SIZE // 2,
            text="★",
            fill="#FFD700",
            font=("Arial", 10, "bold"),
        )
        self._survivor_items[pos] = item_id

    def update_cells(self) -> None:
        """
        Refresh cell colours and survivor stars via canvas.itemconfig().
        No full redraw — only itemconfig calls.
        """
        grid = self.model.disaster_grid

        for x in range(grid.width):
            for y in range(grid.height):
                colour = _CELL_COLOURS[int(grid.grid_state[x, y])]
                self.canvas.itemconfig(f"cell_{x}_{y}", fill=colour)

        for survivor in grid.survivors:
            if survivor.found and survivor.pos in self._survivor_items:
                self.canvas.delete(self._survivor_items.pop(survivor.pos))

    def _schedule_sim(self) -> None:
        """Schedule the next sim step if running and model is not done."""
        if self.running and not self.model.is_done:
            self.root.after(int(self._step_duration_ms), self._sim_step)

    def _sim_step(self) -> None:
        """Advance the model one step and reconcile the drone canvas items."""
        now_ms = time.monotonic() * 1000
        old_positions = {agent.unique_id: agent.pos for agent in self.model.agents}

        self.model.step()
        self.update_cells()

        alive_ids = {agent.unique_id for agent in self.model.agents}

        for uid in list(self._drone_ovals):
            if uid not in alive_ids:
                self.canvas.delete(self._drone_ovals.pop(uid))
                self._drone_interp.pop(uid, None)

        for agent in self.model.agents:
            uid = agent.unique_id
            old_px, old_py = self._cell_centre(*old_positions.get(uid, agent.pos))
            new_px, new_py = self._cell_centre(*agent.pos)
            self._drone_interp[uid] = (old_px, old_py, new_px, new_py, now_ms)

        self._schedule_sim()

    def _render_loop(self) -> None:
        """Reposition drone ovals using linear interpolation at ~60fps."""
        now_ms = time.monotonic() * 1000
        r = DRONE_RADIUS

        for uid, (old_px, old_py, new_px, new_py, start_ms) in self._drone_interp.items():
            if uid not in self._drone_ovals:
                continue
            t = min(1.0, (now_ms - start_ms) / self._step_duration_ms)
            px = old_px + (new_px - old_px) * t
            py = old_py + (new_py - old_py) * t
            self.canvas.coords(self._drone_ovals[uid], px - r, py - r, px + r, py + r)

        self.root.after(16, self._render_loop)

    def _toggle_running(self) -> None:
        """Toggle the simulation loop on/off."""
        self.running = not self.running
        if self.running:
            self._schedule_sim()


def main() -> None:
    """Launch the playground window."""
    root = tk.Tk()
    PlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

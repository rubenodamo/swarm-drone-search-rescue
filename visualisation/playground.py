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
        Initialise the playground with a default model and full UI.

        Args:
            - root: The root Tk window.
        """
        self.root = root
        self.root.title("Swarm Drone Search & Rescue — Playground")
        self.root.resizable(False, False)

        self.running: bool = False
        self._step_duration_ms: float = 1000.0

        self._survivor_items: dict[tuple[int, int], int] = {}
        self._drone_ovals: dict[int, int] = {}
        self._drone_interp: dict[int, tuple[float, float, float, float, float]] = {}

        self._strategy_var = tk.StringVar(value="random")
        self._swarm_var = tk.StringVar(value="6")
        self._hazard_var = tk.StringVar(value="medium")
        self._seed_var = tk.IntVar(value=0)
        self._speed_var = tk.IntVar(value=1)

        self.model = DisasterModel(
            strategy="random",
            swarm_size=6,
            hazard_rate="medium",
            seed=0,
        )

        self._build_left_panel()
        self._build_canvas()
        self._init_drones()
        self.update_cells()

        self.running = True
        self._update_play_pause_label()
        self._schedule_sim()
        self._render_loop()

    def _build_left_panel(self) -> None:
        """Create the left control panel with Parameters and Controls sections."""
        panel = tk.Frame(self.root, padx=8, pady=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        params = tk.LabelFrame(panel, text="Parameters", padx=6, pady=6)
        params.pack(fill=tk.X, pady=(0, 8))

        self._add_option_row(params, 0, "Strategy", self._strategy_var,
                             ["random", "astar", "pheromone"])
        self._add_option_row(params, 1, "Swarm size", self._swarm_var,
                             ["3", "6", "12"])
        self._add_option_row(params, 2, "Hazard rate", self._hazard_var,
                             ["slow", "medium", "fast"])

        tk.Label(params, text="Seed", anchor="w").grid(
            row=3, column=0, sticky="w", pady=2)
        tk.Scale(params, from_=0, to=29, orient=tk.HORIZONTAL,
                 variable=self._seed_var, length=140).grid(
            row=3, column=1, sticky="ew", pady=2)

        tk.Label(params, text="Speed", anchor="w").grid(
            row=4, column=0, sticky="w", pady=2)
        tk.Scale(params, from_=1, to=10, orient=tk.HORIZONTAL,
                 variable=self._speed_var, length=140,
                 command=self._on_speed_change).grid(
            row=4, column=1, sticky="ew", pady=2)

        controls = tk.LabelFrame(panel, text="Controls", padx=6, pady=6)
        controls.pack(fill=tk.X)

        self._play_pause_btn = tk.Button(
            controls, text="Pause", width=12, command=self._toggle_running)
        self._play_pause_btn.pack(fill=tk.X, pady=2)

        tk.Button(controls, text="Step", width=12,
                  command=self._step_once).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Reset", width=12,
                  command=self._reset).pack(fill=tk.X, pady=2)

    def _add_option_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        var: tk.StringVar,
        options: list[str],
    ) -> None:
        """
        Add a label + OptionMenu row to a grid-managed parent widget.

        Args:
            - parent: The parent widget using grid layout.
            - row: Grid row index.
            - label: Text for the label column.
            - var: StringVar bound to the OptionMenu.
            - options: List of option strings.
        """
        tk.Label(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", pady=2)
        tk.OptionMenu(parent, var, *options).grid(
            row=row, column=1, sticky="ew", pady=2)

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
        self.canvas.pack(side=tk.LEFT)

        for x in range(grid.width):
            for y in range(grid.height):
                cx, cy = self._cell_topleft(x, y)
                colour = _CELL_COLOURS[int(grid.grid_state[x, y])]
                self.canvas.create_rectangle(
                    cx, cy,
                    cx + CELL_SIZE, cy + CELL_SIZE,
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
        # Mesa y=0 is bottom; flip so it renders at the bottom of the canvas.
        canvas_y = PADDING + (height - 1 - y) * CELL_SIZE
        return canvas_x, canvas_y

    def _cell_centre(self, x: int, y: int) -> tuple[float, float]:
        """
        Return the canvas pixel centre for grid cell (x, y).

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
        """Scheduled sim loop callback — bails if no longer running."""
        if not self.running:
            return
        self._do_step()
        self._schedule_sim()

    def _do_step(self) -> None:
        """Execute one model step and update the canvas."""
        if self.model.is_done:
            return

        now_ms = time.monotonic() * 1000
        old_positions = {
            agent.unique_id: agent.pos for agent in self.model.agents
        }

        self.model.step()
        self.update_cells()

        alive_ids = {agent.unique_id for agent in self.model.agents}
        for uid in list(self._drone_ovals):
            if uid not in alive_ids:
                self.canvas.delete(self._drone_ovals.pop(uid))
                self._drone_interp.pop(uid, None)

        for agent in self.model.agents:
            uid = agent.unique_id
            old_px, old_py = self._cell_centre(
                *old_positions.get(uid, agent.pos)
            )
            new_px, new_py = self._cell_centre(*agent.pos)
            self._drone_interp[uid] = (old_px, old_py, new_px, new_py, now_ms)

        if self.model.is_done:
            self.running = False
            self._update_play_pause_label()

    def _render_loop(self) -> None:
        """Reposition drone ovals using linear interpolation at ~60fps."""
        now_ms = time.monotonic() * 1000
        r = DRONE_RADIUS

        for uid, (old_px, old_py, new_px, new_py, start_ms) in (
            self._drone_interp.items()
        ):
            if uid not in self._drone_ovals:
                continue
            t = min(1.0, (now_ms - start_ms) / self._step_duration_ms)
            px = old_px + (new_px - old_px) * t
            py = old_py + (new_py - old_py) * t
            self.canvas.coords(
                self._drone_ovals[uid], px - r, py - r, px + r, py + r
            )

        self.root.after(16, self._render_loop)

    def _toggle_running(self) -> None:
        """Toggle the simulation loop and update the Play/Pause button label."""
        self.running = not self.running
        self._update_play_pause_label()
        if self.running:
            self._schedule_sim()

    def _step_once(self) -> None:
        """Advance the model exactly one step regardless of running state."""
        self._do_step()

    def _reset(self) -> None:
        """Reinitialise the model from current widget values and redraw canvas."""
        self.running = False

        for oval_id in self._drone_ovals.values():
            self.canvas.delete(oval_id)
        self._drone_ovals.clear()
        self._drone_interp.clear()

        for item_id in self._survivor_items.values():
            self.canvas.delete(item_id)
        self._survivor_items.clear()

        self.model = DisasterModel(
            strategy=self._strategy_var.get(),
            swarm_size=int(self._swarm_var.get()),
            hazard_rate=self._hazard_var.get(),
            seed=self._seed_var.get(),
        )

        self.update_cells()
        for survivor in self.model.disaster_grid.survivors:
            self._draw_survivor_star(survivor.pos)
        self._init_drones()

        self.running = True
        self._update_play_pause_label()
        self._schedule_sim()

    def _on_speed_change(self, _value: str) -> None:
        """Update step duration when the Speed scale changes."""
        self._step_duration_ms = 1000.0 / self._speed_var.get()

    def _update_play_pause_label(self) -> None:
        """Sync the Play/Pause button text with the current running state."""
        self._play_pause_btn.config(text="Pause" if self.running else "Play")


def main() -> None:
    """Launch the playground window."""
    root = tk.Tk()
    PlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

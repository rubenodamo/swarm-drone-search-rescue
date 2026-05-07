import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.grid import CellType
from src.model.disaster_model import DisasterModel

CELL_SIZE = 28
PADDING = 10

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

        self.model = DisasterModel(
            strategy="random",
            swarm_size=6,
            hazard_rate="medium",
            seed=0,
        )

        self._survivor_items: dict[tuple[int, int], int] = {}

        self._build_canvas()
        self.update_cells()

        self.root.bind("<space>", lambda _: self._manual_step())

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

    def _cell_topleft(self, x: int, y: int) -> tuple[int, int]:
        """
        Return the canvas (top-left) pixel coordinates for grid cell (x, y).

        Args:
            - x: Grid x coordinate.
            - y: Grid y coordinate.

        Returns:
            - (canvas_x, canvas_y) of the top-left corner of the cell.
        """
        height = self.model.disaster_grid.height
        # Mesa y=0 is bottom of grid; flip so it renders at bottom of canvas.
        canvas_x = PADDING + x * CELL_SIZE
        canvas_y = PADDING + (height - 1 - y) * CELL_SIZE
        return canvas_x, canvas_y

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

    def _manual_step(self) -> None:
        """Advance the model one step on Space keypress and refresh display."""
        if not self.model.is_done:
            self.model.step()
            self.update_cells()


def main() -> None:
    """Launch the playground window."""
    root = tk.Tk()
    PlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

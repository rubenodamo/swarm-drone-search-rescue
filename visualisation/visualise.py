import matplotlib.colors as mcolors
import numpy as np
import solara
from matplotlib.figure import Figure

from src.environment.grid import CellType
from src.model.disaster_model import DisasterModel

CELL_COLORS: dict[int, str] = {
    CellType.PASSABLE: "#F0F0F0",
    CellType.OBSTACLE: "#555555",
    CellType.FIRE: "#FF4500",
}
SURVIVOR_COLOR = "#FFD700"


def _apply_pheromone_overlay(
    color_array: np.ndarray,
    pheromone_grid: np.ndarray,
    grid_state: np.ndarray,
) -> None:
    """
    Blends pheromone intensity into color_array in-place (passable cells only).

    Args:
        - color_array: RGBA array of shape (height, width, 4) to modify.
        - pheromone_grid: Raw pheromone values of shape (width, height).
        - grid_state: Cell-type array of shape (width, height).
    """
    max_p = pheromone_grid.max()
    if max_p <= 0:
        return
    intensity = (pheromone_grid.T / max_p)[
        ..., np.newaxis
    ]
    purple = np.array(mcolors.to_rgba("#7B2D8B"))
    passable = (grid_state.T == CellType.PASSABLE)[..., np.newaxis]
    alpha = intensity * passable
    color_array[:] = color_array * (1 - alpha) + purple * alpha


def portray_cell(cell_type: CellType) -> dict:
    """
    Returns the portrayal dict for a given cell type.

    Args:
        - cell_type: The CellType enum value for this cell.

    Returns:
        - Dict with Color, Shape, Filled, Layer, w, h keys.
    """
    base: dict = {"Shape": "rect", "Filled": True, "Layer": 0, "w": 1, "h": 1}
    base["Color"] = CELL_COLORS.get(cell_type, "#F0F0F0")
    return base


@solara.component
def GridView(model: DisasterModel) -> None:
    """
    Renders the disaster grid with cell backgrounds and survivor positions.

    Args:
        - model: The DisasterModel instance to visualise.
    """
    fig = Figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    grid_state = model.disaster_grid.grid_state
    width = model.disaster_grid.width
    height = model.disaster_grid.height

    color_array = np.ones((height, width, 4))
    for cell_type, hex_color in CELL_COLORS.items():
        rgba = np.array(mcolors.to_rgba(hex_color))
        mask = grid_state.T == cell_type
        color_array[mask] = rgba

    if model.strategy == "pheromone":
        _apply_pheromone_overlay(color_array, model.pheromone_grid, grid_state)

    ax.imshow(
        color_array,
        origin="lower",
        extent=[-0.5, width - 0.5, -0.5, height - 0.5],
        aspect="equal",
        interpolation="nearest",
    )

    for survivor in model.disaster_grid.survivors:
        if not survivor.found:
            ax.plot(
                survivor.pos[0],
                survivor.pos[1],
                "o",
                color=SURVIVOR_COLOR,
                markersize=7,
                zorder=3,
            )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Step {model.timestep}  |  "
        f"Survivors: {model.survivors_found_count}/10"
    )

    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")

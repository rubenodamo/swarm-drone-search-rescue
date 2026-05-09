import heapq
from collections import deque

import numpy as np


def astar(
    passable_mask: np.ndarray,
    fire_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """
    Find shortest path from start to goal avoiding obstacles and fire.

    Args:
        - passable_mask: 2D bool array; True where cells are
          passable (not OBSTACLE).
        - fire_mask: 2D bool array; True where cells are on FIRE.
        - start: (x, y) starting position (excluded from returned path).
        - goal: (x, y) target position (included in returned path).

    Returns:
        - List of (x, y) positions from start to goal (not including start),
          or None if no path exists.
    """
    if start == goal:
        return []

    width, height = passable_mask.shape

    def is_reachable(pos: tuple[int, int]) -> bool:
        x, y = pos
        return (
            0 <= x < width
            and 0 <= y < height
            and passable_mask[x, y]
            and not fire_mask[x, y]
        )

    if not is_reachable(goal):
        return None

    def heuristic(pos: tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, start))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], int] = {start: 0}

    neighbours_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            node: tuple[int, int] | None = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if g > g_score.get(current, float("inf")):
            continue

        for dx, dy in neighbours_offsets:
            neighbour = (current[0] + dx, current[1] + dy)
            if not is_reachable(neighbour):
                continue
            new_g = g + 1
            if new_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = new_g
                came_from[neighbour] = current
                f = new_g + heuristic(neighbour)
                heapq.heappush(open_heap, (f, new_g, neighbour))

    return None


def get_nearest_frontier(
    pos: tuple[int, int],
    visited_cells: set[tuple[int, int]],
    grid_state: np.ndarray,
) -> tuple[int, int] | None:
    """
    Find nearest passable, non-FIRE, unvisited cell via BFS.

    Args:
        - pos: (x, y) current agent position.
        - visited_cells: set of (x, y) cells already visited.
        - grid_state: 2D int array of CellType values
          (PASSABLE=0, OBSTACLE=1, FIRE=2).

    Returns:
        - (x, y) of nearest frontier cell, or None if no frontier exists.
    """
    from src.environment.grid import CellType

    width, height = grid_state.shape
    queue: deque[tuple[int, int]] = deque([pos])
    seen: set[tuple[int, int]] = {pos}
    neighbours_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        current = queue.popleft()
        if current not in visited_cells:
            return current
        for dx, dy in neighbours_offsets:
            neighbour = (current[0] + dx, current[1] + dy)
            nx, ny = neighbour
            if (
                neighbour not in seen
                and 0 <= nx < width
                and 0 <= ny < height
                and grid_state[nx, ny] == CellType.PASSABLE.value
            ):
                seen.add(neighbour)
                queue.append(neighbour)

    return None

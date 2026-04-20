"""
planner.py
==========
Contains the A* pathfinding algorithm and the Robot agent class.

A* Algorithm
------------
A* is an informed search algorithm that combines:
    g(n)  – actual cost from start to node n
    h(n)  – heuristic estimate from n to goal  (Manhattan Distance here)
    f(n)  = g(n) + h(n)                        (priority in the open list)

Manhattan Distance heuristic is *admissible* (never over-estimates) on a
4-connected grid, so A* is guaranteed to return the optimal path.

Prioritized Planning (Conflict Resolution)
------------------------------------------
Robots are ranked by priority (lower index = higher priority).
High-priority robots plan first on the static grid.
Lower-priority robots receive a *reservation table* – a time-indexed set of
cells already claimed by higher-priority robots.  When planning, a lower-
priority robot treats those (time, cell) pairs as dynamic obstacles, so it
waits or reroutes rather than collide.

CS6053 LO4 – Rational Agent connection
---------------------------------------
Each Robot is a goal-based rational agent:
    • It perceives its environment (the grid + reservation table).
    • It deliberates using A* to find the utility-maximising action sequence.
    • Its optimality criterion is minimal path length (cost).
"""

import heapq
import time
from typing import Optional

from environment import WarehouseEnv


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan_distance(a: tuple, b: tuple) -> int:
    """Return the Manhattan distance between two (row, col) points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Reservation Table  (for Prioritized Planning)
# ---------------------------------------------------------------------------

class ReservationTable:
    """
    Records which (time_step, row, col) cells are already claimed by
    higher-priority robots, so that lower-priority robots avoid them.
    """

    def __init__(self):
        # Set of (time_step, row, col) tuples
        self._reserved: set = set()

    def reserve(self, path: list):
        """
        Register every cell along *path* at the corresponding time step.
        The robot is assumed to remain at its final cell for extra time steps
        (to avoid tail collisions).

        Parameters
        ----------
        path : list of (row, col)
            Ordered list of positions from start to goal.
        """
        if not path:
            return
        for t, cell in enumerate(path):
            self._reserved.add((t, cell[0], cell[1]))
        # Reserve the goal cell for extra time to prevent follow-on collisions
        goal = path[-1]
        for extra in range(1, len(path) + 1):
            self._reserved.add((len(path) - 1 + extra, goal[0], goal[1]))

    def is_reserved(self, time_step: int, row: int, col: int) -> bool:
        """Return True if the cell is already claimed at this time step."""
        return (time_step, row, col) in self._reserved


# ---------------------------------------------------------------------------
# A* implementation (space–time aware)
# ---------------------------------------------------------------------------

def astar(env: WarehouseEnv,
          start: tuple,
          goal: tuple,
          reservation: Optional[ReservationTable] = None,
          max_time: int = 500) -> Optional[list]:
    """
    Find the shortest collision-free path from *start* to *goal* using A*.

    The search operates in (time, row, col) space so that it can respect the
    reservation table produced by higher-priority robots.

    Parameters
    ----------
    env : WarehouseEnv
        The warehouse grid.
    start : (row, col)
        Starting cell.
    goal : (row, col)
        Target cell.
    reservation : ReservationTable or None
        If provided, the planner avoids (time, cell) entries in the table.
    max_time : int
        Hard cut-off on time steps to prevent infinite loops in crowded grids.

    Returns
    -------
    list of (row, col) or None
        Ordered path from start to goal (inclusive), or None if unreachable.
    """
    if not env.is_valid(start[0], start[1]):
        return None
    if not env.is_valid(goal[0], goal[1]):
        return None

    # Priority queue entries: (f, g, time_step, position, parent)
    # Using a counter as tie-breaker to keep the heap stable
    counter = 0
    open_heap = []

    # State: (time_step, row, col)
    start_state = (0, start[0], start[1])
    heapq.heappush(open_heap, (0, 0, counter, start_state, None))

    # came_from maps state → parent_state
    came_from = {start_state: None}
    # g_cost maps state → cost-so-far
    g_cost = {start_state: 0}

    while open_heap:
        # Heap entry: (f, g, tie_breaker, state, parent_state)
        f, g, _tie, current_state, _parent = heapq.heappop(open_heap)
        t, r, c = current_state

        # Goal test (position only – time doesn't matter at goal)
        if (r, c) == goal:
            return _reconstruct_path(came_from, current_state)

        if t >= max_time:
            continue

        # Expand neighbours (move to adjacent cell or wait in place)
        neighbours = list(env.get_neighbors(r, c)) + [(r, c)]  # include wait
        for nr, nc in neighbours:
            nt = t + 1
            # Skip if another high-priority robot has reserved this cell
            if reservation and reservation.is_reserved(nt, nr, nc):
                continue

            new_state = (nt, nr, nc)
            new_g = g + 1  # uniform cost (each step = 1)

            if new_state not in g_cost or new_g < g_cost[new_state]:
                g_cost[new_state] = new_g
                h = manhattan_distance((nr, nc), goal)
                new_f = new_g + h
                counter += 1
                heapq.heappush(open_heap,
                               (new_f, new_g, counter, new_state, current_state))
                came_from[new_state] = current_state

    return None   # No path found


def _reconstruct_path(came_from: dict, final_state: tuple) -> list:
    """Walk back through the came_from dict to build the (row, col) path."""
    path = []
    state = final_state
    while state is not None:
        _, r, c = state
        path.append((r, c))
        state = came_from[state]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Robot  –  the rational agent
# ---------------------------------------------------------------------------

class Robot:
    """
    A goal-based rational agent that navigates the warehouse using A*.

    Parameters
    ----------
    robot_id : int
        Unique identifier (also determines priority – lower id = higher priority).
    start : (row, col)
        Initial position on the grid.
    goal : (row, col)
        Target cell the robot must reach.
    env : WarehouseEnv
        Shared environment reference.
    """

    def __init__(self, robot_id: int, start: tuple,
                 goal: tuple, env: WarehouseEnv):
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.env = env

        self.path: list = []          # Planned (row, col) path
        self.path_length: int = 0     # Number of steps
        self.planning_time: float = 0.0   # Wall-clock seconds for A*

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, reservation: Optional[ReservationTable] = None):
        """
        Run A* to compute the path from start to goal, respecting the
        reservation table of higher-priority robots.

        Side-effects
        ------------
        Sets self.path, self.path_length, self.planning_time.

        Returns
        -------
        list of (row, col) or None
        """
        t_start = time.perf_counter()
        self.path = astar(self.env, self.start, self.goal, reservation)
        self.planning_time = time.perf_counter() - t_start

        if self.path is not None:
            self.path_length = len(self.path) - 1   # steps = nodes - 1
        else:
            self.path_length = -1   # unreachable

        return self.path

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"Robot(id={self.robot_id}, "
                f"start={self.start}, goal={self.goal}, "
                f"path_len={self.path_length})")


# ---------------------------------------------------------------------------
# Prioritized planner  –  orchestrates all robots
# ---------------------------------------------------------------------------

def prioritized_plan(robots: list, env: WarehouseEnv):
    """
    Run Prioritized Planning over *robots* (sorted by robot_id ascending,
    so robot 0 is highest priority).

    Algorithm
    ---------
    1. Sort robots by id (priority order).
    2. Maintain a shared ReservationTable.
    3. For each robot (high → low priority):
       a. Plan with A*, consulting the reservation table.
       b. If a path is found, register it in the reservation table.

    Parameters
    ----------
    robots : list[Robot]
        All robots to plan for.
    env : WarehouseEnv
        The shared environment.

    Returns
    -------
    dict mapping robot_id → {'path', 'path_length', 'planning_time'}
    """
    reservation = ReservationTable()
    results = {}

    # Sort by priority (robot_id ascending)
    for robot in sorted(robots, key=lambda r: r.robot_id):
        robot.plan(reservation)

        if robot.path:
            reservation.reserve(robot.path)

        results[robot.robot_id] = {
            "path": robot.path,
            "path_length": robot.path_length,
            "planning_time": robot.planning_time,
        }

    return results

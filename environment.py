"""
environment.py
==============
Defines the WarehouseEnv class that models a grid-based warehouse.

Grid encoding
-------------
    0  →  navigable floor cell
    1  →  shelf (static obstacle – robots cannot enter)

CS6053 LO3 / Intelligent Behaviour context
-------------------------------------------
The environment is the *percept source* for every rational agent.  By keeping
the grid representation clean and centralised here, each Robot (planner.py)
can query it as part of its PEAS model:
    Performance measure  – shortest, collision-free path
    Environment          – this 2-D grid
    Actuators            – move commands (N / S / E / W)
    Sensors              – current position + grid view
"""

import numpy as np
import random


class WarehouseEnv:
    """
    A 2-D grid warehouse with randomly placed shelf obstacles.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    obstacle_density : float
        Fraction of cells that should be shelves (0.0 – 1.0).
        Default is 0.20 (20 % of cells are shelves).
    seed : int or None
        Random seed for reproducible layouts.
    """

    def __init__(self, rows: int = 15, cols: int = 15,
                 obstacle_density: float = 0.20, seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density

        # Reproducible random layout
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Start with all floor cells
        self.grid = np.zeros((rows, cols), dtype=np.int8)

        # Randomly place shelves while keeping enough open space
        num_obstacles = int(rows * cols * obstacle_density)
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        obstacle_cells = rng.sample(all_cells, num_obstacles)
        for r, c in obstacle_cells:
            self.grid[r][c] = 1

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_valid(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid and not a shelf."""
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] == 0)

    def get_neighbors(self, row: int, col: int):
        """
        Yield the four cardinal-direction neighbours of a cell that are
        navigable (floor cells inside the grid boundary).
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid(nr, nc):
                yield (nr, nc)

    def find_free_cells(self):
        """Return a list of all floor cells as (row, col) tuples."""
        return [(r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r][c] == 0]

    def __repr__(self):
        return (f"WarehouseEnv({self.rows}×{self.cols}, "
                f"{int(self.obstacle_density * 100)}% shelves)")

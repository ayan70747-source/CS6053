"""Warehouse environment with aisle-like shelf layout and pick-task support."""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


FLOOR = 0
SHELF = 1
PACKING_STATION = 2


class WarehouseGrid:
    """Grid-based warehouse where 0=floor, 1=shelf, 2=packing station."""

    def __init__(self, rows: int = 18, cols: int = 24, seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.packing_station = (rows - 2, cols // 2)
        self._build_aisles()
        self.grid[self.packing_station] = PACKING_STATION

    def _build_aisles(self) -> None:
        """Create shelf islands with clear horizontal and vertical aisles."""
        self.grid.fill(FLOOR)
        self.grid[[0, self.rows - 1], :] = FLOOR
        self.grid[:, [0, self.cols - 1]] = FLOOR

        for c in range(2, self.cols - 2, 4):
            for r in range(2, self.rows - 3):
                if r % 6 in (0, 1):
                    continue
                self.grid[r, c] = SHELF
                if c + 1 < self.cols - 1:
                    self.grid[r, c + 1] = SHELF

        station_r, station_c = self.packing_station
        for rr in range(max(0, station_r - 1), min(self.rows, station_r + 2)):
            for cc in range(max(0, station_c - 2), min(self.cols, station_c + 3)):
                if self.grid[rr, cc] == SHELF:
                    self.grid[rr, cc] = FLOOR

    def is_walkable(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row, col] in (FLOOR, PACKING_STATION)

    def neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_walkable(nr, nc):
                out.append((nr, nc))
        return out

    def floor_cells(self) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] in (FLOOR, PACKING_STATION)
        ]

    def pickable_cells(self) -> List[Tuple[int, int]]:
        """Return floor cells adjacent to at least one shelf (good pick points)."""
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] not in (FLOOR, PACKING_STATION):
                    continue
                for nr, nc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + nr, c + nc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols and self.grid[rr, cc] == SHELF:
                        cells.append((r, c))
                        break
        return [cell for cell in cells if cell != self.packing_station]

    def generate_random_pick_tasks(self, num_tasks: int, rng: random.Random) -> List[Tuple[int, int]]:
        """Generate unique random pick targets from aisle-adjacent floor cells."""
        candidates = self.pickable_cells()
        if len(candidates) < num_tasks:
            raise ValueError(
                f"Not enough pickable cells ({len(candidates)}) for {num_tasks} tasks."
            )
        return rng.sample(candidates, num_tasks)

    def __repr__(self) -> str:
        return f"WarehouseGrid(rows={self.rows}, cols={self.cols}, packing={self.packing_station})"

"""Rational A* planner with prioritized planning and time-space occupancy."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent import RationalAgent
from environment import WarehouseGrid


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ReservationTable:
    """Time-space occupancy for vertex and edge reservations."""

    def __init__(self) -> None:
        self.vertex_occ: set[Tuple[int, int, int]] = set()
        self.edge_occ: set[Tuple[int, int, int, int, int]] = set()

    def reserve_path(self, path: List[Tuple[int, int]], start_time: int = 0, tail_hold: int = 8) -> None:
        if not path:
            return

        for i, (r, c) in enumerate(path):
            t = start_time + i
            self.vertex_occ.add((t, r, c))
            if i > 0:
                pr, pc = path[i - 1]
                self.edge_occ.add((t - 1, pr, pc, r, c))

        gr, gc = path[-1]
        end_t = start_time + len(path) - 1
        for extra_t in range(1, tail_hold + 1):
            self.vertex_occ.add((end_t + extra_t, gr, gc))

    def is_vertex_reserved(self, t: int, r: int, c: int) -> bool:
        return (t, r, c) in self.vertex_occ

    def is_edge_conflict(self, t: int, r1: int, c1: int, r2: int, c2: int) -> bool:
        return (t, r2, c2, r1, c1) in self.edge_occ


@dataclass
class PlanResult:
    path: List[Tuple[int, int]]
    blocked_by_reservation: int
    wait_steps_in_path: int
    wait_better_than_detour: int


class RationalAStarPlanner:
    """Space-time A* planner with transition model (x,y,t)->(x',y',t+1)."""

    def __init__(self, max_time: int = 600):
        self.max_time = max_time

    def plan(
        self,
        env: WarehouseGrid,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        reservation: ReservationTable,
        start_time: int = 0,
    ) -> Optional[PlanResult]:
        if not env.is_walkable(*start) or not env.is_walkable(*goal):
            return None

        start_state = (start_time, start[0], start[1])
        open_heap: List[Tuple[int, int, int, Tuple[int, int, int]]] = []
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_state: None}
        g_cost: Dict[Tuple[int, int, int], int] = {start_state: 0}

        blocked_by_reservation = 0
        tie = 0
        start_h = manhattan_distance(start, goal)
        heapq.heappush(open_heap, (start_h, 0, tie, start_state))

        while open_heap:
            _, cur_g, _, (t, r, c) = heapq.heappop(open_heap)

            if (r, c) == goal:
                path = self._reconstruct_path(came_from, (t, r, c))
                waits, wait_better = self._evaluate_wait_decisions(path, goal)
                return PlanResult(
                    path=path,
                    blocked_by_reservation=blocked_by_reservation,
                    wait_steps_in_path=waits,
                    wait_better_than_detour=wait_better,
                )

            if t - start_time >= self.max_time:
                continue

            candidate_moves = env.neighbors(r, c) + [(r, c)]
            for nr, nc in candidate_moves:
                nt = t + 1

                if reservation.is_vertex_reserved(nt, nr, nc):
                    blocked_by_reservation += 1
                    continue
                if reservation.is_edge_conflict(t, r, c, nr, nc):
                    blocked_by_reservation += 1
                    continue

                nxt = (nt, nr, nc)
                ng = cur_g + 1
                if nxt not in g_cost or ng < g_cost[nxt]:
                    g_cost[nxt] = ng
                    h = manhattan_distance((nr, nc), goal)
                    f = ng + h
                    tie += 1
                    heapq.heappush(open_heap, (f, ng, tie, nxt))
                    came_from[nxt] = (t, r, c)

        return None

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]],
        end_state: Tuple[int, int, int],
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int, int]] = end_state
        while cur is not None:
            _, r, c = cur
            path.append((r, c))
            cur = came_from[cur]
        path.reverse()
        return path

    @staticmethod
    def _evaluate_wait_decisions(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> Tuple[int, int]:
        """Count waits and whether waiting improved immediate f(n) vs moving away."""
        wait_steps = 0
        wait_better = 0

        for i in range(1, len(path)):
            prev = path[i - 1]
            cur = path[i]
            if prev != cur:
                continue

            wait_steps += 1
            wait_h = manhattan_distance(cur, goal)

            detour_h = float("inf")
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cur[0] + dr, cur[1] + dc
                detour_h = min(detour_h, manhattan_distance((nr, nc), goal))

            if wait_h <= detour_h:
                wait_better += 1

        return wait_steps, wait_better


def prioritized_planning(
    env: WarehouseGrid,
    agents: List[RationalAgent],
    planner: RationalAStarPlanner,
) -> Dict[str, object]:
    """Plan two-phase tasks with priority by agent_id and shared reservations."""
    reservation = ReservationTable()
    paths_by_agent: Dict[int, List[Tuple[int, int]]] = {}
    stats = {
        "collisions_avoided": 0,
        "wait_steps": 0,
        "wait_better_than_detour": 0,
    }

    for agent in sorted(agents, key=lambda a: a.agent_id):
        if agent.pick_target is None or agent.delivery_target is None:
            raise ValueError(f"Agent {agent.agent_id} missing task assignment.")

        first_leg = planner.plan(
            env=env,
            start=agent.current_pos,
            goal=agent.pick_target,
            reservation=reservation,
            start_time=0,
        )
        if first_leg is None:
            raise RuntimeError(f"No route to pick target for agent {agent.agent_id}.")

        second_leg = planner.plan(
            env=env,
            start=first_leg.path[-1],
            goal=agent.delivery_target,
            reservation=reservation,
            start_time=len(first_leg.path) - 1,
        )
        if second_leg is None:
            raise RuntimeError(f"No route to delivery target for agent {agent.agent_id}.")

        full_path = first_leg.path + second_leg.path[1:]
        agent.set_paths(first_leg.path, second_leg.path)
        paths_by_agent[agent.agent_id] = full_path

        reservation.reserve_path(full_path, start_time=0)

        stats["collisions_avoided"] += (
            first_leg.blocked_by_reservation + second_leg.blocked_by_reservation
        )
        stats["wait_steps"] += first_leg.wait_steps_in_path + second_leg.wait_steps_in_path
        stats["wait_better_than_detour"] += (
            first_leg.wait_better_than_detour + second_leg.wait_better_than_detour
        )

    return {
        "paths": paths_by_agent,
        "reservation": reservation,
        "stats": stats,
    }

"""Main MAPF simulation runner for CS6053 Topic 8 Action Planning."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from agent import AgentState, RationalAgent
from environment import FLOOR, PACKING_STATION, SHELF, WarehouseGrid
from planner import RationalAStarPlanner, prioritized_planning


SEED = 42
SCENARIOS = [2, 5, 10]
FRAME_INTERVAL_MS = 220
COLORS = [
        "#0ea5e9",
        "#ef4444",
        "#22c55e",
        "#f59e0b",
        "#8b5cf6",
        "#ec4899",
        "#14b8a6",
        "#eab308",
        "#f97316",
        "#84cc16",
]


def pick_agent_starts(env: WarehouseGrid, n_agents: int, rng: random.Random) -> List[Tuple[int, int]]:
        """Place robots near the station while keeping unique cells."""
        sr, sc = env.packing_station
        candidates: List[Tuple[int, int]] = []
        for r in range(max(0, sr - 4), min(env.rows, sr + 5)):
                for c in range(max(0, sc - 6), min(env.cols, sc + 7)):
                        if env.is_walkable(r, c):
                                candidates.append((r, c))

        unique = list(dict.fromkeys(candidates))
        if len(unique) < n_agents:
                raise ValueError("Not enough start cells near the packing station.")
        return rng.sample(unique, n_agents)


def simulate_execution(agents: List[RationalAgent]) -> Tuple[int, Dict[int, List[Tuple[int, int]]], bool]:
        """Execute planned paths tick by tick and evaluate goal test."""
        all_paths = {a.agent_id: list(a.planned_path) for a in agents}
        horizon = max(len(p) for p in all_paths.values())
        timeline: Dict[int, List[Tuple[int, int]]] = {
                aid: path + [path[-1]] * (horizon - len(path)) for aid, path in all_paths.items()
        }

        for t in range(horizon):
                for agent in agents:
                        agent.current_pos = timeline[agent.agent_id][t]
                        if t > 0 and timeline[agent.agent_id][t] != timeline[agent.agent_id][t - 1]:
                                agent.total_steps_taken += 1
                        agent.update_state_from_position()

        goal_met = all(
                a.state == AgentState.IDLE and a.current_pos == a.delivery_target for a in agents
        )
        return horizon - 1, timeline, goal_met


def run_scenario(num_agents: int, seed_offset: int = 0) -> Dict[str, object]:
        rng = random.Random(SEED + seed_offset + num_agents)
        env = WarehouseGrid(rows=18, cols=24, seed=SEED)
        pick_tasks = env.generate_random_pick_tasks(num_agents, rng)
        starts = pick_agent_starts(env, num_agents, rng)

        agents: List[RationalAgent] = []
        for i in range(num_agents):
                agent = RationalAgent(agent_id=i, start_pos=starts[i])
                agent.assign_task(pick_target=pick_tasks[i], delivery_target=env.packing_station)
                agents.append(agent)

        planner = RationalAStarPlanner(max_time=700)
        t0 = time.perf_counter()
        plan_out = prioritized_planning(env=env, agents=agents, planner=planner)
        planning_ms = (time.perf_counter() - t0) * 1000

        total_steps, timeline, goal_met = simulate_execution(agents)
        if not goal_met:
                raise RuntimeError("Goal test failed: not all agents returned to packing station.")

        path_lengths = [len(a.planned_path) - 1 for a in agents]
        avg_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0

        return {
                "env": env,
                "agents": agents,
                "paths": plan_out["paths"],
                "timeline": timeline,
                "total_steps": total_steps,
                "average_path_length": avg_path,
                "collisions_avoided": plan_out["stats"]["collisions_avoided"],
                "wait_steps": plan_out["stats"]["wait_steps"],
                "wait_better_than_detour": plan_out["stats"]["wait_better_than_detour"],
                "computation_time_ms": planning_ms,
        }


def build_animation_frames(timeline: Dict[int, List[Tuple[int, int]]], n_agents: int) -> List[List[Tuple[int, int]]]:
        horizon = len(next(iter(timeline.values())))
        return [[timeline[agent_id][t] for agent_id in range(n_agents)] for t in range(horizon)]


def create_animation(
        env: WarehouseGrid,
        agents: List[RationalAgent],
        timeline: Dict[int, List[Tuple[int, int]]],
        title: str,
):
        frames = build_animation_frames(timeline, len(agents))

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(-0.5, env.rows - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        for r in range(env.rows + 1):
                ax.axhline(r - 0.5, color="#d1d5db", linewidth=0.5, zorder=0)
        for c in range(env.cols + 1):
                ax.axvline(c - 0.5, color="#d1d5db", linewidth=0.5, zorder=0)

        for r in range(env.rows):
                for c in range(env.cols):
                        if env.grid[r, c] == SHELF:
                                ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, color="#8b5a2b", zorder=2))
                        elif env.grid[r, c] == PACKING_STATION:
                                ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, color="#facc15", zorder=1, alpha=0.6))

        for agent in agents:
                color = COLORS[agent.agent_id % len(COLORS)]
                pr, pc = agent.pick_target
                ax.plot(pc, pr, marker="*", color=color, markersize=12, markeredgecolor="black", zorder=4)

        circles = []
        labels = []
        for agent in agents:
                color = COLORS[agent.agent_id % len(COLORS)]
                r0, c0 = frames[0][agent.agent_id]
                circle = plt.Circle((c0, r0), radius=0.33, facecolor=color, edgecolor="black", linewidth=1.4, zorder=5)
                ax.add_patch(circle)
                circles.append(circle)
                label = ax.text(c0, r0, str(agent.agent_id), ha="center", va="center", color="white", fontsize=7, zorder=6)
                labels.append(label)

        frame_text = ax.text(
                0.01,
                0.99,
                "t = 0",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        legend_handles = [
                mpatches.Patch(color=COLORS[a.agent_id % len(COLORS)], label=f"A{a.agent_id} start={a.start_pos} pick={a.pick_target}")
                for a in agents
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8)

        def update(frame_idx: int):
                for i, (circle, label) in enumerate(zip(circles, labels)):
                        rr, cc = frames[frame_idx][i]
                        circle.center = (cc, rr)
                        label.set_position((cc, rr))
                frame_text.set_text(f"t = {frame_idx}")
                return circles + labels + [frame_text]

        ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(frames),
                interval=FRAME_INTERVAL_MS,
                blit=True,
                repeat=True,
        )
        plt.tight_layout()
        return fig, ani


def main() -> None:
        print("=" * 70)
        print("CS6053 MAPF Warehouse Simulation (Rational A* + Prioritized Planning)")
        print("=" * 70)

        rows = []
        scenario_outputs = {}

        for num_agents in SCENARIOS:
                print(f"Running scenario with {num_agents} agents...")
                out = run_scenario(num_agents=num_agents)
                scenario_outputs[num_agents] = out

                rows.append(
                        {
                                "Num_Agents": num_agents,
                                "Total_Steps_Taken": out["total_steps"],
                                "Average_Path_Length": round(out["average_path_length"], 3),
                                "Collisions_Avoided": out["collisions_avoided"],
                                "Computation_Time_MS": round(out["computation_time_ms"], 3),
                        }
                )

                print(
                        f"  steps={out['total_steps']} avg_path={out['average_path_length']:.2f} "
                        f"collisions_avoided={out['collisions_avoided']} "
                        f"wait_eval={out['wait_better_than_detour']}/{out['wait_steps']} "
                        f"compute={out['computation_time_ms']:.2f}ms"
                )

        df = pd.DataFrame(rows)
        csv_path = "warehouse_performance.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved performance data to {csv_path}")
        print(df.to_string(index=False))

        demo_agents = 5
        demo = scenario_outputs[demo_agents]
        fig, ani = create_animation(
                env=demo["env"],
                agents=demo["agents"],
                timeline=demo["timeline"],
                title="Warehouse MAPF Action Planning (5 Agents)",
        )

        gif_path = "warehouse_mapf.gif"
        writer = animation.PillowWriter(fps=5)
        ani.save(gif_path, writer=writer)
        print(f"Saved animation to {gif_path}")
        plt.show()


if __name__ == "__main__":
        main()

"""
simulation.py
=============
Main script for the Multi-Agent Pathfinding (MAPF) warehouse simulation.

What this script does
---------------------
1. Creates a WarehouseEnv (grid with shelves).
2. Randomly assigns distinct start / goal positions to N robots.
3. Runs Prioritized Planning (via planner.py) to give every robot a
   collision-free A* path.
4. Collects performance metrics (path length, planning time) for N = 2, 5, 10.
5. Prints a metrics summary table to the console.
6. Produces a Matplotlib animation showing all robots moving simultaneously.

Usage
-----
    python simulation.py

To change the grid size or obstacle density, edit GRID_ROWS, GRID_COLS,
and OBSTACLE_DENSITY at the top of this file.
"""

import random
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

from environment import WarehouseEnv
from planner import Robot, prioritized_plan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_ROWS = 15
GRID_COLS = 15
OBSTACLE_DENSITY = 0.20    # 20 % of cells are shelves
SEED = 42                  # Reproducibility seed

# Scenario sizes to benchmark
AGENT_COUNTS = [2, 5, 10]

# Animation frame interval in milliseconds
FRAME_INTERVAL_MS = 200

# Colour palette for robots (cycles if more than 10 robots)
ROBOT_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]


# ---------------------------------------------------------------------------
# Helper: assign random, non-overlapping start / goal positions
# ---------------------------------------------------------------------------

def assign_positions(env: WarehouseEnv, n_robots: int,
                     rng: random.Random) -> list:
    """
    Randomly assign distinct start and goal cells to *n_robots* robots.

    Returns a list of (start, goal) tuples, one per robot.
    """
    free_cells = env.find_free_cells()
    if len(free_cells) < 2 * n_robots:
        raise ValueError(
            f"Not enough free cells ({len(free_cells)}) for "
            f"{n_robots} robots (need {2 * n_robots})."
        )

    sampled = rng.sample(free_cells, 2 * n_robots)
    positions = []
    for i in range(n_robots):
        start = sampled[i]
        goal = sampled[n_robots + i]
        positions.append((start, goal))
    return positions


# ---------------------------------------------------------------------------
# Core simulation run for a given number of agents
# ---------------------------------------------------------------------------

def run_scenario(n_robots: int, env: WarehouseEnv) -> dict:
    """
    Set up and solve the MAPF problem for *n_robots* agents.

    Returns
    -------
    dict with keys:
        'robots'   – list[Robot]
        'results'  – output of prioritized_plan()
        'total_planning_time'  – wall-clock seconds
    """
    rng = random.Random(SEED + n_robots)   # Different seed per scenario size
    positions = assign_positions(env, n_robots, rng)

    robots = [
        Robot(robot_id=i, start=positions[i][0],
              goal=positions[i][1], env=env)
        for i in range(n_robots)
    ]

    t0 = time.perf_counter()
    results = prioritized_plan(robots, env)
    total_time = time.perf_counter() - t0

    return {
        "robots": robots,
        "results": results,
        "total_planning_time": total_time,
    }


# ---------------------------------------------------------------------------
# Console metrics summary
# ---------------------------------------------------------------------------

def print_metrics(scenario_data: dict):
    """Print a formatted table of performance metrics."""
    print("\n" + "=" * 60)
    print(f"  MAPF Metrics  –  {len(scenario_data['robots'])} Agents")
    print("=" * 60)
    print(f"{'Robot':<8} {'Start':<12} {'Goal':<12} "
          f"{'Path Len':>10} {'Plan Time (ms)':>16}")
    print("-" * 60)

    for robot in scenario_data["robots"]:
        rid = robot.robot_id
        result = scenario_data["results"][rid]
        path_len = result["path_length"]
        plan_ms = result["planning_time"] * 1000

        path_str = str(path_len) if path_len >= 0 else "UNREACHABLE"
        print(f"  R{rid:<5} {str(robot.start):<12} {str(robot.goal):<12} "
              f"{path_str:>10} {plan_ms:>14.3f}")

    print("-" * 60)
    total_ms = scenario_data["total_planning_time"] * 1000
    print(f"  Total planning wall-clock time: {total_ms:.3f} ms")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

def build_frame_positions(results: dict, n_robots: int) -> list:
    """
    Convert per-robot paths into a list of frames.

    Each frame is a list of (row, col) positions, one per robot.
    Robots that have reached their goal stay at the goal cell.

    Returns
    -------
    list[list[(row, col)]]  –  indexed by [frame][robot_id]
    """
    paths = [results[i]["path"] or [] for i in range(n_robots)]

    # Pad short paths: robot stays at its last known position
    max_len = max((len(p) for p in paths), default=1)
    padded = []
    for path in paths:
        if path:
            # Extend with goal cell repeated
            extension = [path[-1]] * (max_len - len(path))
            padded.append(path + extension)
        else:
            # Robot has no path – stays at start (or [0,0] as fallback)
            padded.append([(0, 0)] * max_len)

    # Transpose: frames × robots
    frames = []
    for t in range(max_len):
        frames.append([padded[i][t] for i in range(n_robots)])

    return frames


def create_animation(env: WarehouseEnv, scenario_data: dict,
                     title: str = "Warehouse MAPF Simulation"):
    """
    Build and display/save a Matplotlib animation of the MAPF solution.

    Parameters
    ----------
    env : WarehouseEnv
    scenario_data : dict   returned by run_scenario()
    title : str
    """
    robots = scenario_data["robots"]
    results = scenario_data["results"]
    n_robots = len(robots)

    frames = build_frame_positions(results, n_robots)

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()   # Row 0 at the top (natural grid orientation)

    # Draw grid lines
    for r in range(env.rows + 1):
        ax.axhline(r - 0.5, color="lightgrey", linewidth=0.5)
    for c in range(env.cols + 1):
        ax.axvline(c - 0.5, color="lightgrey", linewidth=0.5)

    # Draw shelves
    shelf_colour = "#8B6914"
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] == 1:
                rect = plt.Rectangle(
                    (c - 0.45, r - 0.45), 0.90, 0.90,
                    color=shelf_colour, zorder=2
                )
                ax.add_patch(rect)

    # Draw goal markers (stars)
    for robot in robots:
        colour = ROBOT_COLOURS[robot.robot_id % len(ROBOT_COLOURS)]
        gr, gc = robot.goal
        ax.plot(gc, gr, marker="*", markersize=14,
                color=colour, zorder=3, alpha=0.7,
                markeredgecolor="black", markeredgewidth=0.5)

    # Draw start markers (squares)
    for robot in robots:
        colour = ROBOT_COLOURS[robot.robot_id % len(ROBOT_COLOURS)]
        sr, sc = robot.start
        ax.plot(sc, sr, marker="s", markersize=10,
                color=colour, zorder=3, alpha=0.5,
                markeredgecolor="black", markeredgewidth=0.5)

    # Animated robot circles
    robot_circles = []
    for robot in robots:
        colour = ROBOT_COLOURS[robot.robot_id % len(ROBOT_COLOURS)]
        r0, c0 = frames[0][robot.robot_id]
        circle = plt.Circle((c0, r0), 0.35, facecolor=colour,
                             zorder=5, linewidth=1.5,
                             edgecolor="black")
        ax.add_patch(circle)
        robot_circles.append(circle)

    # Robot ID labels (move with circles)
    robot_labels = []
    for robot in robots:
        r0, c0 = frames[0][robot.robot_id]
        lbl = ax.text(c0, r0, str(robot.robot_id),
                      ha="center", va="center",
                      fontsize=7, fontweight="bold",
                      color="white", zorder=6)
        robot_labels.append(lbl)

    # Frame counter text
    frame_text = ax.text(
        0.01, 0.99, "t = 0",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
    )

    # Legend
    legend_handles = []
    for robot in robots:
        colour = ROBOT_COLOURS[robot.robot_id % len(ROBOT_COLOURS)]
        result = results[robot.robot_id]
        pl = result["path_length"]
        pt = result["planning_time"] * 1000
        label = (f"R{robot.robot_id}: {robot.start}→{robot.goal}  "
                 f"len={pl}  ({pt:.1f}ms)")
        legend_handles.append(mpatches.Patch(color=colour, label=label))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=6.5, framealpha=0.85,
              bbox_to_anchor=(1.0, -0.02), ncol=1)

    # ── Update function ────────────────────────────────────────────────────
    def update(frame_idx):
        positions = frames[frame_idx]
        for i, (circle, lbl) in enumerate(
                zip(robot_circles, robot_labels)):
            r, c = positions[i]
            circle.center = (c, r)
            lbl.set_position((c, r))
        frame_text.set_text(f"t = {frame_idx}")
        return robot_circles + robot_labels + [frame_text]

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=FRAME_INTERVAL_MS,
        blit=True,
        repeat=True
    )

    plt.tight_layout()
    return fig, ani


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Multi-Agent Pathfinding (MAPF) – Warehouse Simulation")
    print("  CS6053 Rational Agents Project")
    print("=" * 60)

    # Shared environment for all scenarios
    env = WarehouseEnv(rows=GRID_ROWS, cols=GRID_COLS,
                       obstacle_density=OBSTACLE_DENSITY, seed=SEED)
    print(f"\nEnvironment: {env}")

    # ── Run all scenarios and collect metrics ──────────────────────────────
    all_scenarios = {}
    for n in AGENT_COUNTS:
        print(f"\nRunning scenario: {n} agents …", end=" ", flush=True)
        data = run_scenario(n, env)
        all_scenarios[n] = data
        print("done.")
        print_metrics(data)

    # ── Summary comparison table ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Scalability Summary")
    print("=" * 60)
    print(f"{'Agents':<10} {'Avg Path Len':>14} {'Total Plan (ms)':>18}")
    print("-" * 44)
    for n, data in all_scenarios.items():
        results = data["results"]
        valid_paths = [r["path_length"] for r in results.values()
                       if r["path_length"] >= 0]
        avg_len = sum(valid_paths) / len(valid_paths) if valid_paths else 0
        total_ms = data["total_planning_time"] * 1000
        print(f"  {n:<8} {avg_len:>14.2f} {total_ms:>16.3f}")
    print("=" * 60)

    # ── Animation (uses the 5-agent scenario for the demo) ─────────────────
    demo_n = 5
    print(f"\nGenerating animation for {demo_n}-agent scenario …")
    demo_data = all_scenarios[demo_n]

    # Use non-interactive backend if running headless
    try:
        fig, ani = create_animation(
            env, demo_data,
            title=f"Warehouse MAPF – {demo_n} Agents (A* + Prioritized Planning)"
        )

        # Save animation as GIF
        output_file = "warehouse_mapf.gif"
        print(f"Saving animation to '{output_file}' …", end=" ", flush=True)
        writer = animation.PillowWriter(fps=5)
        ani.save(output_file, writer=writer)
        print("saved.")

        plt.show()

    except Exception as exc:
        print(f"[Warning] Animation could not be displayed: {exc}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()

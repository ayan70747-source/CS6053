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
import json

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
OBSTACLE_DENSITY = 0.20    # 20% of cells are shelves
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


def export_interactive_html(env: WarehouseEnv, scenario_data: dict,
                                                        output_file: str = "warehouse_mapf_interactive.html",
                                                        title: str = "Warehouse MAPF Interactive Viewer"):
        """
        Export a browser-based interactive animation with playback controls.

        Controls include play/pause, frame stepping, reset, a timeline scrubber,
        and speed adjustment.
        """
        robots = scenario_data["robots"]
        results = scenario_data["results"]
        n_robots = len(robots)
        frames = build_frame_positions(results, n_robots)

        data = {
                "rows": env.rows,
                "cols": env.cols,
                "grid": env.grid.tolist(),
                "frames": frames,
                "robots": [
                        {
                                "id": robot.robot_id,
                                "start": list(robot.start),
                                "goal": list(robot.goal),
                                "path_length": results[robot.robot_id]["path_length"],
                                "planning_ms": round(results[robot.robot_id]["planning_time"] * 1000, 3),
                                "color": ROBOT_COLOURS[robot.robot_id % len(ROBOT_COLOURS)],
                        }
                        for robot in robots
                ],
                "title": title,
        }

        template = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Warehouse MAPF Interactive Viewer</title>
    <style>
        :root {
            --bg: #f8f4ea;
            --panel: #fffef8;
            --ink: #1f2937;
            --accent: #0f766e;
            --grid: #d1d5db;
            --shelf: #8b6914;
        }
        body {
            margin: 0;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            background: radial-gradient(circle at 10% 10%, #fff6e1 0%, var(--bg) 45%, #efe8d5 100%);
            color: var(--ink);
        }
        .wrap {
            max-width: 1100px;
            margin: 1.2rem auto;
            padding: 0 1rem;
            display: grid;
            gap: 0.9rem;
            grid-template-columns: 1fr;
        }
        .panel {
            background: var(--panel);
            border: 1px solid #d6d3d1;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            padding: 0.8rem;
        }
        .topbar {
            display: flex;
            gap: 0.8rem;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .title {
            margin: 0;
            font-size: 1.1rem;
            letter-spacing: 0.2px;
        }
        canvas {
            width: 100%;
            height: auto;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            background: #f3f4f6;
            display: block;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.5rem;
            align-items: center;
            margin-top: 0.8rem;
        }
        button {
            border: 1px solid #0f766e;
            background: #f0fdfa;
            color: #134e4a;
            border-radius: 8px;
            padding: 0.5rem 0.6rem;
            font-weight: 700;
            cursor: pointer;
        }
        button:hover { filter: brightness(0.98); }
        input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
        }
        .legend {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.35rem 0.75rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .swatch {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1px solid #111827;
            flex-shrink: 0;
        }
        .stat {
            font-variant-numeric: tabular-nums;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="panel">
            <div class="topbar">
                <h1 class="title" id="title"></h1>
                <div class="stat" id="frameLabel"></div>
            </div>
            <canvas id="simCanvas" width="900" height="900"></canvas>
            <div class="controls">
                <button id="playPauseBtn">Pause</button>
                <button id="prevBtn">Step -1</button>
                <button id="nextBtn">Step +1</button>
                <button id="resetBtn">Reset</button>
                <label>Speed (fps)
                    <input id="fpsSlider" type="range" min="1" max="15" value="5" />
                </label>
                <label>Timeline
                    <input id="frameSlider" type="range" min="0" max="0" value="0" />
                </label>
            </div>
            <div class="legend" id="legend"></div>
        </div>
    </div>

    <script>
        const data = __DATA_JSON__;

        const canvas = document.getElementById("simCanvas");
        const ctx = canvas.getContext("2d");
        const titleEl = document.getElementById("title");
        const frameLabelEl = document.getElementById("frameLabel");
        const legendEl = document.getElementById("legend");

        const playPauseBtn = document.getElementById("playPauseBtn");
        const prevBtn = document.getElementById("prevBtn");
        const nextBtn = document.getElementById("nextBtn");
        const resetBtn = document.getElementById("resetBtn");
        const fpsSlider = document.getElementById("fpsSlider");
        const frameSlider = document.getElementById("frameSlider");

        titleEl.textContent = data.title;
        frameSlider.max = String(data.frames.length - 1);

        for (const r of data.robots) {
            const item = document.createElement("div");
            item.className = "legend-item";
            item.innerHTML = `<span class="swatch" style="background:${r.color}"></span>` +
                                             `R${r.id}: ${r.start} -> ${r.goal} | len=${r.path_length} | ${r.planning_ms}ms`;
            legendEl.appendChild(item);
        }

        let frame = 0;
        let playing = true;
        let fps = Number(fpsSlider.value);
        let timer = null;

        function starPath(cx, cy, outerR, innerR, points) {
            let angle = -Math.PI / 2;
            const step = Math.PI / points;
            ctx.beginPath();
            for (let i = 0; i < points * 2; i++) {
                const r = i % 2 === 0 ? outerR : innerR;
                const x = cx + Math.cos(angle) * r;
                const y = cy + Math.sin(angle) * r;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                angle += step;
            }
            ctx.closePath();
        }

        function draw() {
            const rows = data.rows;
            const cols = data.cols;
            const w = canvas.width;
            const h = canvas.height;
            const cell = Math.min(w / cols, h / rows);

            ctx.clearRect(0, 0, w, h);

            // Floor
            ctx.fillStyle = "#f9fafb";
            ctx.fillRect(0, 0, cols * cell, rows * cell);

            // Shelves
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    if (data.grid[r][c] === 1) {
                        ctx.fillStyle = "#8b6914";
                        ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
                    }
                }
            }

            // Grid lines
            ctx.strokeStyle = "#d1d5db";
            ctx.lineWidth = 1;
            for (let r = 0; r <= rows; r++) {
                ctx.beginPath();
                ctx.moveTo(0, r * cell);
                ctx.lineTo(cols * cell, r * cell);
                ctx.stroke();
            }
            for (let c = 0; c <= cols; c++) {
                ctx.beginPath();
                ctx.moveTo(c * cell, 0);
                ctx.lineTo(c * cell, rows * cell);
                ctx.stroke();
            }

            // Start markers and goal stars
            for (const robot of data.robots) {
                const [sr, sc] = robot.start;
                const [gr, gc] = robot.goal;

                ctx.fillStyle = robot.color + "99";
                ctx.strokeStyle = "#111827";
                ctx.lineWidth = 1;
                ctx.fillRect(sc * cell + cell * 0.2, sr * cell + cell * 0.2, cell * 0.6, cell * 0.6);
                ctx.strokeRect(sc * cell + cell * 0.2, sr * cell + cell * 0.2, cell * 0.6, cell * 0.6);

                starPath(gc * cell + cell * 0.5, gr * cell + cell * 0.5, cell * 0.24, cell * 0.11, 5);
                ctx.fillStyle = robot.color;
                ctx.globalAlpha = 0.75;
                ctx.fill();
                ctx.globalAlpha = 1.0;
                ctx.stroke();
            }

            // Robot positions at current frame
            const positions = data.frames[frame];
            for (let i = 0; i < data.robots.length; i++) {
                const robot = data.robots[i];
                const [rr, rc] = positions[i];
                const cx = rc * cell + cell * 0.5;
                const cy = rr * cell + cell * 0.5;

                ctx.beginPath();
                ctx.arc(cx, cy, cell * 0.32, 0, Math.PI * 2);
                ctx.fillStyle = robot.color;
                ctx.strokeStyle = "#111827";
                ctx.lineWidth = 1.5;
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = `bold ${Math.max(11, Math.floor(cell * 0.3))}px sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(String(robot.id), cx, cy);
            }

            frameLabelEl.textContent = `t = ${frame} / ${data.frames.length - 1}`;
            frameSlider.value = String(frame);
        }

        function tick() {
            if (!playing) return;
            frame = (frame + 1) % data.frames.length;
            draw();
        }

        function restartTimer() {
            if (timer) clearInterval(timer);
            timer = setInterval(tick, Math.max(30, Math.floor(1000 / fps)));
        }

        playPauseBtn.addEventListener("click", () => {
            playing = !playing;
            playPauseBtn.textContent = playing ? "Pause" : "Play";
        });

        prevBtn.addEventListener("click", () => {
            frame = (frame - 1 + data.frames.length) % data.frames.length;
            draw();
        });

        nextBtn.addEventListener("click", () => {
            frame = (frame + 1) % data.frames.length;
            draw();
        });

        resetBtn.addEventListener("click", () => {
            frame = 0;
            draw();
        });

        fpsSlider.addEventListener("input", () => {
            fps = Number(fpsSlider.value);
            restartTimer();
        });

        frameSlider.addEventListener("input", () => {
            frame = Number(frameSlider.value);
            draw();
        });

        draw();
        restartTimer();
    </script>
</body>
</html>
"""

        html = template.replace("__DATA_JSON__", json.dumps(data))
        with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)

        return output_file


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
    print("-" * 60)
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

        html_file = export_interactive_html(
            env,
            demo_data,
            output_file="warehouse_mapf_interactive.html",
            title=f"Warehouse MAPF - {demo_n} Agents (Interactive)",
        )
        print(f"Interactive viewer saved to '{html_file}'.")

        plt.show()

    except Exception as exc:
        print(f"[Warning] Animation could not be displayed: {exc}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()

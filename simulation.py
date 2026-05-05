"""Main MAPF simulation runner for CS6053 Topic 8 Action Planning."""

from __future__ import annotations

import json
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


def write_interactive_html(scenario_outputs: Dict[int, Dict[str, object]], output_path: str) -> None:
                """Write an interactive browser viewer with a scenario selector (2/5/10 agents)."""
                interactive_data = {}
                for num_agents, out in scenario_outputs.items():
                                env: WarehouseGrid = out["env"]
                                agents: List[RationalAgent] = out["agents"]
                                timeline: Dict[int, List[Tuple[int, int]]] = out["timeline"]
                                frames = build_animation_frames(timeline, len(agents))

                                robots = []
                                for agent in agents:
                                                robots.append(
                                                                {
                                                                                "id": int(agent.agent_id),
                                                                                "start": [int(agent.start_pos[0]), int(agent.start_pos[1])],
                                                                                "goal": [int(agent.pick_target[0]), int(agent.pick_target[1])],
                                                                                "path_length": int(max(0, len(agent.planned_path) - 1)),
                                                                                "color": COLORS[agent.agent_id % len(COLORS)],
                                                                }
                                                )

                                interactive_data[str(num_agents)] = {
                                                "rows": int(env.rows),
                                                "cols": int(env.cols),
                                                "grid": [[int(v) for v in row] for row in env.grid.tolist()],
                                                "frames": [[[int(r), int(c)] for (r, c) in frame] for frame in frames],
                                                "robots": robots,
                                                "title": f"Warehouse MAPF - {num_agents} Agents (Interactive)",
                                }

                html = f"""<!doctype html>
<html lang=\"en\"> 
<head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Warehouse MAPF Interactive Viewer</title>
        <style>
                :root {{
                        --bg: #f8f4ea;
                        --panel: #fffef8;
                        --ink: #1f2937;
                        --accent: #0f766e;
                        --grid: #d1d5db;
                        --shelf: #8b6914;
                }}
                body {{
                        margin: 0;
                        font-family: \"Trebuchet MS\", \"Segoe UI\", sans-serif;
                        background: radial-gradient(circle at 10% 10%, #fff6e1 0%, var(--bg) 45%, #efe8d5 100%);
                        color: var(--ink);
                }}
                .wrap {{
                        max-width: 1100px;
                        margin: 1.2rem auto;
                        padding: 0 1rem;
                        display: grid;
                        gap: 0.9rem;
                        grid-template-columns: 1fr;
                }}
                .panel {{
                        background: var(--panel);
                        border: 1px solid #d6d3d1;
                        border-radius: 14px;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                        padding: 0.8rem;
                }}
                .topbar {{
                        display: flex;
                        gap: 0.8rem;
                        align-items: center;
                        justify-content: space-between;
                        flex-wrap: wrap;
                }}
                .title {{
                        margin: 0;
                        font-size: 1.1rem;
                        letter-spacing: 0.2px;
                }}
                .topbar-controls {{
                        display: flex;
                        align-items: center;
                        gap: 0.6rem;
                        font-size: 0.9rem;
                }}
                select {{
                        border: 1px solid #0f766e;
                        background: #f0fdfa;
                        color: #134e4a;
                        border-radius: 8px;
                        padding: 0.35rem 0.45rem;
                        font-weight: 700;
                        cursor: pointer;
                }}
                canvas {{
                        width: 100%;
                        height: auto;
                        border-radius: 10px;
                        border: 1px solid #d1d5db;
                        background: #f3f4f6;
                        display: block;
                }}
                .controls {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                        gap: 0.5rem;
                        align-items: center;
                        margin-top: 0.8rem;
                }}
                button {{
                        border: 1px solid #0f766e;
                        background: #f0fdfa;
                        color: #134e4a;
                        border-radius: 8px;
                        padding: 0.5rem 0.6rem;
                        font-weight: 700;
                        cursor: pointer;
                }}
                button:hover {{ filter: brightness(0.98); }}
                input[type=\"range\"] {{
                        width: 100%;
                        accent-color: var(--accent);
                }}
                .legend {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 0.35rem 0.75rem;
                        margin-top: 0.5rem;
                        font-size: 0.85rem;
                }}
                .legend-item {{
                        display: flex;
                        align-items: center;
                        gap: 0.4rem;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                }}
                .swatch {{
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        border: 1px solid #111827;
                        flex-shrink: 0;
                }}
                .stat {{
                        font-variant-numeric: tabular-nums;
                        font-size: 0.9rem;
                }}
        </style>
</head>
<body>
        <div class=\"wrap\">
                <div class=\"panel\">
                        <div class=\"topbar\">
                                <h1 class=\"title\" id=\"title\"></h1>
                                <div class=\"topbar-controls\">
                                        <label for=\"scenarioSelect\">Scenario</label>
                                        <select id=\"scenarioSelect\"></select>
                                        <div class=\"stat\" id=\"frameLabel\"></div>
                                </div>
                        </div>
                        <canvas id=\"simCanvas\" width=\"900\" height=\"900\"></canvas>
                        <div class=\"controls\">
                                <button id=\"playPauseBtn\">Pause</button>
                                <button id=\"prevBtn\">Step -1</button>
                                <button id=\"nextBtn\">Step +1</button>
                                <button id=\"resetBtn\">Reset</button>
                                <label>Speed (fps)
                                        <input id=\"fpsSlider\" type=\"range\" min=\"1\" max=\"15\" value=\"5\" />
                                </label>
                                <label>Timeline
                                        <input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" />
                                </label>
                        </div>
                        <div class=\"legend\" id=\"legend\"></div>
                </div>
        </div>

        <script>
                const scenarios = {json.dumps(interactive_data)};
                const scenarioKeys = Object.keys(scenarios).map(Number).sort((a, b) => a - b);

                const canvas = document.getElementById("simCanvas");
                const ctx = canvas.getContext("2d");
                const titleEl = document.getElementById("title");
                const frameLabelEl = document.getElementById("frameLabel");
                const legendEl = document.getElementById("legend");
                const scenarioSelect = document.getElementById("scenarioSelect");

                const playPauseBtn = document.getElementById("playPauseBtn");
                const prevBtn = document.getElementById("prevBtn");
                const nextBtn = document.getElementById("nextBtn");
                const resetBtn = document.getElementById("resetBtn");
                const fpsSlider = document.getElementById("fpsSlider");
                const frameSlider = document.getElementById("frameSlider");

                let data = scenarios[String(scenarioKeys[0])];
                let frame = 0;
                let playing = true;
                let fps = Number(fpsSlider.value);
                let timer = null;

                for (const n of scenarioKeys) {{
                        const opt = document.createElement("option");
                        opt.value = String(n);
                        opt.textContent = `${{n}} agents`;
                        scenarioSelect.appendChild(opt);
                }}

                function rebuildLegend() {{
                        legendEl.innerHTML = "";
                        for (const r of data.robots) {{
                                const item = document.createElement("div");
                                item.className = "legend-item";
                                item.innerHTML = `<span class=\"swatch\" style=\"background:${{r.color}}\"></span>` +
                                                                 `R${{r.id}}: ${{r.start}} -> ${{r.goal}} | len=${{r.path_length}}`;
                                legendEl.appendChild(item);
                        }}
                }}

                function setScenario(key) {{
                        data = scenarios[String(key)];
                        frame = 0;
                        titleEl.textContent = data.title;
                        frameSlider.max = String(data.frames.length - 1);
                        frameSlider.value = "0";
                        rebuildLegend();
                        draw();
                }}

                function starPath(cx, cy, outerR, innerR, points) {{
                        let angle = -Math.PI / 2;
                        const step = Math.PI / points;
                        ctx.beginPath();
                        for (let i = 0; i < points * 2; i++) {{
                                const r = i % 2 === 0 ? outerR : innerR;
                                const x = cx + Math.cos(angle) * r;
                                const y = cy + Math.sin(angle) * r;
                                if (i === 0) ctx.moveTo(x, y);
                                else ctx.lineTo(x, y);
                                angle += step;
                        }}
                        ctx.closePath();
                }}

                function draw() {{
                        const rows = data.rows;
                        const cols = data.cols;
                        const w = canvas.width;
                        const h = canvas.height;
                        const cell = Math.min(w / cols, h / rows);

                        ctx.clearRect(0, 0, w, h);

                        ctx.fillStyle = "#f9fafb";
                        ctx.fillRect(0, 0, cols * cell, rows * cell);

                        for (let r = 0; r < rows; r++) {{
                                for (let c = 0; c < cols; c++) {{
                                        if (data.grid[r][c] === 1) {{
                                                ctx.fillStyle = "#8b6914";
                                                ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
                                        }}
                                }}
                        }}

                        ctx.strokeStyle = "#d1d5db";
                        ctx.lineWidth = 1;
                        for (let r = 0; r <= rows; r++) {{
                                ctx.beginPath();
                                ctx.moveTo(0, r * cell);
                                ctx.lineTo(cols * cell, r * cell);
                                ctx.stroke();
                        }}
                        for (let c = 0; c <= cols; c++) {{
                                ctx.beginPath();
                                ctx.moveTo(c * cell, 0);
                                ctx.lineTo(c * cell, rows * cell);
                                ctx.stroke();
                        }}

                        for (const robot of data.robots) {{
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
                        }}

                        const positions = data.frames[frame];
                        for (let i = 0; i < data.robots.length; i++) {{
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
                                ctx.font = `bold ${{Math.max(11, Math.floor(cell * 0.3))}}px sans-serif`;
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText(String(robot.id), cx, cy);
                        }}

                        frameLabelEl.textContent = `t = ${{frame}} / ${{data.frames.length - 1}}`;
                        frameSlider.value = String(frame);
                }}

                function tick() {{
                        if (!playing) return;
                        frame = (frame + 1) % data.frames.length;
                        draw();
                }}

                function restartTimer() {{
                        if (timer) clearInterval(timer);
                        timer = setInterval(tick, Math.max(30, Math.floor(1000 / fps)));
                }}

                scenarioSelect.addEventListener("change", () => {{
                        setScenario(Number(scenarioSelect.value));
                }});

                playPauseBtn.addEventListener("click", () => {{
                        playing = !playing;
                        playPauseBtn.textContent = playing ? "Pause" : "Play";
                }});

                prevBtn.addEventListener("click", () => {{
                        frame = (frame - 1 + data.frames.length) % data.frames.length;
                        draw();
                }});

                nextBtn.addEventListener("click", () => {{
                        frame = (frame + 1) % data.frames.length;
                        draw();
                }});

                resetBtn.addEventListener("click", () => {{
                        frame = 0;
                        draw();
                }});

                fpsSlider.addEventListener("input", () => {{
                        fps = Number(fpsSlider.value);
                        restartTimer();
                }});

                frameSlider.addEventListener("input", () => {{
                        frame = Number(frameSlider.value);
                        draw();
                }});

                scenarioSelect.value = String(scenarioKeys[0]);
                setScenario(scenarioKeys[0]);
                restartTimer();
        </script>
</body>
</html>
"""
                with open(output_path, "w", encoding="utf-8") as f:
                                f.write(html)


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

        interactive_path = "warehouse_mapf_interactive.html"
        write_interactive_html(scenario_outputs=scenario_outputs, output_path=interactive_path)
        print(f"Saved interactive viewer to {interactive_path}")

        demo_agents = 10
        demo = scenario_outputs[demo_agents]
        fig, ani = create_animation(
                env=demo["env"],
                agents=demo["agents"],
                timeline=demo["timeline"],
                title=f"Warehouse MAPF Action Planning ({demo_agents} Agents)",
        )

        gif_path = "warehouse_mapf.gif"
        writer = animation.PillowWriter(fps=5)
        ani.save(gif_path, writer=writer)
        print(f"Saved animation to {gif_path}")
        plt.show()


if __name__ == "__main__":
        main()

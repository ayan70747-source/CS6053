# CS6053 Topic 8: Action Planning

This project implements a Multi-Agent Pathfinding (MAPF) warehouse simulation that demonstrates intelligent behaviour and rational decision-making with time-aware planning.

## Architecture

| File | Role |
|---|---|
| environment.py | WarehouseGrid using NumPy with cell encoding: 0 floor, 1 shelf, 2 packing station. Includes random pick-task generator. |
| planner.py | Rational A* planner with cost function f(n)=g(n)+h(n), Manhattan heuristic, and prioritized planning with time-space occupancy. |
| agent.py | RationalAgent with states: IDLE, MOVING_TO_PICK, MOVING_TO_DELIVERY. |
| simulation.py | Runs automated scenarios for 2, 5, and 10 agents, performs execution loop, logs metrics, and renders animation. |
| requirements.txt | Python dependencies. |

## Initial State

1. Warehouse is generated with aisle-style shelf layout.
2. Packing station is fixed in a walkable depot region.
3. Agents spawn in walkable cells near the station.
4. Each agent receives one pick location and must return to the station.

## Actions

At each tick, valid actions are:

1. Move north
2. Move south
3. Move west
4. Move east
5. Wait in place

## Transition Model

The planner searches in space-time states:

State: (x, y, t)

Transition:

(x, y, t) -> (x', y', t+1)

Where (x', y') is either a walkable neighbour or the same cell for wait.

## Rational Planning Logic

1. A* uses f(n)=g(n)+h(n).
2. g(n) is elapsed path cost (one unit per tick).
3. h(n) is Manhattan distance to the goal.
4. Prioritized planning reserves higher-priority trajectories in time-space.
5. Lower-priority agents treat reserved vertices and edge swaps as dynamic obstacles.
6. Rationality proof in code:
   - Wait is included as a legal action.
   - The simulation tracks cases where waiting is selected and has lower/equal immediate f pressure than moving into a detour direction.

## Goal Test

Simulation succeeds only if all agents satisfy both conditions:

1. State is IDLE.
2. Position equals the packing station after completing pick and delivery.

## Performance Output (Section 6)

Running simulation.py automatically generates warehouse_performance.csv with columns:

1. Num_Agents
2. Total_Steps_Taken
3. Average_Path_Length
4. Collisions_Avoided
5. Computation_Time_MS

## Interactive Visualisation (Local Browser)

The file `warehouse_mapf_interactive.html` is a fully self-contained interactive viewer. All simulation data is embedded inside the file — no server, no Python, and no internet connection is required.

### How to open it

**Option 1 — Double-click (simplest)**

Open your file explorer, navigate to the project folder, and double-click `warehouse_mapf_interactive.html`. It will open in your default web browser.

**Option 2 — From the terminal**

```bash
# macOS
open warehouse_mapf_interactive.html

# Linux
xdg-open warehouse_mapf_interactive.html

# Windows (Command Prompt or PowerShell)
start warehouse_mapf_interactive.html
```

**Option 3 — Python one-liner local server (use if Option 1/2 fails)**

Some browsers block file:// resources. Running a tiny local server fixes this:

```bash
# Python 3
python -m http.server 8000
```

Then open your browser and go to:

```
http://localhost:8000/warehouse_mapf_interactive.html
```

Press `Ctrl+C` in the terminal to stop the server when done.

### Selecting a scenario

A dropdown at the top of the page lets you switch between the three pre-computed scenarios:

| Scenario | Agents |
|---|---|
| 2 Agents | 2 robots navigating to their pick goals |
| 5 Agents | 5 robots with prioritised planning |
| 10 Agents | 10 robots demonstrating conflict avoidance |

Select a scenario from the dropdown and the grid reloads automatically.

### Playback controls

| Control | What it does |
|---|---|
| **Pause / Play** | Toggles the animation on and off |
| **Step -1** | Moves the simulation back by one tick |
| **Step +1** | Advances the simulation forward by one tick |
| **Reset** | Returns the simulation to tick 0 |
| **Speed (fps) slider** | Sets playback speed from 1 to 15 frames per second |
| **Frame scrubber** | Drag to jump directly to any tick in the simulation |

### Reading the visualisation

| Element | Meaning |
|---|---|
| Dark grey blocks | Shelf obstacles |
| Light cell | Floor (walkable aisle) |
| Green tile | Packing station (delivery goal) |
| Star marker | Pick-up location for each robot |
| Coloured circle | Robot (each robot has a unique colour) |
| Circle at packing station | Robot has completed its mission |

### Troubleshooting

- **Blank page or nothing renders** — Try Option 3 (Python server) above. Some browsers restrict loading from the `file://` protocol.
- **Animation is too fast or too slow** — Adjust the Speed (fps) slider.
- **Want to inspect a specific moment** — Pause the animation, then use Step -1 / Step +1 or drag the frame scrubber.

---

## Visualisation (Static GIF)

Matplotlib animation displays:

1. Shelves in warehouse-like aisle blocks
2. Packing station tile
3. Pick locations as stars
4. Robots as coloured circles moving through time

Output file: warehouse_mapf.gif

## Original Contributions

This implementation includes the following original contributions aligned with coursework expectations:

1. A structured aisle-generation method that mimics realistic warehouse lane geometry instead of purely random obstacles.
2. A two-leg per-agent mission model (start -> pick -> delivery) with explicit finite-state tracking.
3. Time-space occupancy with both vertex and edge conflict handling to avoid head-on swap collisions.
4. Explicit wait-vs-detour rationality tracking, logged as measurable planner behaviour.
5. End-to-end automated experiment pipeline that directly writes CSV analysis data for report inclusion.

## What Is Happening In The Simulation

The simulation models an automated warehouse in which multiple robots must complete a two-stage task:

1. Start from a depot-side location.
2. Travel to a pick cell beside a shelf.
3. Collect the item.
4. Return the item to the packing station.

The warehouse layout is created in [environment.py](/workspaces/CS6053/environment.py), each robot's state is managed in [agent.py](/workspaces/CS6053/agent.py), the planning logic is implemented in [planner.py](/workspaces/CS6053/planner.py), and the full experiment loop is run in [simulation.py](/workspaces/CS6053/simulation.py).

The animation is not random movement. It is the visual result of a planning process where each robot computes a legal action sequence before execution.

## Why This Demonstrates Intelligent Behaviour

The project demonstrates intelligent behaviour because every robot is goal-directed and evaluates actions using informed search rather than fixed rules.

You can explain the intelligence like this:

1. Each agent has a clear goal: reach a pick location and then return to the packing station.
2. Each agent reasons about future outcomes using A* search instead of moving greedily.
3. The planner works in space and time, so it reasons about where robots will be and when they will be there.
4. Lower-priority robots adapt their plans around higher-priority robots' reserved future paths.
5. Waiting is treated as a rational action when immediate movement would create a conflict or a worse route.

This means the system is not just animated movement. It is a decision-making system that chooses actions with respect to goals, costs, and constraints.

## Why The Agents Are Rational

A rational agent is one that selects the action sequence most likely to achieve its goal based on available information.

In this simulation, that means:

1. The agent knows the warehouse map.
2. The agent knows which cells are blocked by shelves.
3. The agent considers the future reserved positions of higher-priority robots.
4. The agent chooses the path with minimum estimated total cost using A*.
5. If the direct route is blocked, the agent can compare waiting against taking a longer detour.

This is the core rationality argument for the coursework. The robot does not simply react; it plans.

## How To Explain A* Simply

The planning score is:

f(n) = g(n) + h(n)

Where:

1. g(n) is the real path cost so far, measured in movement ticks.
2. h(n) is the Manhattan distance to the goal.
3. f(n) is the total estimated cost used to rank candidate states.

The Manhattan heuristic is appropriate because robots move in four directions on a grid.

## How Collision Avoidance Works

Collision avoidance is handled by prioritized planning with time-space occupancy.

The explanation to give is:

1. Agents are ordered by priority.
2. The first agent plans its full path.
3. That path is stored in a reservation table with time stamps.
4. The next agent treats those future occupied cells as dynamic obstacles.
5. The planner also prevents edge-swap collisions, where two robots would cross through each other in opposite directions.

This is why the robots appear coordinated even though each one plans its own route.

## What To Show The Professor

During a demo, the clearest flow is:

1. Show the animation in [warehouse_mapf.gif](/workspaces/CS6053/warehouse_mapf.gif).
2. Explain the map symbols:
   - Brown blocks are shelves.
   - The packing station is the delivery point.
   - Stars are pick targets.
   - Coloured circles are robots.
3. Explain the task sequence: each robot goes from start to pick, then from pick to delivery.
4. Explain that the paths are computed by Rational A* and not manually scripted.
5. Show [warehouse_performance.csv](/workspaces/CS6053/warehouse_performance.csv) as the evidence for performance analysis.

## What To Emphasise In The Report Or Viva

If asked what the project proves, focus on these points:

1. It demonstrates action planning because the system generates full action sequences over time.
2. It demonstrates rationality because each agent selects actions that minimize estimated cost while respecting constraints.
3. It demonstrates intelligent behaviour because robots adapt to dynamic occupancy rather than following static shortest paths.
4. It demonstrates performance evaluation because the code automatically records scenario metrics for 2, 5, and 10 agents.

## Short Presentation Script

This project models a warehouse where multiple robots collect items and return them to a packing station. Each robot is a rational agent because it plans using A* search with the cost function f(n)=g(n)+h(n), where h(n) is Manhattan distance. To avoid collisions, I used prioritized planning with a time-space reservation table, so lower-priority robots treat higher-priority robots' future paths as obstacles. The simulation runs scenarios with 2, 5, and 10 agents, logs performance to [warehouse_performance.csv](/workspaces/CS6053/warehouse_performance.csv), and visualizes the coordinated movement in [warehouse_mapf.gif](/workspaces/CS6053/warehouse_mapf.gif).

## Very Short Viva Answer

If you need a shorter answer in a live discussion, say:

This is a multi-agent warehouse planning system. Each robot plans a path to a pick location and back to the packing station using A*. The robots are rational because they choose low-cost action sequences based on the map and the future movements of other robots. Prioritized planning prevents collisions, and the CSV output provides evidence for performance evaluation.

## Run

1. Install dependencies:

   pip install -r requirements.txt

2. Execute simulation:

   python simulation.py

This runs all required scenarios (2, 5, 10 agents), prints metrics, exports warehouse_performance.csv, and saves warehouse_mapf.gif.

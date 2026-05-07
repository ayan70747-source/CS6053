# CS6053 Topic 8: Action Planning — Group 16

For this project we built a Multi-Agent Pathfinding (MAPF) simulation set inside a warehouse. The idea was to show how multiple robots can plan and carry out tasks intelligently without bumping into each other. Each robot has a job: go pick up an item from a shelf and bring it back to the packing station. The interesting part is getting them all to do this at the same time without conflicts.

## What each file does

| File | What we wrote it to do |
|---|---|
| `environment.py` | Builds the warehouse grid using NumPy. Cells are encoded as 0 (floor), 1 (shelf), or 2 (packing station). It also randomly generates pick tasks for each robot. |
| `planner.py` | This is where the A* planning happens. We use f(n) = g(n) + h(n) with Manhattan distance as the heuristic, and we handle conflicts between robots using prioritised planning in space-time. |
| `agent.py` | Manages each robot's state — it can be IDLE, MOVING_TO_PICK, or MOVING_TO_DELIVERY. |
| `simulation.py` | Runs three experiments (2, 5, and 10 robots), collects performance data, and produces the animation. |
| `requirements.txt` | Python packages needed to run everything. |

## How the simulation starts

When the simulation begins:

1. A warehouse is generated with shelves arranged in proper aisles, not just random blocks.
2. The packing station is placed in a fixed walkable area near the bottom of the grid.
3. Robots spawn in walkable cells close to the station.
4. Each robot is assigned one pick location and needs to go get it and come back.

## What a robot can actually do at each step

At every tick, a robot chooses one of five actions:

1. Move north
2. Move south
3. Move west
4. Move east
5. Wait in place (this is more important than it sounds — see below)

## How the planner thinks about movement

Rather than just planning in 2D space, we plan in space-time. Each state the planner considers is:

```
(x, y, t)
```

And each transition looks like:

```
(x, y, t) → (x', y', t+1)
```

where (x', y') is either a neighbouring walkable cell or the same cell if the robot waits. This lets the planner reason about *when* a robot will be somewhere, not just *where*.

## How robots avoid each other

We use prioritised planning. The first robot plans freely. The second robot then treats the first robot's future positions as obstacles, and so on. We handle two types of conflicts:

- **Vertex conflicts** — two robots trying to be in the same cell at the same time.
- **Edge conflicts** — two robots trying to swap positions in a single tick (which would mean they pass through each other).

The rationality bit we wanted to highlight: waiting is a first-class action. When a robot chooses to wait rather than take a detour, we log that decision. It shows the robot genuinely evaluated its options and decided waiting was cheaper.

## When does the simulation consider itself done?

Every robot has to satisfy two things before we declare success:

1. Its state is IDLE.
2. It is physically sitting on the packing station cell.

Both conditions have to be true. A robot that never made it back does not count.

## Performance data

Running `simulation.py` writes a file called `warehouse_performance.csv`. The columns are:

| Column | What it measures |
|---|---|
| `Num_Agents` | How many robots were in that run |
| `Total_Steps_Taken` | Total ticks across all robots |
| `Average_Path_Length` | Mean path length per robot |
| `Collisions_Avoided` | How many potential conflicts the planner stepped around |
| `Computation_Time_MS` | How long the planning took in milliseconds |

We ran this for 2, 5, and 10 robots so we could compare how the planner scales.

## Viewing the simulation interactively (open in your browser)

We built an interactive HTML viewer so you can step through the simulation at your own pace. Everything is embedded in a single file — `warehouse_mapf_interactive.html` — so there is nothing extra to install and no internet needed.

### Opening the file

**The simplest way — just double-click it**

Find `warehouse_mapf_interactive.html` in the project folder and double-click it. It should open straight in your browser.

**From the terminal**

```bash
# macOS
open warehouse_mapf_interactive.html

# Linux
xdg-open warehouse_mapf_interactive.html

# Windows
start warehouse_mapf_interactive.html
```

**If the page loads blank — use a local server**

Some browsers block local file access for security reasons. If you see a blank page, run this instead:

```bash
python -m http.server 8000
```

Then go to:

```
http://localhost:8000/warehouse_mapf_interactive.html
```

Hit `Ctrl+C` when you are done to shut the server down.

### Switching between scenarios

There is a dropdown at the top of the page. Use it to pick between the three runs we recorded:

| Option | What you are looking at |
|---|---|
| 2 Agents | The simplest case — two robots finding paths without getting in each other's way |
| 5 Agents | Mid-scale — you can start to see the planner making robots wait for each other |
| 10 Agents | The busiest run — lots of conflict avoidance happening |

### Controls

| Button / Slider | What it does |
|---|---|
| **Pause / Play** | Stops or starts the animation |
| **Step -1** | Goes back one tick |
| **Step +1** | Goes forward one tick |
| **Reset** | Jumps back to the very start |
| **Speed (fps) slider** | Drag left to slow down, right to speed up (1–15 fps) |
| **Frame scrubber** | Drag to jump to any specific tick |

We recommend pausing and stepping through tick by tick if you want to see exactly how a robot decides to wait rather than move.

### What you are looking at on the grid

| Visual element | What it represents |
|---|---|
| Dark grey blocks | Shelf obstacles (robots cannot enter these) |
| Light/white cells | Walkable floor aisles |
| Green tile | The packing station — where every robot needs to end up |
| Star marker | The pick-up location assigned to a robot |
| Coloured circle | A robot (each one has its own colour) |
| Circle resting on the green tile | That robot has finished its job |

### Quick fixes

- **Blank page** — Use the Python server method above.
- **Too fast to follow** — Drag the speed slider to the left or just use Step +1 to go one frame at a time.
- **Want to look at a specific moment** — Pause first, then use the step buttons or drag the frame scrubber.

---

## Static GIF output

If you just want a quick overview without opening the HTML, running `simulation.py` also produces `warehouse_mapf.gif` which shows:

- The shelf layout
- The packing station
- Pick locations marked as stars
- Each robot as a coloured circle moving through the warehouse

## What we built ourselves

We want to be upfront about what is original work here rather than just using a textbook A* implementation:

1. **Aisle-style warehouse generation** — we wrote a method that creates realistic-looking warehouse aisles rather than scattering random obstacles. The geometry actually looks like a warehouse.
2. **Two-leg mission per robot** — each robot has a proper start → pick → delivery lifecycle tracked with explicit states, not just a single goal.
3. **Edge conflict detection** — on top of the standard vertex conflict check, we also catch swap collisions (where two robots would pass through each other in one tick). This required tracking pairs of (current, next) positions across all robots at each timestep.
4. **Wait-vs-detour logging** — we log every time a robot chooses to wait, along with the f-cost comparison that led to that choice. This gives us concrete evidence of rational behaviour rather than just asserting it.
5. **Automated experiment pipeline** — the whole thing runs end to end and writes the CSV itself. We did not manually record any of the performance numbers.

## What is actually happening when you watch it

Every robot in the simulation is doing the same two-stage job: go to your pick location, then come back to the packing station. They are not just animated — they planned their full path before the animation starts.

The planning is done in `planner.py`, the state of each robot is tracked in `agent.py`, the grid is in `environment.py`, and `simulation.py` ties it all together and runs the experiments.

The key thing to understand is that the robots are not reacting moment to moment. By the time the animation plays, every robot already knows every step it is going to take. The interesting decisions happened during planning, and the metrics in the CSV reflect how well that planning went at different scales.

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

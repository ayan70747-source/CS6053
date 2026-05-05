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

## Visualisation

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

## Run

1. Install dependencies:

   pip install -r requirements.txt

2. Execute simulation:

   python simulation.py

This runs all required scenarios (2, 5, 10 agents), prints metrics, exports warehouse_performance.csv, and saves warehouse_mapf.gif.

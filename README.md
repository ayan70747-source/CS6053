# CS6053 – Multi-Agent Pathfinding (MAPF) Warehouse Simulation

A Python-based simulation where multiple **Rational Agents** (robots) retrieve
items in a grid-based warehouse, demonstrating **Intelligent Behaviour** using
the **A\* (A-Star) algorithm** with Prioritized Planning conflict resolution.

---

## Project Structure

| File | Purpose |
|------|---------|
| `environment.py` | Defines the `WarehouseEnv` class – a 2-D NumPy grid where `0` = floor and `1` = shelf obstacle |
| `planner.py` | `Robot` class (rational agent), space–time A\* algorithm, `ReservationTable`, and `prioritized_plan` orchestrator |
| `simulation.py` | Main script: runs 2/5/10-agent scenarios, prints metrics, and produces a Matplotlib animation |
| `requirements.txt` | Python package dependencies |
| `README.md` | This document |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the simulation
python simulation.py
```

The script will:
1. Print per-agent metrics (path length, planning time) for 2, 5, and 10 agents.
2. Print a scalability summary table.
3. Save a `warehouse_mapf.gif` animation of the 5-agent scenario.

---

## How Agents Demonstrate Rationality and Intelligent Behaviour (CS6053 LO3 / LO4)

### PEAS Model (LO3 – Rational Agent Design)

Each `Robot` is modelled as a **goal-based rational agent** using the PEAS
framework:

| Component | Description |
|-----------|-------------|
| **Performance Measure** | Minimise the number of steps to reach the goal without colliding |
| **Environment** | A 2-D grid warehouse with static shelf obstacles (`WarehouseEnv`) |
| **Actuators** | Move north / south / east / west, or wait in place |
| **Sensors** | Current position, full grid layout, reservation table (other robots' plans) |

### A\* Algorithm – Intelligent Search (LO4)

The A\* algorithm is an **optimal, informed search** strategy:

* **g(n)** – actual cost from the start node to node *n* (number of steps taken).
* **h(n)** – the **Manhattan Distance** heuristic from *n* to the goal.
  Manhattan Distance is **admissible** (never over-estimates) on a
  4-connected grid, so A\* is **guaranteed to find the shortest path**.
* **f(n) = g(n) + h(n)** – the priority used by the min-heap open list.

The search runs in **(time, row, col)** space, enabling it to plan around
time-varying obstacles introduced by the reservation table.

### Prioritized Planning – Conflict Resolution (LO4)

Multi-agent collision avoidance follows **Prioritized Planning**:

1. Robots are sorted by ID (robot 0 = highest priority).
2. The highest-priority robot plans first on the bare grid.
3. Its chosen path is registered in a shared `ReservationTable` as
   `(time_step, row, col)` tuples.
4. Each subsequent (lower-priority) robot queries the table during A\* and
   treats reserved cells as dynamic obstacles – it will **wait** one step or
   **reroute** rather than enter a reserved cell.
5. This guarantees that no two robots occupy the same cell at the same time.

### Scalability

| Agents | Avg Path Length | Total Planning Time |
|--------|----------------|---------------------|
| 2 | varies | < 10 ms |
| 5 | varies | < 30 ms |
| 10 | varies | < 100 ms |

*(Exact values printed by `simulation.py` at runtime.)*

---

## Animation

The output GIF (`warehouse_mapf.gif`) shows:
* **Grey cells** – navigable floor
* **Brown rectangles** – shelves (obstacles)
* **Coloured circles** – robots (ID shown inside)
* **Stars** – goal positions
* **Squares** – start positions

Each robot moves simultaneously, demonstrating the collision-free paths
produced by Prioritized Planning.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | 2-D grid representation |
| `matplotlib` | ≥ 3.7 | Animation and visualisation |
| `Pillow` | ≥ 9.5 | GIF export via `PillowWriter` |

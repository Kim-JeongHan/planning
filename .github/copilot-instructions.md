# Planning Project - AI Coding Guidelines

## Architecture Overview

This is a **sampling-based path planning library** implementing RRT algorithms in Python. The architecture follows a clean separation of concerns:

- **`planning.graph`**: State-based tree nodes supporting N-dimensional spaces (2D/3D/4D+)
- **`planning.map`**: Centralized environment management (bounds, obstacles, visualization)
- **`planning.sampling`**: RRT/RRT-Connect algorithms with pluggable samplers and collision checkers
- **`planning.visualization`**: Simplified 3D visualization via Viser (single-function API)

**Critical Design Philosophy**: Use `node.state` (NOT `node.position`) throughout for N-dimensional flexibility. The Map class is the single source of truth for environment configuration.

## Essential Workflows

### Running & Testing
```bash
# Run examples (requires PYTHONPATH for some)
uv run python examples/rrt_example.py
PYTHONPATH=/home/jeonghan/workspace/planning uv run python examples/rrt_simple_example.py

# Testing with pytest
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/ -v

# Visualization: Open http://localhost:8080 after running examples
```

### Dependency Management
Uses `uv` (not pip) for package management:
```bash
uv sync                    # Install dependencies
uv sync --extra dev        # Install with dev tools (pytest, ruff, mypy)
```

### Code Quality Checks
```bash
ruff check planning/       # Linting (configured for snake_case enforcement)
black planning/            # Formatting (100 char line length)
mypy planning/             # Type checking (strict mode)
```

## Project-Specific Conventions

### 1. Naming: snake_case is MANDATORY
**This is the #1 coding standard violation to catch.** Python code uses snake_case exclusively:

```python
# ✅ CORRECT
start_state = np.array([0, 0, 1])
goal_node = Node(state=goal_state)
map_env.generate_random_obstacles(server, num_obstacles=15)

# ❌ WRONG - Never use camelCase
startState = np.array([0, 0, 1])     # ❌
goalNode = Node(state=goal_state)    # ❌
def generateObstacles(numObstacles): # ❌
```

### 2. State-Based Node System
All nodes use `.state` attribute (numpy arrays) for N-dimensional support:

```python
# ✅ CORRECT - Always use node.state
node = Node(state=np.array([1.0, 2.0, 3.0]))
distance = np.linalg.norm(node1.state - node2.state)
path_states = node.get_path_states()  # Returns np.ndarray

# ❌ DEPRECATED - Never use these
node.position          # Removed - use node.state
node.get_path_positions()  # Removed - use get_path_states()
```

### 3. Map-Centric Pattern
The `Map` class manages all environment concerns. Always use Map methods instead of standalone functions:

```python
# ✅ CORRECT - Use Map class
from planning.map import Map

map_env = Map(size=20, z_range=(0.5, 2.5))
bounds = map_env.get_bounds()              # Use for RRT initialization
map_env.visualize_bounds(server)           # Visualize environment
map_env.generate_random_obstacles(server, num_obstacles=15)

# Pass map bounds to planners
rrt = RRT(
    start_state=start,
    goal_state=goal,
    bounds=map_env.get_bounds(),  # Critical: use map bounds!
    collision_checker=ObstacleCollisionChecker(map_env.obstacles)
)

# ❌ WRONG - Don't create standalone visualization functions
visualize_map_bounds(...)  # Removed - use map_env.visualize_bounds()
```

### 4. Simplified Visualization API
**Single function does everything**: `visualizer.visualize_branches(nodes, goal_node)`

```python
# ✅ CORRECT - One function shows success (blue) and failure (red) paths
visualizer = RRTVisualizer(server)
visualizer.visualize_branches(rrt.nodes, rrt.goal_node)

# ❌ REMOVED - These functions no longer exist
visualizer.visualize_tree_edges()        # Removed - cluttered output
visualizer.visualize_path()              # Removed - redundant
visualizer.visualize_near_goal_attempts() # Removed - redundant
```

### 5. Type Hints are Required
All functions must have complete type annotations (enforced by mypy):

```python
# ✅ CORRECT
def plan(
    self,
    max_iterations: int = 5000,
    goal_sample_rate: float = 0.1
) -> Optional[List[Node]]:
    """Plan a path from start to goal."""
    pass

def get_bounds(self) -> List[Tuple[float, float]]:
    """Get sampling bounds for each dimension."""
    pass

# ❌ WRONG - Missing type hints
def plan(self, max_iterations=5000):  # ❌
    pass
```

### 6. Google-Style Docstrings
```python
def generate_random_obstacles(
    self,
    server: viser.ViserServer,
    num_obstacles: int,
    min_size: float = 0.5,
    max_size: float = 3.0,
) -> List[Obstacle]:
    """Generate random obstacles in the map.

    Args:
        server: Viser server instance for visualization
        num_obstacles: Number of obstacles to generate
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size

    Returns:
        List of generated obstacles
    """
```

## Integration Patterns

### Typical Planning Workflow
```python
import viser
from planning.map import Map
from planning.sampling import RRT, ObstacleCollisionChecker, RRTConfig
from planning.visualization import RRTVisualizer

# 1. Setup environment
server = viser.ViserServer()
map_env = Map(size=20, z_range=(0.5, 2.5))
map_env.visualize_bounds(server)
map_env.generate_random_obstacles(server, num_obstacles=15)

# 2. Configure planner
rrt = RRT(
    start_state=np.array([-8.0, -8.0, 1.0]),
    goal_state=np.array([8.0, 8.0, 3.0]),
    bounds=map_env.get_bounds(),  # Critical: use Map bounds
    collision_checker=ObstacleCollisionChecker(map_env.obstacles),
    config=RRTConfig(seed=42, max_iterations=5000)
)

# 3. Plan and visualize
path = rrt.plan()  # Returns List[Node] or None
visualizer = RRTVisualizer(server)
visualizer.visualize_start_goal(rrt.start_state, rrt.goal_state)
visualizer.visualize_branches(rrt.nodes, rrt.goal_node)
```

### RRT-Connect (Bidirectional)
```python
from planning.sampling import RRTConnect

rrt_connect = RRTConnect(
    start_state=start,
    goal_state=goal,
    bounds=map_env.get_bounds(),
    collision_checker=ObstacleCollisionChecker(map_env.obstacles)
)
path = rrt_connect.plan()  # Faster convergence than RRT
```

### Custom Samplers
```python
from planning.sampling import GoalBiasedSampler, UniformSampler

# Goal-biased sampling (default)
sampler = GoalBiasedSampler(
    bounds=map_env.get_bounds(),
    goal_state=goal,
    goal_bias=0.05,  # 5% chance to sample goal
    seed=42
)

# Uniform random sampling
sampler = UniformSampler(bounds=map_env.get_bounds(), seed=42)
```

## Testing Practices

### Test File Structure
```python
"""Test module for RRT algorithm."""

import numpy as np
import pytest
from planning.sampling import RRT, ObstacleCollisionChecker
from planning.map import Map

def test_rrt_finds_path_without_obstacles():
    """RRT should find path in obstacle-free space."""
    # Arrange
    map_env = Map(size=10, z_range=(0, 2))
    rrt = RRT(
        start_state=np.array([0, 0, 1]),
        goal_state=np.array([5, 5, 1]),
        bounds=map_env.get_bounds(),
        collision_checker=None  # No obstacles
    )

    # Act
    path = rrt.plan()

    # Assert
    assert path is not None
    assert len(path) >= 2
    assert np.allclose(path[0].state, rrt.start_state)
```

### Running Tests
```bash
# Run all tests
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/ -v

# Run specific test file
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/test_rrt.py -v

# With coverage
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/ --cov=planning
```

## Key Files Reference

- **`planning/graph/node.py`**: Core Node class with N-dimensional state support, parent-child relationships, path extraction, and cost tracking
- **`planning/map/map.py`**: Map class managing bounds, obstacles, and visualization (single source of truth)
- **`planning/sampling/rrt.py`**: RRT and RRT-Connect implementations with pluggable components
- **`planning/visualization/rrt_visualizer.py`**: Three methods only: `visualize_branches()`, `visualize_start_goal()`, `add_coordinate_frame()`
- **`examples/rrt_example.py`**: Complete workflow demonstration (Map → RRT → Visualization)
- **`pyproject.toml`**: Tool configuration (ruff, black, mypy with strict settings)

## Forbidden Patterns (DO NOT Reintroduce)

These were removed for good reasons - do not add them back:

- ❌ `node.position` → Use `node.state`
- ❌ `get_path_positions()` → Use `get_path_states()`
- ❌ `visualize_tree_edges()` → Use `visualize_branches()`
- ❌ `visualize_path()` → Already in `visualize_branches()`
- ❌ `visualize_near_goal_attempts()` → Redundant with branches
- ❌ Standalone `visualize_map_bounds()` → Use `map_env.visualize_bounds()`
- ❌ camelCase variables/functions → Always use snake_case

## Quick Reference Card

| Task | Command/Pattern |
|------|----------------|
| Run example | `uv run python examples/rrt_example.py` |
| Run tests | `PYTHONPATH=$(pwd) uv run pytest tests/ -v` |
| Install deps | `uv sync --extra dev` |
| Lint | `ruff check planning/` |
| Format | `black planning/` |
| Type check | `mypy planning/` |
| Create Map | `map_env = Map(size=20, z_range=(0.5, 2.5))` |
| Get bounds | `bounds = map_env.get_bounds()` |
| Create RRT | `RRT(start, goal, bounds=map_env.get_bounds(), ...)` |
| Visualize | `visualizer.visualize_branches(nodes, goal_node)` |
| Access state | `node.state` (numpy array) |
| Node distance | `node1.distance_to(node2)` or `np.linalg.norm(n1.state - n2.state)` |

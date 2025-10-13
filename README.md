# Planning


A Python 3D path planning library with visualization using Viser. Implements pathplanning algorithms with a unified architecture for easy extension and consistent visualization.


## Features

- 🚀 **Unified Architecture**: All planners extend `RRTBase` for consistency
- 🎨 **Simple Visualization**: One API works for all planners - just pass the planner object
- 🌳 **Multiple Algorithms**: RRT (single-tree) and RRT-Connect (bidirectional)
- 📊 **Detailed Analytics**: Track successful paths and failed collision attempts
- 🔧 **Configurable**: Extensive configuration options for sampling, collision checking, and planning
- 📐 **N-Dimensional**: Works with any dimensional state space (2D, 3D, 4D+)
- 🎯 **Obstacle Avoidance**: Integrated collision detection with boxes and spheres

## Requirements

- Python >= 3.10

## Installation

```bash
uv sync

# Install with dev dependencies (includes pytest)
uv sync --extra dev
```

## Usage

Main program:
```bash
uv run python planning/main.py
```

Run examples:
```bash

# Random obstacle map generation example
uv run python examples/obstacle_map_example.py

# RRT node usage example
uv run python examples/node_example.py

# Simple RRT example (no obstacles)
uv run python examples/rrt_simple_example.py

# RRT with obstacles visualization
uv run python examples/rrt_example.py

# RRT with mixed obstacles (boxes and spheres)
uv run python examples/rrt_mixed_obstacles_example.py

# RRT-Connect bidirectional planning with obstacles
uv run python examples/rrt_connect_example.py
```

After running visualization examples, open `http://localhost:8080` in your browser to view the 3D visualization.

## Testing

Run tests using pytest:

```bash
# Install dev dependencies
uv sync --extra dev

```

Run tests:
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_rrt.py -v

```

## Architecture

### Design Philosophy

This library follows a **clean, extensible architecture** based on object-oriented principles:

1. **Unified Base Class (`RRTBase`)**:
   - All RRT variants extend a common abstract base class
   - Ensures consistent API across different algorithms
   - Simplifies visualization and statistics collection

2. **Separation of Concerns**:
   - **Planning**: `planning.sampling` - algorithms and samplers
   - **Environment**: `planning.map` - obstacles and boundaries
   - **Graph**: `planning.graph` - tree nodes and relationships
   - **Visualization**: `planning.visualization` - rendering and display

3. **Pluggable Components**:
   - Collision checkers: Easy to add custom collision detection
   - Samplers: Switch between uniform, goal-biased, or custom sampling
   - Algorithms: Add new RRT variants by extending `RRTBase`

### Key Design Decisions

- **State-based nodes**: Use `node.state` (numpy arrays) for N-dimensional flexibility
- **Map-centric pattern**: `Map` class is the single source of truth for environment
- **Unified visualization**: One API (`visualize_branches(planner)`) works for all algorithms
- **Failed attempt tracking**: RRT-Connect records collision-blocked extensions for analysis

### Adding New Algorithms

To add a new RRT variant:

```python
from planning.sampling.rrt import RRTBase

class MyRRT(RRTBase):
    def __init__(self, start_state, goal_state, bounds, collision_checker, config):
        super().__init__(start_state, goal_state, bounds, collision_checker, ...)
        # Your initialization

    def plan(self) -> list[Node] | None:
        # Your planning algorithm
        pass

    def get_stats(self) -> dict:
        # Return statistics
        pass

    def get_all_nodes(self) -> list[Node]:
        # Return all explored nodes
        pass

    def get_goal_node(self) -> Node | None:
        # Return goal/connection node
        pass
```

That's it! Your new algorithm automatically works with:
- The unified visualization API
- Statistics collection
- All existing examples

## Dependencies

- viser >= 1.0.13
- numpy

## Quick Start

```python
import numpy as np
import viser
from planning.map import Map
from planning.sampling import RRT, RRTConnect, RRTConfig, ObstacleCollisionChecker
from planning.visualization import RRTVisualizer

# Setup
server = viser.ViserServer()
map_env = Map(size=20, z_range=(0.5, 2.5))
map_env.visualize_bounds(server)
map_env.generate_obstacles(server, num_obstacles=10)

# Create planner (RRT or RRT-Connect)
rrt = RRT(
    start_state=np.array([-8.0, -8.0, 1.0]),
    goal_state=np.array([8.0, 8.0, 2.0]),
    bounds=map_env.get_bounds(),
    collision_checker=ObstacleCollisionChecker(map_env.obstacles),
    config=RRTConfig(seed=42)
)

# Plan path
path = rrt.plan()

# Visualize (unified API for all planners!)
visualizer = RRTVisualizer(server)
visualizer.visualize_start_goal(rrt.start_state, rrt.goal_state)
visualizer.visualize_branches(rrt)  # Just pass the planner!

# Get statistics
stats = rrt.get_stats()
print(f"Explored {stats['num_nodes']} nodes")
print(f"Path length: {rrt.get_path_length():.2f}")
```

## Examples

- `examples/obstacle_map_example.py`: Example of generating box obstacles with random sizes in an n×n map
- `examples/node_example.py`: Example of using RRT tree nodes
- `examples/rrt_simple_example.py`: Simple RRT and RRT-Connect examples without obstacles
- `examples/rrt_example.py`: RRT path planning with box obstacles and 3D visualization
- `examples/rrt_mixed_obstacles_example.py`: RRT with mixed obstacle types (boxes and spheres)
- `examples/rrt_connect_example.py`: RRT-Connect bidirectional planning with detailed visualization

## Modules

### planning.map

Provides map-related functionality.

- **`Map`**: Main map class for managing planning environment
  - `get_bounds()`: Get sampling bounds for path planning
  - `generate_random_obstacles()`: Generate random obstacles in the map
  - `visualize_bounds()`: Visualize map boundaries
  - `add_obstacle()`: Add obstacles to the map
  - `is_valid_state()`: Check if a state is within map bounds
  - Automatically manages obstacles and boundaries

- **`Obstacle`**: Abstract base class for obstacles
  - `get_bounds()`: Get obstacle boundaries
  - `contains_point()`: Check point collision
  - `intersects()`: Check obstacle-obstacle collision

- **`BoxObstacle`**: Box-shaped obstacle implementation (AABB)
  - Concrete implementation of Obstacle abstract class
  - Axis-aligned bounding box collision detection

- **`SphereObstacle`**: Sphere-shaped obstacle implementation
  - Concrete implementation of Obstacle abstract class
  - Spherical collision detection
  - Supports sphere-to-sphere and sphere-to-box intersection

### planning.graph

Provides graph and tree structures for path planning algorithms.

- `Node`: RRT tree node class
  - N-dimensional state tracking (2D, 3D, or higher dimensions)
  - Parent-child relationships
  - Cost tracking (for RRT*)
  - Path retrieval and manipulation
  - Distance calculations
- `distance()`: Calculate distance between nodes
- `steer()`: Steer from one node towards another with max distance
- `get_nearest_node()`: Find nearest node from a list
- `get_nodes_within_radius()`: Get nodes within a radius

#### Node Class Features

- **N-dimensional support**: Works with any dimensional state space (2D, 3D, 4D, etc.)
- **State representation**: Uses numpy arrays for efficient n-dimensional state handling
- **Tree operations**: Parent-child relationships, path retrieval, depth calculation
- **Cost management**: Track and update costs (useful for RRT*)
- **Rewiring**: Change parent nodes (for RRT* optimization)
- **Distance calculations**: Euclidean distance to other nodes or states
- **Path extraction**: Get path from root to node or vice versa
- **Indexing**: Access individual dimensions using `node[i]` syntax

### planning.sampling

Provides sampling-based path planning algorithms with a unified architecture.

#### Base Class

- **`RRTBase`**: Abstract base class for all RRT algorithms
  - Provides common initialization and validation
  - Defines standard interface for planning algorithms
  - Enables unified visualization API
  - Methods:
    - `plan()`: Run the planning algorithm
    - `get_stats()`: Get planning statistics
    - `get_all_nodes()`: Get all explored nodes (for visualization)
    - `get_goal_node()`: Get goal/connection node (for visualization)

#### Algorithms

- **`RRT`**: Rapidly-exploring Random Tree (extends `RRTBase`)
  - Single-tree exploration from start to goal
  - Goal-biased sampling
  - Configurable step size and goal tolerance
  - Collision checking support
  - Tracks successful path and failed branches

- **`RRTConnect`**: Bidirectional RRT (extends `RRTBase`)
  - Grows two trees from start and goal simultaneously
  - Faster convergence than standard RRT
  - Attempts to connect trees at each iteration
  - **Tracks failed extension attempts** for detailed visualization
  - Statistics include:
    - Successful nodes in start tree
    - Successful nodes in goal tree
    - Failed collision attempts
    - Total exploration nodes

#### Samplers

- **`UniformSampler`**: Uniform random sampling in the state space
- **`GoalBiasedSampler`**: Samples goal state with configurable probability

#### Collision Checking

- **`CollisionChecker`**: Abstract base class for collision checkers
  - `is_collision_free()`: Check if a state is collision-free
  - `is_path_collision_free()`: Check if a path is collision-free

- **`ObstacleCollisionChecker`**: Checks collisions with obstacles
  - Concrete implementation for obstacle-based collision detection
  - Point collision detection
  - Path collision detection with configurable resolution

- **`EmptyCollisionChecker`**: For obstacle-free environments
  - Always returns True (no collisions)

#### RRT Features

- **Unified architecture**: All algorithms extend `RRTBase` for consistency
- **N-dimensional state space**: Works with any dimensional configuration space (2D, 3D, 4D+)
- **Obstacle avoidance**: Integrated collision checking with path validation
- **Goal-biased sampling**: Balances exploration and exploitation (RRT)
- **Bidirectional search**: Simultaneous tree growth from start and goal (RRT-Connect)
- **Failed attempt tracking**: Records collision-blocked extensions (RRT-Connect)
- **Path extraction**: Returns sequence of nodes from start to goal
- **Tree visualization**: Complete exploration tree access with success/failure classification
- **Statistics tracking**: Comprehensive metrics including:
  - Number of nodes explored
  - Failed collision attempts
  - Path length
  - Success rate
- **Simple visualization**: Unified API - just pass the planner to `visualize_branches()`

#### Configuration

- **`RRTConfig`**: Configuration for RRT algorithm
  - `sampler`: Sampler type (GoalBiasedSampler or UniformSampler)
  - `max_iterations`: Maximum planning iterations
  - `step_size`: Maximum extension distance
  - `goal_tolerance`: Distance threshold for goal
  - `goal_bias`: Probability of sampling goal (0.0-1.0)
  - `seed`: Random seed for reproducibility

- **`RRTConnectConfig`**: Configuration for RRT-Connect algorithm
  - `max_iterations`: Maximum planning iterations
  - `step_size`: Maximum extension distance
  - `goal_tolerance`: Distance threshold for goal
  - `seed`: Random seed for reproducibility

### planning.visualization

Provides unified visualization utilities for all RRT-based path planning algorithms.

#### Unified Visualization API

The visualization system now provides a **unified interface** that works with any `RRTBase`-derived planner (RRT, RRT-Connect, etc.).

#### RRTVisualizer

**Universal visualizer for all RRT algorithms** - works with both single-tree (RRT) and bidirectional (RRT-Connect) planners.

- **`RRTVisualizer`**: Unified visualizer for all RRT algorithms
  - `visualize_branches(planner)`: **Main function** - visualizes all exploration
    - Takes a planner instance directly (RRT or RRT-Connect)
    - Automatically extracts nodes and goal information
    - Blue/Green: Successful paths that led to the goal
    - Red/Gray: Failed paths and collision attempts
    - Works identically for RRT and RRT-Connect
  - `visualize_start_goal(start_state, goal_state)`: Mark start (green) and goal (red) positions

#### Usage Example

```python
from planning.sampling import RRT, RRTConnect
from planning.visualization import RRTVisualizer

# Create visualizer
visualizer = RRTVisualizer(server)

# Works with RRT
rrt = RRT(start, goal, bounds, collision_checker)
path = rrt.plan()
visualizer.visualize_branches(rrt)  # Just pass the planner!

# Works with RRT-Connect (same API!)
rrt_connect = RRTConnect(start, goal, bounds, collision_checker)
path = rrt_connect.plan()
visualizer.visualize_branches(rrt_connect)  # Same simple API!
```

#### Key Features

- **Unified API**: One function works for all RRT variants
  - `visualize_branches(planner)` - pass any `RRTBase` planner
  - No need to manually extract nodes, goal_node, or connection points
  - Automatically handles different planner types
- **Comprehensive visualization**:
  - Success paths (blue for RRT, green for RRT-Connect)
  - Failed exploration branches (red)
  - **Failed collision attempts** (RRT-Connect only) - shows where obstacles blocked expansion
- **Automatic classification**: No need to manually identify success/failure paths
- **N-dimensional support**: Works with any state dimension (visualizes first 3D)

#### RRT-Connect Enhanced Features

RRT-Connect now tracks and visualizes:
- **Successful tree nodes**: Nodes successfully added to start/goal trees
- **Failed extension attempts**: Nodes that couldn't be added due to collisions
- **Detailed statistics**:
  - `num_nodes_start`: Nodes in start tree
  - `num_nodes_goal`: Nodes in goal tree
  - `num_failed_attempts`: Collision-blocked extensions
  - `total_nodes`: Complete exploration count

## Author

- Kim-JeongHan (Kim-JeongHan@naver.com)

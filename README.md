# Planning

A Python 3D visualization and path planning project using Viser.

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
# 3D curve drawing example
uv run python examples/curve_example.py

# Random obstacle map generation example
uv run python examples/obstacle_map_example.py

# RRT node usage example
uv run python examples/node_example.py

# Simple RRT example (no obstacles)
PYTHONPATH=/home/jeonghan/workspace/planning uv run python examples/rrt_simple_example.py

# RRT with obstacles visualization
PYTHONPATH=/home/jeonghan/workspace/planning uv run python examples/rrt_visualization_example.py
```

After running visualization examples, open `http://localhost:8080` in your browser to view the 3D visualization.

## Testing

Run tests using pytest:

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/

# Run specific test file
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/test_node.py

# Run with verbose output
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/ -v

# Run with coverage (if pytest-cov installed)
PYTHONPATH=/home/jeonghan/workspace/planning uv run pytest tests/ --cov=planning
```

## Dependencies

- viser >= 1.0.13
- numpy

## Examples

- `examples/curve_example.py`: Example of drawing 3D spiral and circular curves
- `examples/obstacle_map_example.py`: Example of generating box obstacles with random sizes in an n×n map
- `examples/node_example.py`: Example of using RRT tree nodes
- `examples/rrt_simple_example.py`: Simple RRT and RRT-Connect examples without obstacles
- `examples/rrt_visualization_example.py`: RRT path planning with obstacles and 3D visualization

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

- **`Obstacle`**: Obstacle class
  - `get_bounds()`: Get obstacle boundaries
  - `contains_point()`: Check point collision
  - `intersects()`: Check obstacle-obstacle collision

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

Provides sampling-based path planning algorithms.

#### Algorithms

- **`RRT`**: Rapidly-exploring Random Tree
  - Single-tree exploration from start to goal
  - Goal-biased sampling
  - Configurable step size and goal tolerance
  - Collision checking support

- **`RRTConnect`**: Bidirectional RRT
  - Grows two trees from start and goal
  - Faster convergence than standard RRT
  - Attempts to connect trees at each iteration

#### Samplers

- **`UniformSampler`**: Uniform random sampling in the state space
- **`GoalBiasedSampler`**: Samples goal state with configurable probability

#### Collision Checking

- **`CollisionChecker`**: Checks collisions with obstacles
  - Point collision detection
  - Path collision detection with configurable resolution
- **`EmptyCollisionChecker`**: For obstacle-free environments

#### RRT Features

- **N-dimensional state space**: Works with any dimensional configuration space
- **Obstacle avoidance**: Integrated collision checking
- **Goal-biased sampling**: Balances exploration and exploitation
- **Path extraction**: Returns sequence of nodes from start to goal
- **Tree visualization**: Access to full exploration tree for visualization
- **Statistics tracking**: Number of nodes, path length, success rate

### planning.visualization

Provides simple visualization utilities for path planning algorithms.

- **`RRTVisualizer`**: Visualizer for RRT algorithms
  - `visualize_branches()`: **Main function** - visualizes all exploration paths
    - Blue: Successful paths that led to the goal
    - Red: Failed paths (dead ends)
    - Automatically categorizes all branches
  - `visualize_start_goal()`: Mark start (green) and goal (red) positions
  - `add_coordinate_frame()`: Add coordinate axes to scene

#### Key Features

- **One function to visualize everything**: `visualize_branches()` shows all RRT exploration
- **Color-coded paths**: Blue for success, Red for failure
- **Automatic classification**: No need to manually identify success/failure paths
- **Clean visualization**: Only shows meaningful branches, filters out noise
- **N-dimensional support**: Works with any state dimension (visualizes first 3D)

## Author

- Kim-JeongHan (Kim-JeongHan@naver.com)

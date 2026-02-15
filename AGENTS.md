# AGENTS.md

## 1) Project Summary

This repository (`planning`) is a Python library and demo suite for sampling-based path planning algorithms with obstacle-aware collision checking, designed for 2D/3D robotics planning experiments, visualization, and planner benchmarking.

Main languages:
- Python (primary)
- Bash (docs/image generation scripts)

## 2) Environment Setup

### Required runtimes/tools
- Python >= 3.10
- `uv` (project dependency/runtime manager)
- Git

### Core dependencies
- `viser` for visualization
- `pydantic` for typed config models
- `numpy` (indirect dependency used throughout codebase)
- `tqdm` for progress output

### Optional dependencies
- `torch` (only if using `DiffusionGuidedSampler` and local training/inference)
- `pytest`, `ruff`, `black`, `mypy`, `pre-commit` for dev/test checks

### Setup commands
```bash
uv sync
uv sync --extra dev
uv sync --extra diffuser
```

### Environment variables
- For docs/example rendering scripts, ensure project root is on PYTHONPATH when needed:
  - `export PYTHONPATH=/home/jeonghan/workspace/planning`
- Most commands run with `uv run` and do not require manual `PYTHONPATH` changes.

## 3) Build & Test Commands

### Run examples
- Single planner:
  - `uv run python examples/rrt_example.py --save-image`
  - `uv run python examples/rrt_diffuser_3d_example.py --save-image`
  - `uv run python -m diffuser.cli.train --help`

### Run tests
- All tests:
  - `uv run pytest tests/ -v`
- Single module:
  - `uv run pytest tests/test_rrt.py -v`
  - `uv run pytest tests/test_sampling_diffusion_guided_sampler.py -v`

### Lint / static checks
- Ruff:
  - `uv run ruff check planning/ tests/`
- Black:
  - `uv run black planning/ tests/`
- MyPy:
  - `uv run mypy planning/`
- Pre-commit bundle:
  - `uv run pre-commit run --all-files`

### Documentation image generation
- `bash scripts/update_docs_images.sh`

## 4) Project Structure

- `planning/` (core library)
  - `planning/graph/` : graph/node primitives and search helpers
  - `planning/search/` : graph search (A*)
  - `planning/map/` : map/obstacle generation and bounds utility
  - `planning/collision/` : collision checker interfaces (obstacles, empty checker)
  - `planning/sampling/` : sampler definitions and planner algorithms
    - `planner modules`: `rrt.py`, `rrg.py`, `prm.py`, `sampler.py`
  - `planning/visualization/` : planner-specific visualizers (Viser-based)
- `examples/` : runnable planner demos (RRT / RRT* / RRT-Connect / PRM / PRM* / Informed RRT* / Diffusion 3D)
- `tests/` : pytest suite for graph/search/sampler/planner behavior
- `scripts/` : documentation/image generation helpers
- `docs/` : generated docs images and assets
- Root files:
  - `pyproject.toml` (dependencies, lint/type configs)
  - `.pre-commit-config.yaml` (format/lint hooks)
  - `README.md` (feature docs and usage)

## 5) Coding Conventions

### General style
- Python 3.10+; type hints are expected for functions and attributes where feasible.
- 4-space indentation.
- Use explicit imports and clear variable names.
- Prefer small, targeted patches.
- Keep public-facing signatures backward-compatible unless migration is explicit.

### Linters and formatting
- Ruff and Black are canonical; line length target is 100.
- Ruff import/isort ordering applies (per project config).
- Avoid wildcard imports.

### API/documentation style
- Use Google-style docstrings when modifying/adding modules and classes.
- Keep docstrings concise: purpose, args, returns, raises.
- Prefer `-> None` annotation for side-effect-only functions.

### Framework/library preferences
- Use existing `planner` and `visualizer` abstractions:
  - planners via `RRT`, `RRG`, `PRM`, etc.
  - visualizers under `planning.visualization`
- Prefer Pydantic configs for planner options (`RRTConfig`, `RRTStarConfig`, etc.).
- Use `numpy` for vector/matrix operations.

## 6) Review & Quality

### Error handling
- Validate dimensions for planner state inputs and bounds consistency.
- Raise explicit exceptions for invalid inputs (do not silently clamp unless design requires).
- For optional integrations (e.g., diffuser), fail with actionable messages and clear dependency instructions.

### What to check after edits
- Imports and public exports in package `__init__` files remain consistent.
- New config fields should include defaults and avoid breaking old constructors.
- Unit tests / smoke tests for touched modules should be runnable.
- For new files, ensure module import path is discoverable (`planning/...` package).
- Keep runtime behavior stable in both default and configured modes.

### Review expectations
- Prioritize:
  1) correctness of planner logic,
  2) API compatibility,
  3) type-safety,
  4) deterministic behavior when seeds are set.

### Testing Policy
- Add/extend tests for every new behavior or helper.
- Keep helper functions pure where possible and test them directly.
- For diffusion-dependent tests, guard imports with `pytest.importorskip("planning.diffusion")` when model execution is required.
- Default smoke validation:
  - `uv run pytest tests/<target> -v`
  - `uv run ruff check planning/ tests/ examples/`

### Test Writing Guidelines
- Each behavioral change must include at least one test in `tests/`.
- Use fixture-level helpers instead of global mutable state.
- Prefer deterministic expectations:
  - fixed seeds,
  - fixed inputs,
  - explicit shape/value asserts.
- For new public helpers, add one unit test and one failure-path test.
- Keep regression tests close to modified example behavior:
  - `tests/test_diffusion_trajectory_one_shot_example.py` for one-shot example helpers.
  - `tests/test_diffuser_sampling.py` for sampler/policy behavior.
- Required verification command:
  - `uv run pytest tests/test_diffusion_trajectory_one_shot_example.py -v`
  - `uv run pytest tests/test_diffuser_sampling.py -v`

### Python naming rules
- Files/modules: `snake_case`.
- Functions, methods, variables: `snake_case`.
- Classes, exceptions, pydantic models: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Protocols/type aliases: `PascalCase` (`...Type` suffix if helpful).

### Python code pattern rules
- Separate config loading/parsing from planning execution.
- Keep side-effect code in `main`.
- Build complex logic in small pure helpers, then compose in orchestration functions.
- Validate inputs at boundaries and raise `TypeError`/`ValueError` early.
- Prefer explicit failure paths (`if ...: raise ...`) over deep nesting.
- Keep optional dependencies optional:
  - `pytest.importorskip(...)` in tests,
  - clear runtime `ImportError` message in examples/samplers.

### Code pattern rules
- Separate configuration/load/parsing from execution logic.
- Isolate deterministic logic in helper functions; keep CLI / visualization code thin in `main`.
- Prefer injectable dependencies for testability (예: collision checker, policy object).
- Prefer early returns over deeply nested branches.

## 7) Operational Rules

- Do not remove or edit files unrelated to the task unless explicitly requested.
- Prefer `apply_patch` for file edits.
- Avoid broad, destructive operations (`rm -rf`, `git reset --hard`, etc.) unless user explicitly asks.
- Avoid reading the entire repository blindly; read narrowly targeted files needed for the task.
- For dependency-sensitive modules, handle missing optional deps gracefully and loudly (import-time or explicit runtime error with clear message).
- Maintain no regression: existing examples and tests should keep working after changes.
- Do not introduce broad side effects in examples (e.g., deleting files, long loops without stop conditions, network calls).

## 8) Helpful Examples

### Common workflow: create and run a planner
```python
import numpy as np
from planning.map import Map
from planning.collision import ObstacleCollisionChecker
from planning.sampling import RRT, RRTConfig

map_env = Map(size=20, z_range=(0.5, 2.5))
checker = ObstacleCollisionChecker(map_env.obstacles)

planner = RRT(
    start_state=(0.0, 0.0, 1.0),
    goal_state=(10.0, 10.0, 1.5),
    bounds=map_env.get_bounds(),
    collision_checker=checker,
    config=RRTConfig(max_iterations=3000, step_size=0.5, seed=42),
)
path = planner.plan()
```

### Common workflow: custom sampler with planner kwargs
```python
from planning.sampling import RRT, RRTConfig
from planning.sampling.sampler import UniformSampler

planner = RRT(
    start_state=(0.0, 0.0, 1.0),
    goal_state=(5.0, 5.0, 2.0),
    bounds=[(0, 10), (0, 10), (0, 3)],
    config=RRTConfig(
        sampler=UniformSampler,
        sampler_kwargs={"seed": 123},
    ),
)
```

### Diffusion-guided 3D example (requires optional dependency)
```bash
uv run python examples/rrt_diffuser_3d_example.py --save-image
```

### Helpful commands
- Install deps: `uv sync --extra dev`
- Run one test file: `uv run pytest tests/test_rrt.py -v`
- Regenerate docs images: `bash scripts/update_docs_images.sh`

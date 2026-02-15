"""Command line interface for training local diffusion/value models."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train local diffuser models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file with default arguments.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to trajectory dataset.")
    parser.add_argument("--output-root", type=str, default=None, help="Checkpoint output root.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Trajectory horizon. If omitted, inferred from dataset length.",
    )
    parser.add_argument("--state-dim", type=int, default=None, help="State dimension.")
    parser.add_argument(
        "--n-diffusion-steps", type=int, default=None, help="Diffusion steps."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--log-every", type=int, default=None, help="Print loss every N epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["constant", "step", "cosine"],
        help="Learning-rate schedule for training: constant, step, or cosine.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=None,
        help="Step size for lr-schedule='step'.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=None,
        help="Decay factor for lr-schedule='step'.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=None,
        help="Minimum learning rate for schedules.",
    )
    parser.add_argument("--discount", type=float, default=None, help="Discount used for value checkpoint path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no-value", dest="train_value", action="store_false", default=None, help="Skip value model training.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    yaml_defaults = _load_yaml(args.config)
    values = _coalesce_values(args, yaml_defaults)
    if not values["dataset"]:
        parser.error(
            "No dataset specified. Provide --dataset or set `dataset` in the YAML config."
        )

    ckpts = train(
        dataset=values["dataset"],
        output_root=values["output_root"],
        horizon=values["horizon"],
        state_dim=values["state_dim"],
        n_diffusion_steps=values["n_diffusion_steps"],
        epochs=values["epochs"],
        batch_size=values["batch_size"],
        learning_rate=values["learning_rate"],
        lr_schedule=values["lr_schedule"],
        lr_step_size=values["lr_step_size"],
        lr_gamma=values["lr_gamma"],
        lr_min=values["lr_min"],
        discount=values["discount"],
        seed=values["seed"],
        train_value=values["train_value"],
        log_every=values["log_every"],
    )
    print("Saved checkpoints:")
    for path in ckpts:
        print(f"- {path}")
    return 0


def _load_yaml(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}

    resolved = Path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. Provide a valid path to --config."
        )

    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {resolved}")
    return payload


def _coalesce_values(args: argparse.Namespace, yaml_values: dict[str, object]) -> dict[str, object]:
    def pick(name: str, fallback: object) -> object:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        if name in yaml_values:
            return yaml_values[name]
        return fallback

    dataset = cast_or_raise_str(pick("dataset", None))
    return {
        "dataset": dataset,
        "output_root": pick("output_root", "logs"),
        "horizon": cast_or_raise_optional_int(
            pick("horizon", None), "horizon"
        ),
        "state_dim": cast_or_raise_int(pick("state_dim", 3), "state-dim"),
        "n_diffusion_steps": cast_or_raise_int(
            pick("n_diffusion_steps", 100), "n-diffusion-steps"
        ),
        "epochs": cast_or_raise_int(pick("epochs", 1), "epochs"),
        "batch_size": cast_or_raise_int(pick("batch_size", 16), "batch-size"),
        "learning_rate": cast_or_raise_float(
            pick("learning_rate", 1e-3), "learning-rate"
        ),
        "lr_schedule": cast_or_raise_strict(
            pick("lr_schedule", "constant"), {"constant", "step", "cosine"}
        ),
        "lr_step_size": cast_or_raise_optional_int(pick("lr_step_size", 100), "lr-step-size"),
        "lr_gamma": cast_or_raise_float(
            pick("lr_gamma", 0.5), "lr-gamma"
        ),
        "lr_min": cast_or_raise_float(pick("lr_min", 1e-5), "lr-min"),
        "discount": cast_or_raise_float(pick("discount", 1.0), "discount"),
        "seed": cast_or_raise_optional_int(pick("seed", None), "seed"),
        "log_every": cast_or_raise_int(pick("log_every", 100), "log-every"),
        "train_value": pick("train_value", True),
    }


def cast_or_raise_float(value: object, name: str) -> float:
    if value is None:
        raise TypeError(f"`{name}` must be a number, got {value!r}")
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise TypeError(f"`{name}` must be a number, got {type(value)!r}")
    return value


def cast_or_raise_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer, got {type(value)!r}")
    return value


def cast_or_raise_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise TypeError(f"`{name}` must be an integer or null, got {type(value)!r}")


def cast_or_raise_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"dataset must be a string, got {type(value)!r}")
    return value


def cast_or_raise_strict(value: object, choices: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"value must be one of {sorted(choices)}, got {type(value)!r}")
    if value not in choices:
        raise TypeError(f"value must be one of {sorted(choices)}, got {value!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

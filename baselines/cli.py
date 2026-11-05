"""Unified command line entry point for running JaxMARL baselines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from .registry import ALGO_REGISTRY, FACMAC_REGISTRY
except ImportError:
    from registry import ALGO_REGISTRY, FACMAC_REGISTRY


def _load_launcher():
    try:
        from .run_minimal_baseline_set import run_combo
        from .run_minimal_baseline_set import ALGO_REGISTRY as launcher_registry
    except ImportError:
        from run_minimal_baseline_set import run_combo
        from run_minimal_baseline_set import ALGO_REGISTRY as launcher_registry
    if launcher_registry != ALGO_REGISTRY:
        raise RuntimeError("baseline registries are out of sync")
    return run_combo


def _registry_run(registry: dict, algorithm: str, suite: str, env: str) -> dict:
    registry_entry = registry.get((algorithm, suite))
    run = {
        "name": f"{algorithm}_{env}",
        "suite": suite,
        "env": env,
        "algo": algorithm,
        "seeds_per_gpu": 1,
    }
    if registry_entry is None:
        run["implemented"] = False
    else:
        script, alg_key, alg_prefix = registry_entry
        run.update(
            implemented=True,
            script=script,
            alg_key=alg_key,
            alg_prefix=alg_prefix,
        )
    return run


def _facmac_run(algorithm: str, suite: str, env: str) -> dict:
    target = FACMAC_REGISTRY.get((algorithm, suite))
    run = {
        "name": f"{algorithm}_{env}",
        "suite": suite,
        "env": env,
        "algo": algorithm,
        "implemented": target is not None,
    }
    if target is not None:
        config, env_config, env_arg = target
        run.update(config=config, env_config=env_config, env_arg=env_arg)
    return run


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more JaxMARL baseline algorithms with WandB logging."
    )
    parser.add_argument("--algorithm", "-a")
    parser.add_argument("--suite")
    parser.add_argument("--env", help="Environment/map name understood by the suite.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "jaxmarl"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds-per-gpu", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Rollout length per update (default: 128).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Training environment timesteps (default: 1000000).",
    )
    parser.add_argument("--checkpoint-path", help="Checkpoint file to save or resume.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save every N training updates (default: 1).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Pass an additional native FACMAC Sacred override (repeatable).",
    )
    parser.add_argument(
        "--no-version-tag",
        action="store_true",
        help="Do not add the JaxMARL version to WandB tags and groups.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the target without launching training.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_targets",
        help="List registered algorithm/suite combinations.",
    )
    args = parser.parse_args(argv)

    if args.list_targets:
        return args
    missing = [
        option
        for option, value in (("--algorithm", args.algorithm), ("--suite", args.suite), ("--env", args.env))
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    if args.seeds < 1 or args.seeds_per_gpu < 1:
        parser.error("--seeds and --seeds-per-gpu must be positive")
    if args.steps < 1 or args.total_timesteps < 1 or args.checkpoint_interval < 1:
        parser.error("--steps, --total-timesteps, and --checkpoint-interval must be positive")
    return args


def _validate_run(run: dict) -> bool:
    if not run["implemented"]:
        print(
            f"ERROR: no registry entry for ({run['algo']}, {run['suite']})",
            file=sys.stderr,
        )
        return False
    script_path = Path(__file__).resolve().parent.parent / run["script"]
    if not script_path.exists():
        print(f"ERROR: script missing: {run['script']}", file=sys.stderr)
        return False
    if run["alg_key"]:
        config_path = script_path.parent / "config" / "alg" / f"{run['alg_key']}.yaml"
        if not config_path.exists():
            print(f"ERROR: config missing: {config_path}", file=sys.stderr)
            return False
    return True


def _validate_facmac_run(run: dict) -> bool:
    if not run["implemented"]:
        print(
            f"ERROR: no FACMAC entry for ({run['algo']}, {run['suite']})",
            file=sys.stderr,
        )
        return False
    root = Path(__file__).resolve().parent.parent / "facmac" / "src"
    for relative in (
        Path("main.py"),
        Path("config") / "algs" / f"{run['config']}.yaml",
        Path("config") / "envs" / f"{run['env_config']}.yaml",
    ):
        if not (root / relative).exists():
            print(f"ERROR: FACMAC file missing: {root / relative}", file=sys.stderr)
            return False
    return True


def _jax_suite_env_args(suite: str, env: str, alg_prefix: str) -> list[str]:
    if suite == "smax":
        return [f"{alg_prefix}MAP_NAME={env}"]
    if suite == "mamujoco":
        return [f"{alg_prefix}ENV_NAME={env}"]
    if suite == "mpe":
        mpe_names = {
            "simple": "MPE_simple_v3",
            "simple_spread": "MPE_simple_spread_v3",
            "simple_tag": "MPE_simple_tag_v3",
        }
        return [f"{alg_prefix}ENV_NAME={mpe_names.get(env, env)}"]
    return [f"{alg_prefix}ENV_NAME={env}"]


def _run_facmac(
    run: dict,
    *,
    seed: int,
    gpu: int,
    steps: int | None,
    overrides: list[str],
) -> bool:
    root = Path(__file__).resolve().parent.parent / "facmac" / "src"
    command = [
        sys.executable,
        "main.py",
        f"--config={run['config']}",
        f"--env-config={run['env_config']}",
        "with",
        f"seed={seed}",
        f"env_args.{run['env_arg']}={run['env']}",
        *overrides,
    ]
    if steps is not None:
        command.append(f"t_max={steps}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.run(command, cwd=root, env=environment).returncode == 0


def _run_jax_facmac(
    run: dict,
    *,
    project: str,
    entity: str,
    wandb_mode: str,
    seed: int,
    seeds: int,
    gpus: list[int],
    steps: int | None,
    total_timesteps: int,
    checkpoint_path: str | None,
    checkpoint_interval: int,
) -> bool:
    """Run the JAX FACMAC entry point without importing the Hydra batch launcher."""
    for index in range(seeds):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / run["script"]),
            f"+alg={run['alg_key']}",
            f"PROJECT={project}",
            f"ENTITY={entity}",
            f"WANDB_MODE={wandb_mode}",
            f"SEED={seed + index}",
            "NUM_SEEDS=1",
        ]
        command += [
            f"alg.NUM_STEPS={steps}",
            f"alg.TOTAL_TIMESTEPS={total_timesteps}",
            f"CHECKPOINT_PATH={checkpoint_path or ''}",
            f"CHECKPOINT_INTERVAL={checkpoint_interval}",
        ]
        command += _jax_suite_env_args(run["suite"], run["env"], run["alg_prefix"])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        environment["WANDB_RUN_GROUP"] = f"{run['algo']}-{run['env']}"
        environment["WANDB_NAME"] = f"{run['algo']}-{run['env']}-{index + 1}"
        print(
            f"Running {run['name']} seed {index + 1}/{seeds} "
            f"on GPU {gpus[index % len(gpus)]} for {total_timesteps} timesteps",
            flush=True,
        )
        if subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, env=environment).returncode:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_targets:
        for algorithm, suite in sorted((*ALGO_REGISTRY, *FACMAC_REGISTRY)):
            family = "facmac" if (algorithm, suite) in FACMAC_REGISTRY else "jaxmarl"
            print(f"{algorithm:<18} {suite:<12} {family}")
        return 0

    run = _registry_run(ALGO_REGISTRY, args.algorithm, args.suite, args.env)
    facmac_run = _facmac_run(args.algorithm, args.suite, args.env)
    if facmac_run["implemented"]:
        if not _validate_facmac_run(facmac_run):
            return 1
        if args.dry_run:
            return 0
        passed = all(
            _run_facmac(
                facmac_run,
                seed=args.seed + index,
                gpu=args.gpus[index % len(args.gpus)],
                steps=args.steps,
                overrides=args.override,
            )
            for index in range(args.seeds)
        )
        return 0 if passed else 1
    if not _validate_run(run):
        return 1
    if args.dry_run:
        return 0

    if run["script"].startswith(
        ("baselines/FACMAC/", "baselines/RODE/", "baselines/I2C/", "baselines/NDQ/")
    ):
        return int(
            not _run_jax_facmac(
                run,
                project=args.project,
                entity=args.entity,
                wandb_mode=args.wandb_mode,
                seed=args.seed,
                seeds=args.seeds,
                gpus=args.gpus,
                steps=args.steps,
                total_timesteps=args.total_timesteps,
                checkpoint_path=args.checkpoint_path,
                checkpoint_interval=args.checkpoint_interval,
            )
        )

    run_combo = _load_launcher()
    print(
        f"Running {run['name']} for {args.seeds} seed(s) "
        f"on GPU(s) {args.gpus} -> WandB project '{args.project}'"
    )
    _, passed, _ = run_combo(
        run,
        project=args.project,
        entity=args.entity,
        wandb_mode=args.wandb_mode,
        seed=args.seed,
        seeds=args.seeds,
        gpus=args.gpus,
        seeds_per_gpu=args.seeds_per_gpu,
        tag_jaxmarl_version=not args.no_version_tag,
        steps=args.steps,
        total_timesteps=args.total_timesteps,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

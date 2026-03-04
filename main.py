#!/usr/bin/env python3

import argparse
import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from core import export_all, recommend_all, recommend_single, validate_project

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "app": {
        "mode": "export",
        "data_dir": None,
        "output_dir": "outputs",
    },
    "single": {
        "tasks": 10000,
        "distribution": "50/50",
        "objective": "a",
    },
    "all": {
        "objective": "a",
    },
    "agent": {
        "prompt": "For 10000 tasks and 50/50 distribution solve objective a",
    },
    "validate": {
        "output_dir": "outputs",
    },
}


def load_config(config_path):
    cfg_path = Path(config_path).expanduser().resolve()
    config = {
        "app": dict(DEFAULT_CONFIG["app"]),
        "single": dict(DEFAULT_CONFIG["single"]),
        "all": dict(DEFAULT_CONFIG["all"]),
        "agent": dict(DEFAULT_CONFIG["agent"]),
        "validate": dict(DEFAULT_CONFIG["validate"]),
    }

    if not cfg_path.exists():
        return config, cfg_path.parent

    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    for section in ("app", "single", "all", "agent", "validate"):
        if section in raw and isinstance(raw[section], dict):
            config[section].update(raw[section])

    return config, cfg_path.parent


def resolve_path(cli_value, cfg_value, cfg_base_dir):
    if cli_value is not None:
        return str(Path(cli_value).expanduser().resolve())

    if cfg_value is None:
        return None

    p = Path(cfg_value).expanduser()
    if not p.is_absolute():
        p = cfg_base_dir / p
    return str(p.resolve())


def pick(cli_value, cfg_value, name):
    value = cli_value if cli_value is not None else cfg_value
    if value is None:
        raise SystemExit(f"Missing value for {name}. Set it in CLI or config file.")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Simple agent solution for HPC Theme 1")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "app_config.toml"), help="Path to config file")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to SortedAvgDiffAll88Datasets (optional, can be in config)",
    )

    sub = parser.add_subparsers(dest="cmd")

    one = sub.add_parser("single", help="Solve one scenario")
    one.add_argument("--tasks", type=int)
    one.add_argument("--distribution")
    one.add_argument("--objective", choices=["a", "b", "c"])

    all_cmd = sub.add_parser("all", help="Solve all 88 scenarios for one objective")
    all_cmd.add_argument("--objective", choices=["a", "b", "c"])

    export_cmd = sub.add_parser("export", help="Export rec_a.csv, rec_b.csv, rec_c.csv")
    export_cmd.add_argument("--output-dir")

    validate_cmd = sub.add_parser("validate", help="Validate datasets and exported recommendations")
    validate_cmd.add_argument("--output-dir")

    agent_cmd = sub.add_parser("agent", help="Run OpenAI Agent once")
    agent_cmd.add_argument("--prompt")

    return parser


def run_single(data_dir, tasks, distribution, objective):
    result = recommend_single(tasks, distribution, objective, data_dir)
    quad = (
        result["count_vm_1"],
        result["count_vm_2"],
        result["count_vm_3"],
        result["count_vm_4"],
    )
    print(f"recommended_vm_quad={quad}")
    print(result)


def run_all(data_dir, objective):
    rows = recommend_all(objective, data_dir)
    print(f"recommendations={len(rows)}")
    for row in rows[:10]:
        print(
            f"{row['tasks_count']:>5} {row['distribution']:<7} -> "
            f"({row['count_vm_1']},{row['count_vm_2']},{row['count_vm_3']},{row['count_vm_4']}) "
            f"score={row['score']}"
        )


def run_export(data_dir, output_dir):
    exported = export_all(output_dir=output_dir, data_dir=data_dir)
    for objective, path in exported.items():
        print(f"[{objective}] {path}")


def run_validate(data_dir, output_dir):
    report = validate_project(data_dir=data_dir, output_dir=output_dir)

    print("Validation summary")
    print(f"- datasets_count: {report['datasets_count']}")
    print(f"- full_grid: {report['full_grid']}")
    print(f"- task_values: {report['task_values']}")
    print(f"- distribution_values: {report['distribution_values']}")

    missing_success = report["datasets_without_success100"]
    if not missing_success:
        print("- datasets_without_success100: 0")
    else:
        print(f"- datasets_without_success100: {len(missing_success)}")
        for name in missing_success:
            print(f"  - {name}")

    print("- output_status:")
    for objective in ("a", "b", "c"):
        status = report["output_status"].get(objective, {})
        if not status.get("exists"):
            print(f"  - rec_{objective}.csv: missing")
            continue

        print(
            "  - rec_{o}.csv: rows={rows}, objectives={obj}, unique_dataset_files={uniq}".format(
                o=objective,
                rows=status.get("rows"),
                obj=status.get("objectives"),
                uniq=status.get("unique_dataset_files"),
            )
        )


def run_agent(data_dir, prompt):
    try:
        from agents import Runner
    except ImportError:
        raise SystemExit("openai-agents is not installed. Run: pip install -r requirements.txt")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it into .env or system env variables.")

    from agent_runtime import build_agent

    agent = build_agent(data_dir=data_dir)
    try:
        result = Runner.run_sync(agent, prompt)
    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message or "Error code: 429" in message:
            raise SystemExit(
                "OpenAI API quota is exceeded (429 insufficient_quota). "
                "Direct modes single/all/export/validate still work."
            )
        raise SystemExit(f"Agent run failed: {message}")

    print(result.final_output)


def main():
    load_dotenv(SCRIPT_DIR / ".env")

    parser = build_parser()
    args = parser.parse_args()
    cfg, cfg_base_dir = load_config(args.config)

    mode = str(args.cmd or cfg["app"].get("mode", "single")).strip().lower()
    data_dir = resolve_path(args.data_dir, cfg["app"].get("data_dir"), cfg_base_dir)

    if mode == "single":
        tasks = pick(getattr(args, "tasks", None), cfg["single"].get("tasks"), "single.tasks")
        distribution = pick(
            getattr(args, "distribution", None),
            cfg["single"].get("distribution"),
            "single.distribution",
        )
        objective = pick(
            getattr(args, "objective", None),
            cfg["single"].get("objective"),
            "single.objective",
        )
        run_single(data_dir, int(tasks), str(distribution), str(objective))
        return

    if mode == "all":
        objective = pick(
            getattr(args, "objective", None),
            cfg["all"].get("objective"),
            "all.objective",
        )
        run_all(data_dir, str(objective))
        return

    if mode == "export":
        output_dir = resolve_path(
            getattr(args, "output_dir", None),
            cfg["app"].get("output_dir", "outputs"),
            cfg_base_dir,
        )
        run_export(data_dir, output_dir)
        return

    if mode == "validate":
        output_dir = resolve_path(
            getattr(args, "output_dir", None),
            cfg["validate"].get("output_dir", cfg["app"].get("output_dir", "outputs")),
            cfg_base_dir,
        )
        run_validate(data_dir, output_dir)
        return

    if mode == "agent":
        prompt = pick(
            getattr(args, "prompt", None),
            cfg["agent"].get("prompt"),
            "agent.prompt",
        )
        run_agent(data_dir, str(prompt))
        return

    raise SystemExit(f"Unknown mode: {mode}. Use single/all/export/validate/agent.")


if __name__ == "__main__":
    main()

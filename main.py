#!/usr/bin/env python3
"""HPC Infrastructure Optimizer — Multi-Agent System.

Finds optimal VM infrastructure (count_vm_1..4) for 88 HPC datasets
using specialist agents + Pareto front + TOPSIS.
"""

import sys
from pathlib import Path

import pandas as pd

from data_loader import find_data_dir, get_all_datasets, get_dataset, load_dataset
from agents import Coordinator
from agents.resource_agent import OBJECTIVE_COLUMN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(SCRIPT_DIR.parent / "data" / "SortedAvgDiffAll88Datasets")
OUTPUT_DIR = SCRIPT_DIR / "outputs"

OBJECTIVE_DESC = {
    "a": "energy + makespan + all_vm_count",
    "b": "energy + makespan + all_vm_mips",
    "c": "energy + makespan + all_vm_cores",
}

OUTPUT_COLUMNS = [
    "dataset_file", "tasks_count", "distribution", "objective",
    "count_vm_1", "count_vm_2", "count_vm_3", "count_vm_4",
    "total_energy_consumption", "simulation_time",
    "all_vm_count", "all_vm_mips", "all_vm_cores", "topsis_score",
]


def solve_single(data_dir, tasks, distribution, objective, verbose=True):
    """Solve one scenario and return the best candidate."""
    csv_path = get_dataset(data_dir, tasks, distribution)
    candidates = load_dataset(csv_path)
    if not candidates:
        print(f"  No valid candidates in {csv_path.name}")
        return None

    coordinator = Coordinator(objective)
    best, logs = coordinator.solve(candidates, verbose=verbose)

    if verbose:
        for line in logs:
            print(f"  {line}")

    return best, csv_path.name


def solve_all(data_dir, objective, verbose=True):
    """Solve all 88 datasets for one objective."""
    index = get_all_datasets(data_dir)
    results = []

    for i, ((tasks, dist), csv_path) in enumerate(sorted(index.items()), 1):
        candidates = load_dataset(csv_path)
        if not candidates:
            if verbose:
                print(f"  [{i:>2}/88] {csv_path.name}: no valid candidates, skipping")
            continue

        coordinator = Coordinator(objective)
        best, _ = coordinator.solve(candidates)

        if best:
            best["dataset_file"] = csv_path.name
            best["tasks_count"] = tasks
            best["distribution"] = dist
            best["objective"] = objective
            results.append(best)

        if verbose:
            quad = f"({best['count_vm_1']},{best['count_vm_2']},{best['count_vm_3']},{best['count_vm_4']})" if best else "NONE"
            print(f"  [{i:>2}/88] {csv_path.name}: {quad}")

    return results


def export_results(data_dir):
    """Export rec_a.csv, rec_b.csv, rec_c.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for obj in ("a", "b", "c"):
        print(f"\n  Objective {obj} ({OBJECTIVE_DESC[obj]}):")
        rows = solve_all(data_dir, obj, verbose=True)
        path = OUTPUT_DIR / f"rec_{obj}.csv"
        pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
        print(f"  -> Saved {len(rows)} rows to {path}")


def print_result(best, csv_name=None):
    """Pretty-print a single result."""
    if not best:
        print("\n  No result found.")
        return
    print(f"\n  Result{f' ({csv_name})' if csv_name else ''}:")
    print(f"  count_vm_1 = {best['count_vm_1']}")
    print(f"  count_vm_2 = {best['count_vm_2']}")
    print(f"  count_vm_3 = {best['count_vm_3']}")
    print(f"  count_vm_4 = {best['count_vm_4']}")
    print(f"  energy     = {best['total_energy_consumption']}")
    print(f"  makespan   = {best['simulation_time']}")
    print(f"  vm_count   = {best['all_vm_count']}")
    print(f"  vm_mips    = {best['all_vm_mips']}")
    print(f"  vm_cores   = {best['all_vm_cores']}")
    print(f"  TOPSIS     = {best.get('topsis_score', 'N/A')}")


def ask(prompt, default=None):
    """Ask user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else default


def menu():
    data_dir = DEFAULT_DATA_DIR

    while True:
        print("\n=== HPC Infrastructure Optimizer (Multi-Agent) ===")
        print(f"  Data: {data_dir}")
        print()
        print("  1. Solve single scenario")
        print("  2. Solve all 88 datasets (objective a: energy+makespan+vm_count)")
        print("  3. Solve all 88 datasets (objective b: energy+makespan+mips)")
        print("  4. Solve all 88 datasets (objective c: energy+makespan+cores)")
        print("  5. Export all results (rec_a.csv, rec_b.csv, rec_c.csv)")
        print("  6. Change data directory")
        print("  0. Exit")
        print()

        choice = ask("Choice", "1")

        if choice == "0":
            print("  Bye!")
            break

        elif choice == "1":
            tasks = ask("Number of tasks", "10000")
            dist = ask("Distribution (e.g. 50/50)", "50/50")
            obj = ask("Objective (a/b/c)", "a")
            print(f"\n  Solving: {tasks} tasks, {dist}, objective {obj} ({OBJECTIVE_DESC.get(obj, '?')})...")
            try:
                result = solve_single(data_dir, int(tasks), dist, obj)
                if result:
                    best, csv_name = result
                    print_result(best, csv_name)
            except Exception as e:
                print(f"  Error: {e}")

        elif choice in ("2", "3", "4"):
            obj = {"2": "a", "3": "b", "4": "c"}[choice]
            print(f"\n  Solving all 88 datasets for objective {obj} ({OBJECTIVE_DESC[obj]})...")
            try:
                rows = solve_all(data_dir, obj)
                print(f"\n  Done: {len(rows)}/88 datasets solved.")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            print("\n  Exporting all objectives...")
            try:
                export_results(data_dir)
                print("\n  All exports complete!")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "6":
            new_dir = ask("Path to SortedAvgDiffAll88Datasets", data_dir)
            try:
                find_data_dir(new_dir)
                data_dir = new_dir
                print(f"  Data directory set to: {data_dir}")
            except FileNotFoundError as e:
                print(f"  Error: {e}")

        else:
            print("  Unknown option, try again.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_DIR
        export_results(data_dir)
    else:
        menu()


if __name__ == "__main__":
    main()

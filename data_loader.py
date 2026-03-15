"""Load and filter HPC simulation CSV datasets."""

import re
from pathlib import Path

import pandas as pd

NAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.csv$")
DIST_RE = re.compile(r"^(\d{1,3})\D+(\d{1,3})$")

VM_MIPS = {1: 2000, 2: 4000, 3: 12000, 4: 16000}
VM_CORES = {1: 2, 2: 2, 3: 4, 4: 4}
VM_POWER = {1: 2, 2: 3, 3: 5, 4: 6}


def find_data_dir(data_dir=None):
    if data_dir:
        p = Path(data_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset directory not found: {p}")
        return p

    cwd = Path.cwd()
    for candidate in [cwd / "SortedAvgDiffAll88Datasets",
                      cwd.parent / "SortedAvgDiffAll88Datasets",
                      cwd.parent / "data" / "SortedAvgDiffAll88Datasets"]:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot find SortedAvgDiffAll88Datasets. Use --data-dir or place it nearby."
    )


def get_all_datasets(data_dir=None):
    root = find_data_dir(data_dir)
    index = {}
    for fp in sorted(root.glob("*.csv")):
        m = NAME_RE.match(fp.name)
        if not m:
            continue
        tasks = int(m.group(1))
        short = int(m.group(2))
        long_ = int(m.group(3))
        index[(tasks, f"{short}/{long_}")] = fp
    if not index:
        raise RuntimeError(f"No CSV datasets found in {root}")
    return index


def get_dataset(data_dir, tasks, distribution):
    index = get_all_datasets(data_dir)
    m = DIST_RE.match(str(distribution).strip())
    if not m:
        raise ValueError(f"Bad distribution format: {distribution}")
    key = (int(tasks), f"{m.group(1)}/{m.group(2)}")
    if key not in index:
        raise ValueError(f"Dataset not found for {key}")
    return index[key]


def _num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    success = _num(df.get("percentage_successful_tasks", pd.Series(dtype=float)), 0.0)
    df = df.loc[success >= 100.0].copy()
    if df.empty:
        return []

    vm1 = _num(df.get("count_vm_1", pd.Series(dtype=float))).astype(int)
    vm2 = _num(df.get("count_vm_2", pd.Series(dtype=float))).astype(int)
    vm3 = _num(df.get("count_vm_3", pd.Series(dtype=float))).astype(int)
    vm4 = _num(df.get("count_vm_4", pd.Series(dtype=float))).astype(int)
    makespan = _num(df.get("simulation_time", pd.Series(dtype=float))).astype(float)

    energy_csv = _num(df.get("total_energy_consumption", pd.Series(dtype=float)), float("nan"))
    energy_calc = ((vm1 * VM_POWER[1] + vm2 * VM_POWER[2]
                    + vm3 * VM_POWER[3] + vm4 * VM_POWER[4]) * makespan) / 1000.0
    energy = energy_csv.fillna(energy_calc).astype(float)

    vm_count = _num(df.get("all_vm_count", pd.Series(dtype=float)))
    vm_count = vm_count.where(vm_count > 0, vm1 + vm2 + vm3 + vm4).astype(int)

    vm_mips = _num(df.get("all_vm_mips", pd.Series(dtype=float)))
    vm_mips = vm_mips.where(
        vm_mips > 0,
        vm1 * VM_MIPS[1] + vm2 * VM_MIPS[2] + vm3 * VM_MIPS[3] + vm4 * VM_MIPS[4],
    ).astype(int)

    vm_cores = _num(df.get("all_vm_cores", pd.Series(dtype=float)))
    vm_cores = vm_cores.where(
        vm_cores > 0,
        vm1 * VM_CORES[1] + vm2 * VM_CORES[2] + vm3 * VM_CORES[3] + vm4 * VM_CORES[4],
    ).astype(int)

    result = pd.DataFrame({
        "count_vm_1": vm1, "count_vm_2": vm2, "count_vm_3": vm3, "count_vm_4": vm4,
        "total_energy_consumption": energy,
        "simulation_time": makespan,
        "all_vm_count": vm_count,
        "all_vm_mips": vm_mips,
        "all_vm_cores": vm_cores,
    })
    return result.to_dict(orient="records")

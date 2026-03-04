from pathlib import Path
import re
import pandas as pd

OBJECTIVE_TARGET = {
    "a": "all_vm_count",
    "b": "all_vm_mips",
    "c": "all_vm_cores",
}

OUTPUT_COLUMNS = [
    "dataset_file",
    "tasks_count",
    "distribution",
    "objective",
    "count_vm_1",
    "count_vm_2",
    "count_vm_3",
    "count_vm_4",
    "total_energy_consumption",
    "simulation_time",
    "all_vm_count",
    "all_vm_mips",
    "all_vm_cores",
    "score",
    "diff_energy_pct",
    "diff_time_pct",
    "diff_target_pct",
]

NAME_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)\.csv$")
DIST_PATTERN = re.compile(r"^(\d{1,3})\D+(\d{1,3})$")


def resolve_data_dir(data_dir=None):
    if data_dir:
        path = Path(data_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset directory not found: {path}")
        return path

    cwd = Path.cwd()
    candidates = [
        cwd / "SortedAvgDiffAll88Datasets",
        cwd.parent / "SortedAvgDiffAll88Datasets",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Cannot auto-detect SortedAvgDiffAll88Datasets. Use --data-dir /path/to/SortedAvgDiffAll88Datasets"
    )


def parse_distribution(distribution):
    m = DIST_PATTERN.match(str(distribution).strip())
    if not m:
        raise ValueError("distribution must look like 50/50")

    short_part = int(m.group(1))
    long_part = int(m.group(2))
    if short_part + long_part != 100:
        raise ValueError("distribution must sum to 100")
    return short_part, long_part


def build_dataset_index(data_dir):
    index = {}
    for file_path in Path(data_dir).glob("*.csv"):
        m = NAME_PATTERN.match(file_path.name)
        if not m:
            continue

        tasks = int(m.group(1))
        short_part = int(m.group(2))
        long_part = int(m.group(3))
        index[(tasks, short_part, long_part)] = file_path

    if not index:
        raise RuntimeError(f"no datasets found in {data_dir}")
    return index


def route_dataset(index, tasks_count, distribution):
    short_part, long_part = parse_distribution(distribution)
    path = index.get((int(tasks_count), short_part, long_part))
    if path is None:
        raise ValueError(f"dataset not found for ({tasks_count}, {short_part}/{long_part})")
    return path


def _num(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _text(df, col):
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str).str.strip()


def load_candidates(dataset_path):
    df = pd.read_csv(dataset_path)

    success = _num(df, "percentage_successful_tasks", 0.0)
    df = df.loc[success >= 100.0].copy()
    if df.empty:
        raise RuntimeError(f"no valid candidates in dataset: {dataset_path}")

    vm1 = _num(df, "count_vm_1", 0).astype(int)
    vm2 = _num(df, "count_vm_2", 0).astype(int)
    vm3 = _num(df, "count_vm_3", 0).astype(int)
    vm4 = _num(df, "count_vm_4", 0).astype(int)
    makespan = _num(df, "simulation_time", 0.0).astype(float)

    csv_energy = _num(df, "total_energy_consumption", float("nan"))
    calc_energy = ((vm1 * 2 + vm2 * 3 + vm3 * 5 + vm4 * 6) * makespan) / 1000.0
    energy = csv_energy.fillna(calc_energy).astype(float)

    all_vm_count = _num(df, "all_vm_count", 0)
    all_vm_count = all_vm_count.where(all_vm_count > 0, vm1 + vm2 + vm3 + vm4).astype(int)

    all_vm_mips = _num(df, "all_vm_mips", 0)
    all_vm_mips = all_vm_mips.where(
        all_vm_mips > 0,
        vm1 * 2000 + vm2 * 4000 + vm3 * 12000 + vm4 * 16000,
    ).astype(int)

    all_vm_cores = _num(df, "all_vm_cores", 0)
    all_vm_cores = all_vm_cores.where(
        all_vm_cores > 0,
        vm1 * 2 + vm2 * 2 + vm3 * 4 + vm4 * 4,
    ).astype(int)

    out = pd.DataFrame(
        {
            "dataset_file": dataset_path.name,
            "all_tasks_count": _num(df, "all_tasks_count", 0).astype(int),
            "short_long_tasks_distribution": _text(df, "short_long_tasks_distribution"),
            "simulation_time": makespan,
            "total_energy_consumption": energy,
            "all_vm_count": all_vm_count,
            "all_vm_mips": all_vm_mips,
            "all_vm_cores": all_vm_cores,
            "count_vm_1": vm1,
            "count_vm_2": vm2,
            "count_vm_3": vm3,
            "count_vm_4": vm4,
        }
    )
    return out.to_dict(orient="records")


def _pct_diff(value, min_value):
    if min_value == 0:
        return 0.0 if value == 0 else float("inf")
    return ((value - min_value) / min_value) * 100.0


def choose_best(objective, candidates):
    objective = str(objective).lower().strip()
    if objective not in OBJECTIVE_TARGET:
        raise ValueError("objective must be a, b or c")

    target_col = OBJECTIVE_TARGET[objective]

    min_energy = min(float(c["total_energy_consumption"]) for c in candidates)
    min_time = min(float(c["simulation_time"]) for c in candidates)
    min_target = min(float(c[target_col]) for c in candidates)

    best_key = None
    best = None

    for c in candidates:
        energy = float(c["total_energy_consumption"])
        makespan = float(c["simulation_time"])
        target_val = float(c[target_col])

        d_e = _pct_diff(energy, min_energy)
        d_t = _pct_diff(makespan, min_time)
        d_x = _pct_diff(target_val, min_target)
        score = (d_e + d_t + d_x) / 3.0

        key = (
            score,
            energy,
            makespan,
            target_val,
            float(c["all_vm_count"]),
            float(c["count_vm_1"]),
            float(c["count_vm_2"]),
            float(c["count_vm_3"]),
            float(c["count_vm_4"]),
        )

        if best_key is None or key < best_key:
            best_key = key
            best = {
                "dataset_file": c["dataset_file"],
                "tasks_count": int(c["all_tasks_count"]),
                "distribution": str(c["short_long_tasks_distribution"]),
                "objective": objective,
                "count_vm_1": int(c["count_vm_1"]),
                "count_vm_2": int(c["count_vm_2"]),
                "count_vm_3": int(c["count_vm_3"]),
                "count_vm_4": int(c["count_vm_4"]),
                "total_energy_consumption": round(energy, 6),
                "simulation_time": round(makespan, 6),
                "all_vm_count": int(c["all_vm_count"]),
                "all_vm_mips": int(c["all_vm_mips"]),
                "all_vm_cores": int(c["all_vm_cores"]),
                "score": round(score, 6),
                "diff_energy_pct": round(d_e, 6),
                "diff_time_pct": round(d_t, 6),
                "diff_target_pct": round(d_x, 6),
            }

    return best


def recommend_single(tasks_count, distribution, objective, data_dir=None):
    root = resolve_data_dir(data_dir)
    index = build_dataset_index(root)
    dataset_path = route_dataset(index, tasks_count, distribution)
    candidates = load_candidates(dataset_path)
    return choose_best(objective, candidates)


def _ordered_dataset_entries(index):
    # Stable order: tasks ascending, distribution 100/0 -> 0/100
    return sorted(index.items(), key=lambda kv: (kv[0][0], -kv[0][1]))


def recommend_all(objective, data_dir=None):
    root = resolve_data_dir(data_dir)
    index = build_dataset_index(root)

    rows = []
    for (tasks, short_part, long_part), dataset_path in _ordered_dataset_entries(index):
        candidates = load_candidates(dataset_path)
        best = choose_best(objective, candidates)

        # Keep naming consistent with file name to avoid malformed source data values.
        best["tasks_count"] = int(tasks)
        best["distribution"] = f"{short_part}/{long_part}"

        rows.append(best)

    return rows


def export_all(output_dir="outputs", data_dir=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exported = {}
    for objective in ("a", "b", "c"):
        rows = recommend_all(objective, data_dir)
        target = out / f"rec_{objective}.csv"
        pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(target, index=False)
        exported[objective] = str(target)

    return exported


def validate_project(data_dir=None, output_dir="outputs"):
    root = resolve_data_dir(data_dir)
    index = build_dataset_index(root)

    task_values = sorted({k[0] for k in index.keys()})
    dist_values = sorted({(k[1], k[2]) for k in index.keys()})

    datasets_without_success100 = []
    for path in index.values():
        df = pd.read_csv(path)
        if "percentage_successful_tasks" not in df.columns:
            datasets_without_success100.append(path.name)
            continue

        ok = (
            pd.to_numeric(df["percentage_successful_tasks"], errors="coerce").fillna(0) >= 100
        ).any()
        if not ok:
            datasets_without_success100.append(path.name)

    output_status = {}
    out = Path(output_dir)
    for objective in ("a", "b", "c"):
        file_path = out / f"rec_{objective}.csv"
        if not file_path.exists():
            output_status[objective] = {"exists": False}
            continue

        df = pd.read_csv(file_path)
        output_status[objective] = {
            "exists": True,
            "rows": int(len(df)),
            "objectives": sorted(df["objective"].dropna().unique().tolist()) if "objective" in df.columns else [],
            "unique_dataset_files": int(df["dataset_file"].nunique()) if "dataset_file" in df.columns else 0,
        }

    return {
        "datasets_count": len(index),
        "task_values": task_values,
        "distribution_values": [f"{a}/{b}" for a, b in dist_values],
        "full_grid": len(index) == len(task_values) * len(dist_values),
        "datasets_without_success100": datasets_without_success100,
        "output_status": output_status,
    }


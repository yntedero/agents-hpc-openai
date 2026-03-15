# HPC Infrastructure Optimizer — Multi-Agent System

University project (Tema 1): Agent-based approach to find optimal VM infrastructure for HPC simulations.

## Task

**Input:** number of tasks + distribution type (e.g. 10000 tasks, 50/50)
**Output:** VM quadruple `(count_vm_1, count_vm_2, count_vm_3, count_vm_4)` minimizing:

- **a)** energy + makespan + all_vm_count
- **b)** energy + makespan + all_vm_mips
- **c)** energy + makespan + all_vm_cores

Works for all 88 input datasets (8 task counts x 11 distributions).

## Method: Multi-Agent System + Pareto + TOPSIS

Three **specialist agents** each evaluate candidates by one criterion:

| Agent | Criterion |
|-------|-----------|
| EnergyAgent | total energy consumption |
| TimeAgent | simulation time (makespan) |
| ResourceAgent | vm_count / mips / cores (depends on objective) |

A **Coordinator** merges their proposals:
1. Each agent proposes its top-N candidates
2. Coordinator computes **Pareto front** (non-dominated solutions)
3. Coordinator applies **TOPSIS** ranking on the Pareto front
4. Returns the best compromise solution

## VM Types

| VM | MIPS | Cores | Power (W) |
|----|------|-------|-----------|
| VM1 | 2000 | 2 | 2 |
| VM2 | 4000 | 2 | 3 |
| VM3 | 12000 | 4 | 5 |
| VM4 | 16000 | 4 | 6 |

## Project Structure

```
agents-hpc-openai/
├── main.py              — interactive menu
├── data_loader.py       — CSV dataset loading
├── pareto.py            — Pareto front computation
├── topsis.py            — TOPSIS ranking
├── agents/
│   ├── base_agent.py    — base agent class
│   ├── energy_agent.py  — minimizes energy
│   ├── time_agent.py    — minimizes makespan
│   ├── resource_agent.py— minimizes resource metric
│   └── coordinator.py   — orchestrates agents
└── outputs/             — rec_a.csv, rec_b.csv, rec_c.csv
```

## Requirements

- Python 3.11+
- pandas, numpy

```bash
pip install -r requirements.txt
```

## Usage

Interactive mode:
```bash
python main.py
```

Export all results:
```bash
python main.py --export
```

## Pipeline

```
CloudSim (Java) → 88 CSV datasets → Multi-Agent System → VM recommendation
```

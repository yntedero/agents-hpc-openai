# agents-hpc-openai

Simple agent-based solution for **HPC Theme 1**:

Find a VM infrastructure tuple
`(count_vm_1, count_vm_2, count_vm_3, count_vm_4)`
from input:
- tasks count (for example `10000`)
- task distribution (for example `50/50`)
- objective (`a`, `b`, or `c`)

The project works over 88 prepared CSV datasets and returns one best VM tuple per scenario.

## Objective mapping

- `a`: minimize `energy + makespan + all_vm_count`
- `b`: minimize `energy + makespan + all_vm_mips`
- `c`: minimize `energy + makespan + all_vm_cores`

## How this fits CloudSim

- CloudSim generates simulation data.
- This project reads those CSV results.
- Agent/AI logic selects the best infrastructure tuple.

Pipeline:
`CloudSim -> 88 CSV datasets -> this project -> VM recommendation`

## Project structure

- `main.py` - CLI entrypoint and config-based run
- `core.py` - core selection logic
- `agent_runtime.py` - OpenAI agent tools wrapper
- `app_config.toml` - central config
- `outputs/` - generated `rec_a.csv`, `rec_b.csv`, `rec_c.csv`

## Requirements

- Python 3.11+
- `pandas`
- `openai-agents`
- `python-dotenv`

Install:

```bash
python -m pip install -r requirements.txt
```

## Environment

Create `.env` from `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key
```

`single`, `all`, `export`, `validate` modes do not require API calls.
`agent` mode requires `OPENAI_API_KEY`.

## Config-first run

Main settings are in `app_config.toml`.

Important fields:
- `[app].mode` = `single | all | export | validate | agent`
- `[app].data_dir` = path to `SortedAvgDiffAll88Datasets`
- `[app].output_dir` = output folder

Run with config:

```bash
python main.py
```

## CLI examples

```bash
python main.py single --tasks 10000 --distribution 50/50 --objective a
python main.py all --objective b
python main.py export --output-dir outputs
python main.py validate
python main.py agent --prompt "For 10000 tasks and 50/50 distribution solve objective a"
```

## Validation

`validate` checks:
- all 88 datasets are present
- full scenario grid exists (`8 task values x 11 distributions`)
- every dataset has at least one row with `percentage_successful_tasks >= 100`
- `rec_a.csv`, `rec_b.csv`, `rec_c.csv` are consistent

## Deliverables

Main generated outputs:
- `outputs/rec_a.csv`
- `outputs/rec_b.csv`
- `outputs/rec_c.csv`

Each file contains 88 recommendations (one per dataset).

"""Agent that minimizes the resource metric (vm_count / mips / cores)."""

from .base_agent import BaseAgent

OBJECTIVE_COLUMN = {
    "a": "all_vm_count",
    "b": "all_vm_mips",
    "c": "all_vm_cores",
}


class ResourceAgent(BaseAgent):
    name = "ResourceAgent"

    def __init__(self, objective="a"):
        if objective not in OBJECTIVE_COLUMN:
            raise ValueError(f"Objective must be a, b, or c (got {objective})")
        self.objective = objective
        self.criterion = OBJECTIVE_COLUMN[objective]
        self.name = f"ResourceAgent({objective}:{self.criterion})"

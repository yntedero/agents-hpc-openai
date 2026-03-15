"""Coordinator agent — merges specialist proposals via Pareto front + TOPSIS."""

from .energy_agent import EnergyAgent
from .time_agent import TimeAgent
from .resource_agent import ResourceAgent, OBJECTIVE_COLUMN
from pareto import pareto_front
from topsis import topsis_rank


class Coordinator:
    """Orchestrates specialist agents and selects the final VM configuration."""

    def __init__(self, objective="a", top_n=20):
        self.objective = objective
        self.top_n = top_n
        self.target_col = OBJECTIVE_COLUMN[objective]
        self.agents = [
            EnergyAgent(),
            TimeAgent(),
            ResourceAgent(objective),
        ]

    def solve(self, candidates, verbose=False):
        """Find the best VM quadruple for the given candidates.

        Returns (best_candidate, log_lines).
        """
        logs = []

        # 1. Each agent proposes its top-N
        proposals = set()
        for agent in self.agents:
            top = agent.evaluate(candidates, self.top_n)
            logs.append(agent.report(candidates))
            for c in top:
                key = (c["count_vm_1"], c["count_vm_2"], c["count_vm_3"], c["count_vm_4"])
                proposals.add(key)

        # 2. Collect unique candidates from proposals
        merged = []
        seen = set()
        for c in candidates:
            key = (c["count_vm_1"], c["count_vm_2"], c["count_vm_3"], c["count_vm_4"])
            if key in proposals and key not in seen:
                seen.add(key)
                merged.append(c)

        logs.append(f"[Coordinator] Merged proposals: {len(merged)} unique configurations")

        # 3. Pareto front
        objectives = ["total_energy_consumption", "simulation_time", self.target_col]
        front = pareto_front(merged, objectives)
        logs.append(f"[Coordinator] Pareto front: {len(front)} solutions")

        if not front:
            front = merged[:5] if merged else candidates[:5]
            logs.append("[Coordinator] Pareto front empty, using top merged candidates")

        # 4. TOPSIS ranking
        criteria = [
            ("total_energy_consumption", "min"),
            ("simulation_time", "min"),
            (self.target_col, "min"),
        ]
        ranked = topsis_rank(front, criteria)
        best = ranked[0] if ranked else None

        if best:
            logs.append(
                f"[Coordinator] TOPSIS best: "
                f"({best['count_vm_1']},{best['count_vm_2']},{best['count_vm_3']},{best['count_vm_4']}) "
                f"energy={best['total_energy_consumption']} "
                f"makespan={best['simulation_time']} "
                f"{self.target_col}={best[self.target_col]} "
                f"score={best['topsis_score']}"
            )

        return best, logs

"""Base class for specialist agents."""


class BaseAgent:
    """An agent that ranks candidates by a single criterion."""

    name: str = "BaseAgent"
    criterion: str = ""  # column name to sort by

    def evaluate(self, candidates, top_n=20):
        """Return *top_n* best candidates sorted by this agent's criterion."""
        ranked = sorted(candidates, key=lambda c: float(c[self.criterion]))
        return ranked[:top_n]

    def report(self, candidates, top_n=3):
        """Return a short text summary of the agent's top picks."""
        top = self.evaluate(candidates, top_n)
        if not top:
            return f"[{self.name}] No valid candidates."
        best = top[0]
        return (
            f"[{self.name}] Best: "
            f"({best['count_vm_1']},{best['count_vm_2']},{best['count_vm_3']},{best['count_vm_4']}) "
            f"{self.criterion}={best[self.criterion]}"
        )

"""Pareto front computation for multi-objective minimization."""


def dominates(a, b, objectives):
    """Return True if candidate *a* dominates *b* (all objectives minimized)."""
    dominated = False
    for obj in objectives:
        va, vb = float(a[obj]), float(b[obj])
        if va > vb:
            return False
        if va < vb:
            dominated = True
    return dominated


def pareto_front(candidates, objectives):
    """Return the Pareto-optimal subset (non-dominated candidates)."""
    front = []
    for c in candidates:
        is_dominated = False
        for other in candidates:
            if other is c:
                continue
            if dominates(other, c, objectives):
                is_dominated = True
                break
        if not is_dominated:
            front.append(c)
    return front

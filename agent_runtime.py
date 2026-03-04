import json

from core import export_all, recommend_all, recommend_single, validate_project


def build_agent(data_dir=None):
    # Lazy import so direct mode works without openai-agents installed.
    from agents import Agent, function_tool

    @function_tool
    def solve_single(tasks_count: int, distribution: str, objective: str) -> str:
        """Return best VM quadruple for one scenario."""
        result = recommend_single(tasks_count, distribution, objective, data_dir)
        return json.dumps(result, ensure_ascii=False)

    @function_tool
    def solve_all(objective: str) -> str:
        """Return summary for all 88 datasets for objective a/b/c."""
        rows = recommend_all(objective, data_dir)
        payload = {
            "objective": objective,
            "datasets_count": len(rows),
            "preview": rows[:5],
        }
        return json.dumps(payload, ensure_ascii=False)

    @function_tool
    def export_results(output_dir: str = "outputs") -> str:
        """Export rec_a.csv, rec_b.csv, rec_c.csv."""
        payload = export_all(output_dir=output_dir, data_dir=data_dir)
        return json.dumps(payload, ensure_ascii=False)

    @function_tool
    def validate_setup(output_dir: str = "outputs") -> str:
        """Validate datasets coverage and exported recommendation files."""
        payload = validate_project(data_dir=data_dir, output_dir=output_dir)
        return json.dumps(payload, ensure_ascii=False)

    return Agent(
        name="HPCSimpleAgent",
        instructions=(
            "You help choose VM infrastructure for HPC Theme 1. "
            "Use tools for calculations and validation, do not invent values."
        ),
        tools=[solve_single, solve_all, export_results, validate_setup],
    )

"""Agent that minimizes simulation time (makespan)."""

from .base_agent import BaseAgent


class TimeAgent(BaseAgent):
    name = "TimeAgent"
    criterion = "simulation_time"

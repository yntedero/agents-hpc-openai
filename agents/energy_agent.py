"""Agent that minimizes total energy consumption."""

from .base_agent import BaseAgent


class EnergyAgent(BaseAgent):
    name = "EnergyAgent"
    criterion = "total_energy_consumption"

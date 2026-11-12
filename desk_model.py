"""Stable types for Streamlit session state and cached plans."""
from dataclasses import dataclass

from scheduler import Metrics, Plan, Scenario


@dataclass(frozen=True)
class Settings:
    scene: str
    budget: int
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class CheckedPlan:
    scenario: Scenario
    plan: Plan
    metrics: Metrics
    status: str

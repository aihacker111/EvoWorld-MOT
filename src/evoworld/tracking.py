"""Tracker selection; v1 checkpoint inference retains the original hypothesis ecology."""

from evoworld.al_tracking import LifeTracker
from evoworld.legacy_tracking import EcologyTracker as LegacyEcologyTracker


class EcologyTracker:
    def __new__(cls, model):
        if model.config.tracker_mode == "legacy":
            return LegacyEcologyTracker(model)
        return LifeTracker(model)

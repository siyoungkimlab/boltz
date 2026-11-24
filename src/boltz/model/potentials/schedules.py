import math
from abc import ABC


class ParameterSchedule(ABC):
    def compute(self, t):
        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t):
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (
                math.exp(self.alpha * t) - 1
            ) / (math.exp(self.alpha) - 1)
        else:
            return self.start + (self.end - self.start) * t


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]


class LinearInterpolation(ParameterSchedule):
    """``start_val`` at ``start_t`` falling linearly to ``end_val`` at
    ``end_t``, and 0 outside that interval.

    Used by the ligand force-field presets for ``guidance_ramp``, where the
    guidance weight rises from 0 at t = 1 to its full value at t = 0 instead of
    switching on at a threshold.
    """

    def __init__(self, start_t, end_t, start_val, end_val):
        self.start_t = start_t
        self.end_t = end_t
        self.start_val = start_val
        self.end_val = end_val

    def compute(self, t):
        if (t > self.start_t) or (t < self.end_t):
            return 0.0
        return self.start_val - (self.start_val - self.end_val) * (
            self.start_t - t
        ) / (self.start_t - self.end_t)

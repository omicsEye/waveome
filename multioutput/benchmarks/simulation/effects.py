"""
Effect function factories and observation likelihoods for longitudinal
data simulation.
"""

import numpy as np
from typing import Callable

# =============================================================================
# 1. Effect Function Factories
# =============================================================================


def create_spike_effect(
    time_start: float, time_end: float, magnitude: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory to create a function that simulates a spike event."""
    def effect_func(t: np.ndarray) -> np.ndarray:
        return np.where((t >= time_start) & (t < time_end), magnitude, 0)
    return effect_func


def create_linear_increase_effect(
    slope: float, intercept_time: float = 0
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory to create a function that simulates a linear trend."""
    def effect_func(t: np.ndarray) -> np.ndarray:
        return slope * np.maximum(0, t - intercept_time)
    return effect_func


def create_trapezoid_effect(
    t_ramp_start: float, t_ramp_end: float, magnitude: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory for a flat->ramp->plateau effect.

    Returns 0 for t < t_ramp_start, a linear ramp from t_ramp_start to
    t_ramp_end, then magnitude for t >= t_ramp_end.
    """
    slope = magnitude / max(1e-6, t_ramp_end - t_ramp_start)

    def effect_func(t: np.ndarray) -> np.ndarray:
        return np.where(
            t < t_ramp_start, 0.0,
            np.where(t < t_ramp_end, slope * (t - t_ramp_start), magnitude)
        )
    return effect_func


def create_perturbation_effect(
    t_perturb: float, half_life: float, magnitude: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory for an instantaneous perturbation with exponential relaxation.

    f(t) = magnitude * exp(-ln(2)/half_life * (t - t_perturb)) for t >= t_perturb,
    0 otherwise. After 3 half-lives the effect is ~12.5% of magnitude.
    """
    decay_rate = np.log(2) / max(1e-6, half_life)

    def effect_func(t: np.ndarray) -> np.ndarray:
        return np.where(
            t >= t_perturb,
            magnitude * np.exp(-decay_rate * (t - t_perturb)),
            0.0
        )
    return effect_func


def create_periodic_effect(
    amplitude: float, period: float, phase: float = 0
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory to create a function that simulates a periodic effect
    (e.g., circadian)."""
    def effect_func(t: np.ndarray) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * t / period + phase)
    return effect_func


# =============================================================================
# 2. Likelihood Functions
# =============================================================================


def gaussian_likelihood(mu: np.ndarray, dispersion: float) -> np.ndarray:
    """Samples from a Gaussian distribution. dispersion = std dev."""
    return mu + np.random.normal(0, dispersion, size=mu.shape)


def negative_binomial_likelihood(
    mu: np.ndarray, dispersion: float
) -> np.ndarray:
    """Samples from a Negative Binomial distribution to model over-dispersed
    counts. variance = mu + mu^2 / dispersion."""
    mu = np.maximum(mu, 1e-6)
    n = dispersion
    p = n / (n + mu)
    return np.random.negative_binomial(n, p, size=mu.shape)

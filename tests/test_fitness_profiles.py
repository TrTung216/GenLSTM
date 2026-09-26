import numpy as np

from src.fitness_function import FITNESS_PROFILES, combined_fitness


def test_fitness_profiles_sum_to_one():
    for weights in FITNESS_PROFILES.values():
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_fitness_profiles_are_distinct_and_valid():
    assert set(FITNESS_PROFILES) == {"A", "B", "C"}
    for weights in FITNESS_PROFILES.values():
        assert all(0.0 <= value <= 1.0 for value in weights.values())


def test_combined_fitness_accepts_each_profile():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.02, -0.01, 0.01, -0.02])
    for weights in FITNESS_PROFILES.values():
        score = combined_fitness(y_true, y_pred, **weights)
        assert 0.0 <= score <= 1.0

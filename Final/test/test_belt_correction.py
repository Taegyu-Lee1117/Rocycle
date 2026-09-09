"""Unit tests for rocycle_robot.geometry.belt_correction.

Regression test using the 3 pen-marked ground-truth points from 3일차.
"""
from rocycle_robot.geometry.belt_correction import correct_belt_xy


def test_correction_matches_measured_ground_truth_within_1cm():
    cases = [
        ((266.70, -275.96), (251.28, -192.68)),
        ((312.85, -291.62), (315.16, -214.69)),
        ((366.95, -293.56), (415.15, -217.74)),
        ((321.66711012, -308.72222455), (339.80, -242.61)),
        ((411.96129097, -250.56058956), (487.79, -146.43)),
    ]
    for (raw_x, raw_y), (true_x, true_y) in cases:
        x, y = correct_belt_xy(raw_x, raw_y)
        assert abs(x - true_x) < 10.0
        assert abs(y - true_y) < 10.0


def test_correction_is_deterministic():
    a = correct_belt_xy(300.0, -250.0)
    b = correct_belt_xy(300.0, -250.0)
    assert a == b

"""Unit tests for ConstantVelocityKalman1D.

Includes the "정지 상태 예측 검증" invariant from v40/v44: with zero
velocity input, predicted position must equal the current measured
position (this is the software half of that check -- the hardware
half still needs a live stationary-belt test with the real camera).
"""

from rocycle_robot.geometry.kalman_1d import ConstantVelocityKalman1D


def test_zero_velocity_predicted_equals_current():
    kf = ConstantVelocityKalman1D(process_var=1.0, measurement_var=20.0)
    x_true = 300.0
    dt = 0.1
    for _ in range(30):
        kf.update(x_true, dt)

    assert abs(kf.state.vx) < 1.0  # settled near zero velocity
    predicted = kf.predict_position_at(dt_ahead=3.0)
    assert abs(predicted - x_true) < 2.0


def test_constant_velocity_tracked_and_predicted():
    kf = ConstantVelocityKalman1D(process_var=1.0, measurement_var=5.0)
    x0 = 100.0
    v_true = 20.0  # mm/s
    dt = 0.1
    t = 0.0
    x_true = x0
    for _ in range(50):
        kf.update(x_true, dt)
        t += dt
        x_true = x0 + v_true * t

    assert abs(kf.state.vx - v_true) < 2.0

    dt_ahead = 3.0
    predicted = kf.predict_position_at(dt_ahead)
    expected = x_true + v_true * dt_ahead
    assert abs(predicted - expected) < 5.0


def test_first_update_initializes_without_crashing():
    kf = ConstantVelocityKalman1D(process_var=1.0, measurement_var=20.0)
    state = kf.update(250.0, dt=0.1)
    assert state.x == 250.0
    assert state.vx == 0.0

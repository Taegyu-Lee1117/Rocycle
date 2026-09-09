"""1D constant-velocity Kalman filter for belt-direction (X) tracking.

[전제 원칙] Y/Z는 상수로 취급하고 X(벨트 진행 방향)만 추정한다 —
4일차 실측(그리퍼 자기 정렬이 Y축 오차를 흡수하지만 X축은 흡수하지
않음, CLAUDE.md 참고)이 이 설계를 뒷받침한다. 이 파일은 X 하나만
다룬다.

State: [x, vx] (mm, mm/s). Constant-velocity model:
    x_{k+1}  = x_k + vx_k * dt
    vx_{k+1} = vx_k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanState:
    x: float
    vx: float
    P: np.ndarray  # 2x2 covariance


class ConstantVelocityKalman1D:
    """Minimal constant-velocity KF for one axis.

    process_var: process noise on velocity (mm^2/s^3-ish, tune empirically).
    measurement_var: measurement noise on position (mm^2) -- start from the
    belt_correction RMS residual (4.5mm -> var ~= 20) as a reasoned default,
    see config default in tracking.yaml.
    """

    def __init__(
        self,
        process_var: float,
        measurement_var: float,
        initial_x: float = 0.0,
        initial_vx: float = 0.0,
        initial_pos_var: float = 1e4,
        initial_vel_var: float = 1e4,
    ) -> None:
        self.q = process_var
        self.r = measurement_var
        self.state = KalmanState(
            x=initial_x,
            vx=initial_vx,
            P=np.diag([initial_pos_var, initial_vel_var]).astype(float),
        )
        self._initialized = False

    def reset(self, x: float, vx: float = 0.0) -> None:
        self.state = KalmanState(x=x, vx=vx, P=np.diag([1e4, 1e4]).astype(float))
        self._initialized = True

    def predict(self, dt: float) -> KalmanState:
        """Advance the filter's own state estimate by dt (no measurement)."""
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * self.q
        x_vec = np.array([self.state.x, self.state.vx])
        x_vec = F @ x_vec
        P = F @ self.state.P @ F.T + Q
        self.state = KalmanState(x=float(x_vec[0]), vx=float(x_vec[1]), P=P)
        return self.state

    def update(self, z_x: float, dt: float) -> KalmanState:
        """Predict dt forward then fuse a new position measurement z_x."""
        if not self._initialized:
            self.reset(z_x)
            return self.state

        self.predict(dt)

        H = np.array([[1.0, 0.0]])
        z = np.array([z_x])
        x_vec = np.array([self.state.x, self.state.vx])

        y = z - H @ x_vec  # innovation
        S = H @ self.state.P @ H.T + self.r
        K = self.state.P @ H.T / S  # 2x1 gain

        x_vec = x_vec + (K.flatten() * y[0])
        P = (np.eye(2) - K @ H) @ self.state.P

        self.state = KalmanState(x=float(x_vec[0]), vx=float(x_vec[1]), P=P)
        return self.state

    def predict_position_at(self, dt_ahead: float) -> float:
        """Predicted X at (now + dt_ahead), without mutating filter state."""
        return self.state.x + self.state.vx * dt_ahead

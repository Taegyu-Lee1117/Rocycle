"""벨트 평면 예측 좌표에 대한 경험적 보정 — 3일차 실측 기반.

`pixel_to_base_point`(T_cam2base + belt_plane)의 예측이 벨트 구간
에서 실측 대비 체계적으로 압축되는 현상을 발견했다(3개 코너
펜 마킹 실측, 거리보존 검산 병행 확인). 근본 원인은 eye-to-hand
캘리브레이션 샘플의 깊이(카메라로부터 거리) 다양성 부족 — X,Y
방향 체커보드-그리퍼 오프셋 역산 표준편차는 1~2mm인데 Z(깊이)
방향만 7.7mm(최대 28mm)로 벌어짐, 이게 벨트처럼 캘리브레이션
샘플 범위 밖·먼 곳에서 오차로 증폭됨. 근본 수정(강성 마운트로
재캘리브레이션)은 오늘 시도했으나 마운팅 재현성 문제로 오히려
악화돼 보류 — 대신 실측 3점으로 피팅한 **평면 내 유사변환
(similarity transform)**으로 경험적 보정한다.

주의: 이 보정은 **belt_plane 위의 점에서만** 검증됐다. 다른
평면(예: 통 안쪽)에 적용하려면 그 평면에서 별도로 재보정해야
한다 — 오차의 원인(깊이 다양성 부족)이 평면마다 다르게 나타날
수 있기 때문(설계문서 v32 13절 39번 참고).
"""

from __future__ import annotations

import numpy as np

# 3일차 실측 5점(펜 마킹) 최소자승 피팅 결과.
# true_xy = A * raw_xy + B (A,B는 복소수, 유사변환: 배율+회전+평행이동)
# (raw_xy, true_xy는 base 프레임 (x,y), mm 단위, 복소수 x+iy로 표현)
_A = complex(1.650303241103378, 0.01761152097969898)
_B = complex(-197.71143039587997, 260.07775145570713)

# 5점 피팅 잔차(mm) — RMS 4.5mm, 최대 8.6mm. 이 이하 정밀도는 기대하지 말 것
FIT_RESIDUALS_MM = (4.48, 8.62, 2.12, 1.67, 1.26)
FIT_RMS_MM = 4.54


def correct_belt_xy(x: float, y: float) -> tuple[float, float]:
    """`pixel_to_base_point`가 belt_plane으로 예측한 (x,y)에 경험적
    보정을 적용해 실측에 더 가까운 (x,y)를 반환한다. z는 그대로 둔다
    (z 방향 오차는 별도로 특정되지 않았고, 펜 z-오프셋으로 이미
    일부 흡수됨).
    """
    raw = complex(x, y)
    corrected = _A * raw + _B
    return corrected.real, corrected.imag

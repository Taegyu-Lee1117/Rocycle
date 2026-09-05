"""최소 충돌 모델 — 웹 Claude v32 2절 제안.

완전한 경로 계획(MoveIt 등)이 아니라, "이 xy 영역에서는 z가 이
값 아래로 내려가지 않는다"는 단일 규칙만 강제한다(v32 2-5절
구현 범위, 5일 스코프 준수). 3일차 벨트 압박 사고(목표 z가
테이블면과 벨트면 사이 애매한 값이어서 벨트 프레임을 누른
사고)를 사전에 막는 것이 유일한 목적이다.

xy_bounds는 사각형 근사다 — 통처럼 원형인 구조물도 사각형으로
근사한다(v32 요청2 반론 대상, CLI 판단은 설계문서 참고: 과도하게
보수적일 수 있으나 안전 방향이라 5일 스코프에서는 허용).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class ObstacleCollisionError(ValueError):
    """목표 좌표가 구조물의 xy 범위 안이면서 z가 안전 하한 아래일 때."""


@dataclass(frozen=True)
class Obstacle:
    """물리 구조물 하나. xy_bounds 안에서는 z가 z_floor 아래로 못 내려간다.

    z_floor: 그 xy 영역에서 허용되는 최소 z(안전 여유 포함해서 호출
    측이 미리 계산해 넣을 것 — 이 클래스는 여유를 자동으로 더하지
    않는다).
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_floor: float
    note: str = ""

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def check_target_safe(x: float, y: float, z: float, obstacles: Iterable[Obstacle]) -> None:
    """목표 (x,y,z)가 어떤 obstacle의 xy 범위 안이면서 z_floor 아래면 예외를 던진다.

    여러 obstacle과 동시에 겹치면 가장 먼저 위반한 것을 보고한다
    (개별 xy 영역이 겹치지 않게 설계하는 것을 권장 — 겹치면 어느
    쪽이든 위반 시 거부되므로 안전성 자체는 유지된다).
    """
    for obs in obstacles:
        if obs.contains_xy(x, y) and z < obs.z_floor:
            raise ObstacleCollisionError(
                f"target ({x:.2f}, {y:.2f}, {z:.2f})는 '{obs.name}' 영역 "
                f"(x:[{obs.x_min},{obs.x_max}] y:[{obs.y_min},{obs.y_max}]) 안이고 "
                f"z_floor={obs.z_floor:.2f}보다 낮습니다. {obs.note}".rstrip()
            )


def load_obstacles_from_dict(data: dict) -> list[Obstacle]:
    """설정 파일(9절 스키마 `obstacles` 블록)에서 읽은 dict를 Obstacle 리스트로 변환.

    값이 없는(null) 항목은 건너뛴다 — 아직 티칭 전인 통 등.
    """
    obstacles: list[Obstacle] = []
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        bounds = cfg.get("xy_bounds")
        z_floor = cfg.get("z_floor")
        if bounds is None or z_floor is None:
            continue
        x_min, x_max, y_min, y_max = bounds
        obstacles.append(
            Obstacle(
                name=name,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_floor=z_floor,
                note=cfg.get("note", ""),
            )
        )
    return obstacles

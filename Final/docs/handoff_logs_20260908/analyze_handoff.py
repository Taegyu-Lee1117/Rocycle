#!/usr/bin/env python3
"""핸드오버 폴링 로그 분석 -- 회차별 속도/변위 집계.

사용법:
    python3 ~/analyze_handoff.py ~/handoff_test_1744.log [더 많은 로그...]

각 로그에서 `[HANDOFF] poll ...` 라인을 파싱해 아래를 뽑는다:
  - 수평속도 피크(순간 최고값)와 상위 5개, 중앙값
  - `-Z 제외` 누적변위 최대 (dz_effective = max(dz, 0.0))
  - 기존 3D 변위 최대 (수정 전 값과 비교용)
  - 릴리즈 판정 재생(임의 임계값으로 시뮬레이션 가능)

**주의 -- 속도 값의 해상도 한계**: 로그의 `speed_horiz`는 순간값이지만
`poll_interval_sec`(기본 0.3초) 구간의 평균이다. 0.3초보다 짧은 과도
스파이크는 평균에 희석돼 안 보인다. "제어 대역폭 한계" 가설을 더
정밀하게 보려면 폴링 주기를 줄여야 하는데, 그러면 `pull_speed_consecutive`
(연속 N회)의 실제 시간 의미도 같이 바뀌므로 판정 동작이 달라진다.
"""
import re
import statistics
import sys

POLL = re.compile(
    r"poll elapsed=([\d.]+)s pos=\(([\d.\-]+),([\d.\-]+),([\d.\-]+)\).*?"
    r"dt=([\d.]+)s speed_horiz=([\d.]+)mm/s"
)
BASE = re.compile(r"baseline captured, start_pos=\[([\d.\-]+), ([\d.\-]+), ([\d.\-]+)\]")


def parse(path):
    """로그 하나에서 (start_pos, 폴링 리스트)를 뽑는다. 한 로그에 여러
    회차가 있으면 baseline이 나올 때마다 새 회차로 나눈다."""
    runs = []
    start = None
    polls = []
    for line in open(path):
        m = BASE.search(line)
        if m:
            if start is not None and polls:
                runs.append((start, polls))
            start = tuple(float(x) for x in m.groups())
            polls = []
            continue
        m = POLL.search(line)
        if m and start is not None:
            polls.append(
                (
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                )
            )
    if start is not None and polls:
        runs.append((start, polls))
    return runs


def summarize(start, polls, speed_th=3.0, disp_th=10.0, consec=2):
    speeds = [p[5] for p in polls]
    disp_new = []
    disp_3d = []
    for _, x, y, z, _, _ in polls:
        dx, dy, dz = x - start[0], y - start[1], z - start[2]
        disp_new.append((dx * dx + dy * dy + max(dz, 0.0) ** 2) ** 0.5)
        disp_3d.append((dx * dx + dy * dy + dz * dz) ** 0.5)

    fast = 0
    fired = None
    for i, p in enumerate(polls):
        fast = fast + 1 if p[5] >= speed_th else 0
        if fired is None and fast >= consec and disp_new[i] > disp_th:
            fired = (p[0], p[5], disp_new[i])

    print(f"  폴링 {len(polls)}회, 지속 {polls[-1][0]:.2f}초, start_pos={start}")
    print(f"  수평속도  피크 {max(speeds):.1f} mm/s   상위5 {sorted(speeds, reverse=True)[:5]}")
    print(f"            중앙값 {statistics.median(speeds):.1f}  평균 {statistics.mean(speeds):.1f}")
    print(f"  누적변위  -Z제외 최대 {max(disp_new):.2f} mm   (기존 3D 최대 {max(disp_3d):.1f} mm)")
    if fired:
        print(f"  판정(speed>={speed_th}, disp>{disp_th}, 연속{consec}): "
              f"릴리즈 at {fired[0]:.2f}s (speed={fired[1]:.1f}, disp={fired[2]:.1f}mm)")
    else:
        print(f"  판정(speed>={speed_th}, disp>{disp_th}, 연속{consec}): 릴리즈 안 함 (타임아웃)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for path in sys.argv[1:]:
        runs = parse(path)
        if not runs:
            print(f"\n##### {path} -- 폴링 데이터 없음")
            continue
        for i, (start, polls) in enumerate(runs, 1):
            label = f"{path}" + (f"  (회차 {i}/{len(runs)})" if len(runs) > 1 else "")
            print(f"\n##### {label}")
            summarize(start, polls)
    return 0


if __name__ == "__main__":
    sys.exit(main())

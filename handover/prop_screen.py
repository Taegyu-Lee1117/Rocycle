#!/usr/bin/env python3
"""소품 스크리닝: 벨트로 물체를 화면 좌->우로 흘리며 x 위치별 승자 클래스를 기록.
   전 구간에서 기대 클래스가 100%로 나오는 소품만 촬영에 사용한다.
   사용법: python3 prop_screen.py <기대클래스>        예) python3 prop_screen.py paper
"""
import rclpy, time, json, sys
from rclpy.node import Node
from std_msgs.msg import String
from collections import Counter

EXPECT = sys.argv[1] if len(sys.argv) > 1 else None
BELT_Y_MIN = 470.0          # tracking_pick_node 의 belt_pixel_y_min
ANCHOR_R   = 0.95           # y1 + 0.95*(y2-y1)
X_STOP     = 1230           # 이 x2 를 넘으면 벨트 정지(끝에서 떨어짐 방지)

rclpy.init(); n = Node('prop_screen')
pub = n.create_publisher(String, '/conveyor_command', 10)
rows = []

def cb(msg):
    try: dets = json.loads(msg.data)
    except Exception: return
    best = {}
    for d in dets:
        bb = d.get('bbox')
        if not bb: continue
        y1, y2 = float(bb[1]), float(bb[3])
        if y1 + (y2 - y1) * ANCHOR_R < BELT_Y_MIN: continue
        c = d['class_name']
        if c not in best or d['confidence'] > best[c][0]:
            best[c] = (d['confidence'], [float(v) for v in bb])
    if not best: return
    w = max(best.items(), key=lambda kv: kv[1][0])
    x1, y1, x2, y2 = w[1][1]
    rows.append((time.time(), x1, x2, y1, y2, w[0], w[1][0],
                 {k: v[0] for k, v in best.items()}))

n.create_subscription(String, '/recycle_detection/detections', cb, 10)

def belt(cmd):
    for _ in range(5):
        pub.publish(String(data=cmd)); rclpy.spin_once(n, timeout_sec=0.1)

print(f"기대 클래스: {EXPECT or '(지정 안 함)'} -- 벨트 START")
belt('START')
# [버그수정] 정지조건이 검출에만 걸려 있어서, 검출이 거의 안 되는 물체는
# 240초 내내 벨트가 돌아 끝에서 떨어졌다(장갑 실측). 벨트 실측속도
# 4.98mm/s로 화면 전폭(약 680px = 365mm)을 지나는 데 약 73초이므로
# 하드 상한을 95초로 둔다 -- 검출 여부와 무관하게 반드시 멈춘다.
MAX_RUN_S = 95
t0 = time.time(); reason = f'상한 {MAX_RUN_S}초 도달'
while time.time() - t0 < MAX_RUN_S:
    rclpy.spin_once(n, timeout_sec=0.2)
    if rows and rows[-1][2] > X_STOP:
        reason = '우측 끝 도달'; break
belt('STOP')
print(f"벨트 {time.time()-t0:.0f}s 구동 후 정지 ({reason})\n")

if not rows:
    print("!! 벨트 영역에서 검출된 프레임이 없습니다."); rclpy.shutdown(); sys.exit(1)

# 100px 구간별 집계
buckets = {}
for r in rows:
    cx = (r[1] + r[2]) / 2
    b = int(cx // 100) * 100
    buckets.setdefault(b, Counter())[r[5]] += 1

print(f"{'cx 구간':<12} {'프레임':>5}  구간별 승자 분포")
bad = []
for b in sorted(buckets):
    c = buckets[b]; tot = sum(c.values())
    dist = "  ".join(f"{k} {v*100//tot}%" for k, v in c.most_common())
    flag = ''
    if EXPECT and c.most_common(1)[0][0] != EXPECT:
        flag = '   <== 불일치'; bad.append(b)
    print(f"{b:4d}-{b+99:<7d} {tot:5d}  {dist}{flag}")

win = Counter(r[5] for r in rows); tot = sum(win.values())
print(f"\n전체 {tot}프레임: " + "  ".join(f"{k} {v}({v*100//tot}%)" for k, v in win.most_common()))
if EXPECT:
    ok = not bad and win.most_common(1)[0][0] == EXPECT and win[EXPECT] == tot
    print(f"\n판정: {'통과 -- 전 구간 ' + EXPECT + ' 100%, 소품으로 사용 가능' if ok else '탈락 -- 위치에 따라 클래스가 바뀜, 소품으로 부적합'}")
rclpy.shutdown()

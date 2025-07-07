#!/usr/bin/env python3
# ======================================================================
#  run_all.py · v1.5.1  (UTF-8 hardening + Step-5 loop + progress monitor + range fix)
#  Author  : CDL · 2025-07-07 改进版
# ======================================================================
from __future__ import annotations
import subprocess, sys, pathlib, os, time, re, json, atexit

# ─── 路径常量 ─────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[1]
PY3_DIR   = ROOT / "PY3"
ENGINE    = PY3_DIR / "step5_sampling_engine.py"
PY        = sys.executable
ENV       = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # 子脚本也 UTF-8

# 监控文件
MON_DIR   = ROOT / "sampling_engine"
MON_DIR.mkdir(exist_ok=True, parents=True)
PROGRESS  = MON_DIR / "progress.json"
LOG_PATH  = MON_DIR / "progress.log"
PID_PATH  = MON_DIR / "pid.txt"

# ─── 把自身 PID 写入文件，退出时删除 ────────────────────────────────
PID_PATH.write_text(str(os.getpid()))
atexit.register(lambda: PID_PATH.unlink(missing_ok=True))

# ─── 简易写进度工具函数 ─────────────────────────────────────────────
def write_progress(**kw):
    """ kw: stage / grid / done / total / message """
    kw["ts"] = time.time()
    PROGRESS.write_text(json.dumps(kw, ensure_ascii=False))

# ─── 设定经纬度范围过滤 ──────────────────────────────────────────────
def in_range(grid_id: str,
             lat_min=49.4, lat_max=50.6,
             lon_min=-4.6, lon_max=1.6) -> bool:
    try:
        lat_str, lon_str = grid_id.split("_")
        lat, lon = float(lat_str), float(lon_str)
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    except Exception:
        return False

# ─── 五大步骤脚本列表 ───────────────────────────────────────────────
STEPS = [
    ("Step-1  清洗",            PY3_DIR / "step1_load_and_clean.py"),
    ("Step-2  趋势分析",        PY3_DIR / "step2_analysis.py"),
    ("Step-3  Kriging/盲区",    PY3_DIR / "step3_kriging_blindzone.py"),
    ("Step-4  Priority & 报告", PY3_DIR / "step4_priority_report.py"),
]

# ─── 统一运行子脚本的封装 ───────────────────────────────────────────
def run(name: str, cmd: list[str]):
    print(f"\n＝＝＝ {name} ＝＝＝")
    write_progress(stage=name)                         # 写阶段进度
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=ENV,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1)
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        for ln in proc.stdout:
            print(ln, end="")                          # 控制台
            log_f.write(ln)                            # 写日志文件
    if proc.wait():
        write_progress(stage=name, message="FAILED")
        sys.exit(f"{name} FAILED")

# ─── Step-1 ~ Step-4 依次执行 ───────────────────────────────────────
for stage_name, script in STEPS:
    run(stage_name, [PY, str(script)])

# ─── Step-5 采样引擎（多网格循环，范围限定） ─────────────────────────
ts_csv = ROOT / "ch4_sampling_result/ts_df.csv"
import pandas as pd
grids = pd.read_csv(ts_csv, usecols=["grid_id"])["grid_id"].unique().tolist()
print(f"\n＝＝＝ Step-5  Sampling Engine : {len(grids)} grids ＝＝＝")
write_progress(stage="Step-5", total=len(grids))

for i, gid in enumerate(grids, 1):
    if not in_range(gid):
        print(f"Grid {gid} 超出限定范围，跳过计算。")
        # 写进度确保前端进度正确
        write_progress(stage="Step-5", grid=gid, done=i, total=len(grids))
        continue

    t0 = time.perf_counter()
    write_progress(stage="Step-5", grid=gid, done=i, total=len(grids))
    print(f"[{i:>3}/{len(grids)}] {gid}")
    try:
        run(f"Sampling {gid}", [
            PY, str(ENGINE),
            "--grid_id", gid, "--eps", "5",
            "--boot", "1000", "--w", "0.25,0.25,0.25,0.25"
        ])
    except Exception as e:
        print(f"Sampling {gid} FAILED: {e}")
        write_progress(stage="Step-5", grid=gid, done=i, total=len(grids), message="FAILED")
        sys.exit(f"Sampling {gid} FAILED")

    print(f"    ✓ {gid}  {time.perf_counter()-t0:.1f}s")

print("\n✓ backend 5-step pipeline finished.")
write_progress(stage="FINISHED")

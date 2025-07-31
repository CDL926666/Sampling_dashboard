# Y:\Bishe_project\PY3\run_all.py

from __future__ import annotations
import subprocess
import sys
import pathlib
import os
import time
import json
import atexit

# 区域和采样参数
EU_LAT_MIN, EU_LAT_MAX = 35.0, 72.0
EU_LON_MIN, EU_LON_MAX = -25.0, 45.0
DEFAULT_EPS = 5.0
DEFAULT_BOOT = 1000
DEFAULT_W = "0.25,0.25,0.25,0.25"

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY3_DIR = ROOT / "PY3"
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# 监控目录和状态文件
MON_DIR = ROOT / "sampling_engine"
MON_DIR.mkdir(exist_ok=True, parents=True)
PROGRESS = MON_DIR / "progress.json"
LOG_PATH = MON_DIR / "progress.log"
PID_PATH = MON_DIR / "pid.txt"
PID_PATH.write_text(str(os.getpid()))
atexit.register(lambda: PID_PATH.unlink(missing_ok=True))

# 写入进度到json文件
def write_progress(**kw):
    kw["ts"] = time.time()
    PROGRESS.write_text(json.dumps(kw, ensure_ascii=False))

# 判断格点是否在有效范围
def in_range(grid_id: str,
             lat_min=EU_LAT_MIN, lat_max=EU_LAT_MAX,
             lon_min=EU_LON_MIN, lon_max=EU_LON_MAX) -> bool:
    try:
        lat_str, lon_str = grid_id.split("_")
        lat, lon = float(lat_str), float(lon_str)
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    except Exception:
        return False

# 需要依次执行的步骤
STEPS = [
    ("Step-1  Clean",        PY3_DIR / "step1_load_and_clean.py"),
    ("Step-2  Analysis",     PY3_DIR / "step2_analysis.py"),
    ("Step-3  Kriging",      PY3_DIR / "step3_kriging_blindzone.py"),
    ("Step-4  Priority",     PY3_DIR / "step4_priority_report.py"),
]

# 调用子进程运行各步骤脚本
def run_step(name: str, cmd: list[str]):
    print(f"\n==== {name} ====")
    write_progress(stage=name)
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=ENV,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        for ln in proc.stdout:
            print(ln, end="")
            log_f.write(ln)
    if proc.wait():
        write_progress(stage=name, message="FAILED")
        sys.exit(f"{name} FAILED")

# 顺序执行所有主步骤
for stage_name, script in STEPS:
    run_step(stage_name, [PY, str(script)])

import pandas as pd
ts_csv = ROOT / "ch4_sampling_result/ts_df.csv"
if not ts_csv.exists():
    sys.exit(f"ts_df.csv not found at {ts_csv}")
grids = pd.read_csv(ts_csv, usecols=["grid_id"])["grid_id"].unique().tolist()
print(f"\n==== Step-5 Sampling Engine: {len(grids)} grids ====")
write_progress(stage="Step-5", total=len(grids))

ENGINE = PY3_DIR / "step5_sampling_engine.py"
skipped = []
successful = []

# 按格点逐个采样
for i, gid in enumerate(grids, 1):
    if not in_range(gid):
        print(f"Grid {gid} out of region, skipped.")
        write_progress(stage="Step-5", grid=gid, done=i, total=len(grids))
        skipped.append({"grid_id": gid, "reason": "out_of_region"})
        continue
    t0 = time.perf_counter()
    write_progress(stage="Step-5", grid=gid, done=i, total=len(grids))
    print(f"[{i:>4}/{len(grids)}] {gid} ...")
    try:
        run_step(f"Sampling {gid}", [
            PY, str(ENGINE),
            "--grid_id", gid,
            "--eps", str(DEFAULT_EPS),
            "--boot", str(DEFAULT_BOOT),
            "--w", DEFAULT_W
        ])
        successful.append(gid)
        print(f"    ✓ {gid}  {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        msg = f"Sampling {gid} failed: {e}"
        print(f"Warning: {msg}")
        write_progress(stage="Step-5", grid=gid, done=i, total=len(grids), message="FAILED")
        skipped.append({"grid_id": gid, "reason": str(e)})
        continue

# 统计并输出最终结果
pd.DataFrame(skipped).to_csv(MON_DIR / "skipped_grids.csv", index=False)
pd.DataFrame({"grid_id": successful}).to_csv(MON_DIR / "successful_grids.csv", index=False)

print(f"\nStep-5 Sampling summary:")
print(f"  Grids completed: {len(successful)}")
print(f"  Grids skipped/failed: {len(skipped)}")
if skipped:
    print("  See skipped_grids.csv for details.")
if successful:
    print("  See successful_grids.csv for completed grid IDs.")

print("\n✓ Backend 5-step pipeline finished.")
write_progress(stage="FINISHED")

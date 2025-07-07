#!/usr/bin/env python3
# ======================================================================
#  uploader.py · v1.9.1  (优化进程管理、日志flush及前端刷新调用)
# ======================================================================
from __future__ import annotations
import io, sys, shutil, tempfile, subprocess, textwrap, gzip, pathlib, os, re, time, json, atexit
from datetime import datetime

import pandas as pd
import streamlit as st
from filelock import FileLock, Timeout

# ── CONFIG ────────────────────────────────────────────────────────────
MAX_PART_MB  = 200
COMPRESS_OUT = False

RAW_ROOT     = pathlib.Path("user_uploads")
FINAL_PATH   = pathlib.Path("final/output_ch4_flux_qc.csv")
BACKEND_CMD  = [sys.executable, "PY3/run_all.py"]
LOCK_PATH    = pathlib.Path("sampling_engine/run_all.lock")

# 进度 / 日志 / PID 文件（与 run_all.py / app.py 保持一致）
MON_DIR   = pathlib.Path("sampling_engine")
PROGRESS  = MON_DIR / "progress.json"
LOG_PATH  = MON_DIR / "progress.log"
PID_PATH  = MON_DIR / "pid.txt"

for p in (RAW_ROOT, LOCK_PATH.parent, MON_DIR):
    p.mkdir(parents=True, exist_ok=True)

_log_re = re.compile(r"(error|fail|exception|traceback|warning)", re.I)
_mb = lambda b: round(b / 2**20, 2)

# 安全清理PID文件，防止残留
def cleanup_pid():
    try:
        PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
atexit.register(cleanup_pid)  # 程序退出自动清理

# ── 工具函数 ──────────────────────────────────────────────────────────
def _detect_enc(p: pathlib.Path) -> str:
    try:
        import magic
        mime = magic.from_file(str(p), mime=True)
        if "charset=" in mime:
            ch = mime.split("charset=")[-1].lower()
            return "gbk" if ch.startswith("gb") else "utf-8"
    except ImportError:
        pass
    return "utf-8"

def _merge(parts: list[pathlib.Path], dst: pathlib.Path) -> pathlib.Path:
    tot = sum(p.stat().st_size for p in parts)
    done, bar = 0, st.progress(0.0, text="Merging …")
    if COMPRESS_OUT and dst.suffix != ".gz":
        dst = dst.with_suffix(dst.suffix + ".gz")
        opener = lambda p: gzip.open(p, "wt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "w", encoding="utf-8")

    with opener(dst) as fout:
        for i, part in enumerate(parts):
            with open(part, encoding=_detect_enc(part), errors="ignore") as fin:
                if i: next(fin)                         # 跳过额外表头
                for chunk in iter(lambda: fin.read(1 << 20), ""):
                    if not chunk:
                        break
                    fout.write(chunk)
                    done += len(chunk.encode())
                    bar.progress(done / tot,
                                 text=f"Merging … {_mb(done)}/{_mb(tot)} MB")
    bar.empty()
    return dst

# ── 进程控制按钮区域，单独封装，确保任何状态下都显示 ─────────────────
def _render_stop_button():
    st.markdown("---")
    st.caption("💻 进程控制")
    if PID_PATH.exists():
        try:
            pid = PID_PATH.read_text().strip()
            st.warning(f"后端运行中，PID={pid}")
            if st.button("⛔ 强制终止后台任务"):
                try:
                    os.kill(int(pid), 9)
                    cleanup_pid()
                    # 清理进度状态，防止前端误判后台仍在运行
                    PROGRESS.write_text(json.dumps({"stage": "FINISHED", "ts": time.time()}))
                    st.success("已尝试终止后台任务。请刷新页面。")
                except Exception as e:
                    st.error(f"终止失败：{e}")
        except Exception as e:
            st.error(f"读取 PID 文件出错: {e}")
    else:
        st.success("后端空闲，可新任务上传。")

# ── UI 主入口 ─────────────────────────────────────────────────────────
def render() -> None:
    st.subheader("📤 上传 CH₄ CSV（≤ 200 MB / part）")

    # 检查后台运行状态，不阻断上传界面，只禁用上传按钮
    running_stage = None
    is_running = False
    if PROGRESS.exists():
        try:
            info = json.loads(PROGRESS.read_text())
            running_stage = info.get("stage")
            is_running = running_stage not in ("FINISHED", "starting")
        except Exception:
            pass

    if is_running:
        st.info(f"后端正在运行：{running_stage}，上传按钮已禁用，请稍候。")

    files = st.file_uploader(
        "一次选完所有分片；若单文件＞200 MB，请先本地切分",
        type="csv",
        accept_multiple_files=True,
        key="uploader",
        disabled=is_running,
    )

    # 无论后台是否运行，都显示停止按钮，方便强制停止
    _render_stop_button()

    if not files:
        st.info("等待文件 …")
        return

    # ---------- 保存上传临时文件 ----------
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="csv_parts_"))
    parts: list[pathlib.Path] = []
    oversize = False
    with st.status("保存上传文件 …", expanded=False) as stat:
        for uf in files:
            buf = uf.getbuffer()
            mb = _mb(len(buf))
            stat.write(f"• {uf.name} {mb:.2f} MB")
            if mb > MAX_PART_MB:
                oversize = True
                stat.write(f"  ❌ 超出 {MAX_PART_MB} MB 限制")
            else:
                p = tmpdir / uf.name
                p.write_bytes(buf)
                parts.append(p)
        stat.update(label="上传完毕 ✓" if not oversize else "上传中断",
                    state="complete" if not oversize else "error")
    if oversize:
        st.error(f"有文件超过 {MAX_PART_MB} MB，请切分后再上传。")
        return

    parts.sort()
    st.success(f"共 {len(parts)} 分片，全部 ≤ {MAX_PART_MB} MB")

    # ---------- 表头一致性检查 ----------
    hdrs = [open(p, encoding=_detect_enc(p), errors="ignore").readline().strip()
            for p in parts]
    if len(set(hdrs)) != 1:
        st.error("❌ 分片表头不一致，请检查文件。")
        return
    st.success("Header 一致性通过")

    # ---------- 预览 ----------
    if st.toggle("预览合并后前 5 行"):
        df_prev = pd.concat(
            (pd.read_csv(p, nrows=5, encoding=_detect_enc(p)) for p in parts),
            ignore_index=True).head(5)
        st.dataframe(df_prev)

    # ---------- 合并 + 运行后端 ----------
    if st.button("🚀 Merge & Run backend"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 备份分片
        raw_dir = RAW_ROOT / ts / "parts"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for p in parts: shutil.copy2(p, raw_dir / p.name)
        st.write(f"📦 分片备份 → {raw_dir}")

        FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged = _merge(parts, FINAL_PATH)
        st.success(f"✅ 合并完成 → {merged}")

        # 清空旧监控文件，写入初始状态
        PROGRESS.write_text(json.dumps({"stage": "starting", "ts": time.time()}))
        LOG_PATH.write_text("")

        # ---------- 调后台脚本并实时刷新 ----------
        try:
            with FileLock(str(LOCK_PATH), timeout=1):
                stage_bar = st.progress(0.0)
                sub_bar   = st.progress(0.0)
                stage_txt = st.empty()
                sub_txt   = st.empty()

                t0 = time.perf_counter()
                grids_done = total_grids = 0
                step_idx   = 0
                log_bad: list[str] = []

                proc = subprocess.Popen(
                    BACKEND_CMD,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )

                # 把 PID 写入文件，供前端终止
                PID_PATH.write_text(str(proc.pid))

                with open(LOG_PATH, "a", encoding="utf-8") as log_f:
                    for line in proc.stdout:
                        log_f.write(line)
                        log_f.flush()  # 立即写入日志，保证实时更新
                        if _log_re.search(line):
                            log_bad.append(line.rstrip())

                        if m := re.search(r"＝{3,}\s*(Step-\d)", line):
                            step_idx += 1
                            stage_txt.text(f"⚙️ {m.group(1)} …")
                            stage_bar.progress(min(step_idx / 5, 1.0))
                            continue

                        if "Step-5" in line and "grids" in line:
                            m2 = re.search(r"(\d+)\s+grids", line)
                            if m2:
                                total_grids = int(m2.group(1))
                            continue

                        if m3 := re.match(r"\[\s*(\d+)/(\d+)]", line):
                            grids_done = int(m3.group(1))
                            total_grids = int(m3.group(2))
                            elapsed = time.perf_counter() - t0
                            eta = elapsed / max(grids_done, 1) * (total_grids - grids_done)
                            sub_txt.text(f"📈 Sampling {grids_done}/{total_grids} "
                                         f"(ETA ≈ {int(eta//60)} m {int(eta%60)} s)")
                            sub_bar.progress(grids_done / total_grids)
                            continue

                rc = proc.wait()
                stage_bar.progress(1.0)
                sub_bar.progress(1.0)

                if log_bad:
                    st.error("⚠️ 发现关键告警行：")
                    st.code("\n".join(log_bad[-200:]), language="bash")
                else:
                    st.success("流程全部 OK，详细日志见 console。")

                # 清理 PID
                cleanup_pid()

                if rc == 0:
                    from streamlit.runtime.caching import cache_data
                    cache_data.clear()
                    PROGRESS.write_text(json.dumps({"stage": "FINISHED", "ts": time.time()}))
                    # 这里不直接调用 st.rerun(), 由用户手动刷新页面
                else:
                    PROGRESS.write_text(json.dumps({"stage": "FAILED", "ts": time.time()}))
                    st.error("后端脚本返回非零，详见上方告警。")

        except Timeout:
            st.warning("⚠️ 后端正被其他任务占用，请稍后再试。")

st.caption("Uploader v1.9.1 © CDL")

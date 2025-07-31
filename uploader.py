# Y:\Bishe_project\uploader.py 

from __future__ import annotations
import io, sys, shutil, tempfile, subprocess, gzip, pathlib, os, re, time, json, atexit
from datetime import datetime

import pandas as pd
import streamlit as st
from filelock import FileLock, Timeout

MAX_PART_MB  = 200
COMPRESS_OUT = False

RAW_ROOT   = pathlib.Path("user_uploads")
FINAL_PATH = pathlib.Path("final/output_ch4_flux_qc.csv")

BACKEND_CMD = [sys.executable, "PY3/run_all.py"]
LOCK_PATH   = pathlib.Path("sampling_engine/run_all.lock")

MON_DIR   = pathlib.Path("sampling_engine")
PROGRESS  = MON_DIR / "progress.json"
LOG_PATH  = MON_DIR / "progress.log"
PID_PATH  = MON_DIR / "pid.txt"

for p in (RAW_ROOT, LOCK_PATH.parent, MON_DIR, FINAL_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)

_log_re = re.compile(r"(error|fail|exception|traceback|warning)", re.I)
_mb     = lambda b: round(b / 2**20, 2)

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _cleanup_pid():
    PID_PATH.unlink(missing_ok=True)
atexit.register(_cleanup_pid)

def _detect_enc(p: pathlib.Path) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                f.readline()
            return enc
        except Exception:
            continue
    return "utf-8"

def _merge(parts: list[pathlib.Path], dst: pathlib.Path) -> pathlib.Path:
    total = sum(p.stat().st_size for p in parts)
    done  = 0
    bar   = st.progress(0.0, "Merging files…")
    opener = gzip.open if COMPRESS_OUT else open
    mode   = "wt" if COMPRESS_OUT else "w"

    with opener(dst, mode=mode, encoding="utf-8") as fout:
        for idx, part in enumerate(parts):
            with open(part, encoding=_detect_enc(part)) as fin:
                if idx: next(fin)
                for line in fin:
                    fout.write(line)
                    done += len(line.encode())
                    if done % (5 << 20) < len(line):
                        bar.progress(done / total, f"Merging {_mb(done)}/{_mb(total)} MB")
    bar.empty()
    return dst

def _backend_stage() -> str|None:
    try:
        if PROGRESS.exists():
            return json.loads(PROGRESS.read_text()).get("stage")
    except Exception:
        pass
    return None

def _render_backend_status():
    st.caption("Backend progress / log")
    stage = _backend_stage()
    if stage:
        st.write(f"Current stage: {stage}")
    else:
        st.info("No backend progress record.")
    try:
        if stage == "Step-5":
            info = json.loads(PROGRESS.read_text())
            st.progress(info.get("done", 0) / max(info.get("total", 1), 1))
        ts = json.loads(PROGRESS.read_text()).get("ts") if stage else None
        if ts:
            st.caption(time.strftime("Updated: %Y-%m-%d %H:%M:%S", time.localtime(ts)))
    except Exception:
        pass
    try:
        if LOG_PATH.exists():
            logs = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
            if logs:
                st.code("\n".join(logs[-12:]), language="bash")
    except Exception as e:
        st.error(f"Read log failed: {e}")
    if st.button("Refresh backend status", key="refresh_backend"):
        if _backend_stage() == "FINISHED":
            try:
                from streamlit.runtime.caching import cache_data
                cache_data.clear()
            except Exception:
                pass
            st.session_state.pop("datasets", None)
        _safe_rerun()

def _render_stop_button():
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        if st.button(f"Terminate backend (PID {pid})"):
            try:
                os.kill(pid, 9)
                _cleanup_pid()
                PROGRESS.write_text(json.dumps({"stage": "FINISHED", "ts": time.time()}))
                st.success("Backend terminated.")
                st.session_state.pop("datasets", None)
            except Exception as e:
                st.error(f"Terminate failed: {e}")
    else:
        st.success("Backend idle.")

def _launch_backend():
    with FileLock(str(LOCK_PATH), timeout=5):
        out_fh = open(LOG_PATH, "a", encoding="utf-8")
        proc = subprocess.Popen(
            BACKEND_CMD,
            stdout=out_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            start_new_session=True
        )
        PID_PATH.write_text(str(proc.pid))
        PROGRESS.write_text(json.dumps({"stage": "starting", "ts": time.time()}))
        st.success(f"Backend started (PID {proc.pid}).")

def render() -> None:
    _render_backend_status()
    _render_stop_button()
    stage_now = _backend_stage()
    uploading_disabled = stage_now not in (None, "FINISHED")
    if uploading_disabled:
        st.info("Backend running, upload disabled.")

    files = st.file_uploader(
        "Upload CH₄ CSV parts (≤200 MB each)",
        type="csv",
        accept_multiple_files=True,
        disabled=uploading_disabled,
    )
    if not files:
        st.info("Waiting for file upload…")
        return

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="upload_"))
    parts, oversize = [], False
    with st.status("Saving uploaded files…", expanded=False) as status:
        for uf in files:
            size_mb = _mb(len(uf.getbuffer()))
            status.write(f"- {uf.name} ({size_mb:.2f} MB)")
            if size_mb > MAX_PART_MB:
                oversize = True
                status.write(f"Exceeds {MAX_PART_MB} MB limit")
            else:
                p = tmpdir / uf.name
                p.write_bytes(uf.getbuffer())
                parts.append(p)
        status.update(label="Upload successful" if not oversize else "Upload interrupted",
                      state="complete" if not oversize else "error")
    if oversize:
        shutil.rmtree(tmpdir)
        st.error("Some files exceeded size limit.")
        return

    hdrs = [open(p, encoding=_detect_enc(p)).readline().strip() for p in parts]
    if len(set(hdrs)) != 1:
        shutil.rmtree(tmpdir)
        st.error("Inconsistent headers detected.")
        return
    st.success("Header check passed.")

    if st.toggle("Preview first 5 rows"):
        preview = pd.concat(
            [pd.read_csv(p, encoding=_detect_enc(p), nrows=5) for p in parts],
            ignore_index=True,
        ).head()
        st.dataframe(preview)

    if st.button("Merge & Run backend"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_dir = RAW_ROOT / ts
        raw_dir.mkdir(parents=True, exist_ok=True)
        for p in parts:
            shutil.copy2(p, raw_dir)
        merged = _merge(parts, FINAL_PATH)
        shutil.rmtree(tmpdir)
        st.success(f"Merged → {merged}")
        try:
            from streamlit.runtime.caching import cache_data
            cache_data.clear()
        except Exception:
            pass
        st.session_state.pop("datasets", None)
        try:
            _launch_backend()
        except Timeout:
            st.warning("Another backend task is running.")
        except Exception as exc:
            st.error(f"Failed to start backend: {exc}")
            _cleanup_pid()

st.caption("© CDL")

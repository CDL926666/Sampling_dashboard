#!/usr/bin/env python3
# ======================================================================
#  app.py  ·  Three-Tab Entry  (Spatial / Optimizer / Upload)
#  Author : CDL · 2025-07
# ======================================================================
from __future__ import annotations
import time
import streamlit as st
import os, pathlib, json

import tab_spatial
import tab_optimizer
import uploader
from common import load_all, numeric_safe_cast

# 1. 必须先 set_page_config
st.set_page_config(
    page_title="CH₄ Scientific-Sampling Dashboard",
    layout="wide",
    page_icon="🛰️",
)

# ──────── 关键数据文件检查 ────────
DATA_DIR = pathlib.Path("ch4_sampling_result")
TS_DF_PATH = DATA_DIR / "ts_df.csv"

has_data = False
df = blind_df = hist_df = ts_df = None

if TS_DF_PATH.exists():
    _fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

    @_fragment  # type: ignore[arg-type]
    def _load_data():
        df_main, df_blind, df_hist, df_ts = load_all()
        return numeric_safe_cast(df_main), df_blind, df_hist, df_ts

    if "datasets" not in st.session_state:
        st.session_state["datasets"] = _load_data()

    df, blind_df, hist_df, ts_df = st.session_state["datasets"]
    has_data = df is not None and not df.empty

# ──────── 后台监控区 ────────
MON_DIR = pathlib.Path("sampling_engine")
PROGRESS = MON_DIR / "progress.json"
LOG_PATH = MON_DIR / "progress.log"
PID_PATH = MON_DIR / "pid.txt"

if "refresh_key" not in st.session_state:
    st.session_state["refresh_key"] = 0

def manual_refresh():
    st.session_state["refresh_key"] += 1

with st.container():
    c1, c2, c3 = st.columns([2, 4, 2])

    with c1:
        st.caption(f"🗂️ 当前工作目录：{os.getcwd()}")
        result_dir = pathlib.Path("ch4_sampling_result")
        result_files = list(result_dir.glob("*"))
        st.caption(
            f"📁 ch4_sampling_result/ 目录状态：{'存在' if result_dir.exists() else '缺失'}\n"
            f"文件列表：\n" +
            ("\n".join(f"  - {f.name}" for f in result_files) if result_files else "  （无文件）")
        )

    with c2:
        st.caption("⏳ 后台进度 / 日志")
        try:
            if PROGRESS.exists():
                info = json.loads(PROGRESS.read_text())
                running_stage = info.get("stage")
                st.write(f"**当前阶段：** {running_stage or '-'}")
                if running_stage == 'Step-5':
                    done = info.get('done', 0)
                    total = info.get('total', 1)
                    st.progress(done / max(total, 1))
                ts = info.get('ts')
                if ts:
                    st.caption(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}")
            else:
                st.info("暂无后台进度记录。")
        except Exception as e:
            st.error(f"读取进度文件出错: {e}")

        try:
            if LOG_PATH.exists():
                logs = LOG_PATH.read_text(encoding='utf-8', errors='ignore').splitlines()
                if logs:
                    st.code("\n".join(logs[-12:]), language="bash")
        except Exception as e:
            st.error(f"读取日志文件出错: {e}")

    with c3:
        st.caption("💻 进程控制")
        if PID_PATH.exists():
            try:
                pid = PID_PATH.read_text().strip()
                st.warning(f"后端运行中，PID={pid}")
                if st.button("⛔ 强制终止后台任务"):
                    try:
                        import os
                        os.kill(int(pid), 9)
                        PID_PATH.unlink(missing_ok=True)
                        PROGRESS.write_text(json.dumps({"stage": "FINISHED", "ts": time.time()}))
                        st.success("已尝试终止后台任务。请刷新页面。")
                    except Exception as e:
                        st.error(f"终止失败：{e}")
            except Exception as e:
                st.error(f"读取 PID 文件出错: {e}")
        else:
            st.success("后端空闲，可新任务上传。")

    if st.button("🔄 手动刷新后台状态"):
        manual_refresh()

_ = st.empty()
_.text(f"刷新计数: {st.session_state['refresh_key']}")

# ──────── Tab 布局 ────────
if has_data:
    tab_sp, tab_op, tab_up = st.tabs(
        ["📌 Spatial Priority", "🧮 Sampling Optimizer", "📤 Upload"]
    )
else:
    tab_up, tab_sp, tab_op = st.tabs(
        ["📤 Upload", "📌 Spatial Priority", "🧮 Sampling Optimizer"]
    )

with tab_up:
    uploader.render()
    if not has_data:
        st.info("📥 还没有数据，请先上传文件生成主数据。")

with tab_sp:
    if has_data:
        tab_spatial.render(df, blind_df, hist_df, ts_df)
    else:
        st.info("📥 请先在 “Upload” 页上传并生成主数据。")

with tab_op:
    if has_data:
        tab_optimizer.render(df, ts_df)
    else:
        st.info("📥 暂无主数据，完成上传及后台计算后再试。")

st.toast("Dashboard ready ✓" if has_data else "请先上传数据⚠️", icon="✅" if has_data else "ℹ️")

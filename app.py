# Y:\Bishe_project\app.py

from __future__ import annotations
import os, json, time, pathlib
import streamlit as st

import tab_spatial, tab_optimizer, uploader
from common import load_all, numeric_safe_cast

# 页面配置
st.set_page_config(page_title="CH4 Scientific-Sampling Dashboard", layout="wide")

# 常量
DATA_DIR   = pathlib.Path("ch4_sampling_result")
TS_DF_PATH = DATA_DIR / "ts_df.csv"
MON_DIR    = pathlib.Path("sampling_engine")
PROGRESS   = MON_DIR / "progress.json"
PID_PATH   = MON_DIR / "pid.txt"

# Streamlit rerun（兼容不同版本）
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Session 初始化
if "datasets" not in st.session_state:
    st.session_state["datasets"] = (None, None, None, None)
if "backend_info" not in st.session_state:
    st.session_state["backend_info"] = {}

# 数据加载
def _load_main_data() -> tuple:
    main, blind, hist, ts = load_all()
    main = numeric_safe_cast(main) if main is not None else None
    return main, blind, hist, ts

def refresh_dataset_cache(force: bool = False) -> None:
    need = force or st.session_state["datasets"][0] is None \
        or getattr(st.session_state["datasets"][0], "empty", True)
    if need and TS_DF_PATH.exists():
        st.session_state["datasets"] = _load_main_data()

# 进程控制和状态
def render_process_control() -> None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("Process control")
        if PID_PATH.exists():
            try:
                pid = int(PID_PATH.read_text().strip())
                st.warning(f"Backend running, PID={pid}")
                if st.button("Force terminate backend task"):
                    try:
                        os.kill(pid, 9)
                        PID_PATH.unlink(missing_ok=True)
                        PROGRESS.write_text(json.dumps({"stage": "FINISHED", "ts": time.time()}))
                        st.success("Backend killed, refresh the page.")
                    except Exception as e:
                        st.error(f"Terminate failed: {e}")
            except Exception as e:
                st.error(f"Read PID failed: {e}")
        else:
            st.success("Backend idle. Ready for new upload.")
    with col2:
        if st.button("Refresh backend status"):
            try:
                if PROGRESS.exists():
                    with open(PROGRESS, encoding="utf-8") as f:
                        st.session_state["backend_info"] = json.load(f)
                else:
                    st.session_state["backend_info"] = {}
            except Exception as e:
                st.warning(f"Read progress.json failed: {e}")
            if st.session_state["backend_info"].get("stage") == "FINISHED":
                try:
                    from streamlit.runtime.caching import cache_data
                    cache_data.clear()
                except Exception:
                    pass
                st.session_state["datasets"] = (None, None, None, None)
                refresh_dataset_cache(force=True)
            safe_rerun()

def main() -> None:
    st.title("CH₄ Scientific-Sampling Dashboard")
    render_process_control()
    refresh_dataset_cache()
    df, blind_df, hist_df, ts_df = st.session_state["datasets"]
    has_data = df is not None and not df.empty

    if has_data:
        tab_sp, tab_op, tab_up = st.tabs(["Spatial Priority", "Sampling Optimizer", "Upload"])
    else:
        tab_up, tab_sp, tab_op = st.tabs(["Upload", "Spatial Priority", "Sampling Optimizer"])

    with tab_up:
        try:
            uploader.render()
            if not has_data:
                st.info("No data yet, please upload a file first.")
        except Exception as e:
            st.error(f"Uploader error: {e}")

    with tab_sp:
        if has_data:
            try:
                tab_spatial.render(df, blind_df, hist_df, ts_df)
            except Exception as e:
                st.error(f"Spatial tab error: {e}")
        else:
            st.info("Need valid data, go to Upload tab first.")

    with tab_op:
        if has_data:
            try:
                tab_optimizer.render(df, ts_df)
            except Exception as e:
                st.error(f"Optimizer tab error: {e}")
        else:
            st.info("Need valid data, finish upload and backend steps first.")

    st.toast("Dashboard ready." if has_data else "Awaiting data upload.")

# 程序入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("A serious system error occurred, please refresh or contact the administrator.")
        st.code(str(e))

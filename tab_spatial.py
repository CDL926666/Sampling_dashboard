#!/usr/bin/env python3
# ======================================================================
#  tab_spatial.py  ·  Spatial Priority 页面（云端兼容版）
#  Author: CDL + ChatGPT 2025-07
# ======================================================================
from __future__ import annotations
import io, glob, zipfile, pathlib, os
import numpy as np
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from common import DIR, FILES, PRIO_LV, PRIO_COL

# ─── 保证云端地图可用 ────────────────────────────────────────────────
pio.templates.default = "plotly_dark"
px.set_mapbox_access_token(
    "pk.eyJ1IjoicGxvdGx5dXNlciIsImEiOiJjaWZ4dmFhOG8wMDU2dW9vY2dyd2Z6N3RjIn0.2dUAQKqKAF-6EMbbVfwn9w"
)

# ======================================================================
def render(df: pd.DataFrame, blind_df: pd.DataFrame,
           hist_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Filters")
        sel_lv = st.multiselect("Priority Level", PRIO_LV, default=PRIO_LV)
        show_blind = st.checkbox("Show only blind-zones", value=False,
                                 disabled=blind_df.empty)
        lat_rng = st.slider("Latitude Range",
                            float(df.latitude.min()), float(df.latitude.max()),
                            (float(df.latitude.min()), float(df.latitude.max())))
        lon_rng = st.slider("Longitude Range",
                            float(df.longitude.min()), float(df.longitude.max()),
                            (float(df.longitude.min()), float(df.longitude.max())))
        kw = st.text_input("Fuzzy search (visible cols)")

        st.divider()
        st.caption(f"🔧 Working dir: **{pathlib.Path.cwd()}**")

        # ---------- ZIP download ----------
        with st.expander("Download result files (.zip)"):
            if st.button("Build ZIP"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in glob.glob(f"{DIR}/*"):
                        zf.write(f, arcname=os.path.basename(f))
                buf.seek(0)
                st.session_state["zip"] = buf.getvalue()
            if "zip" in st.session_state:
                st.download_button("⬇️ Download ZIP", st.session_state["zip"],
                                   file_name="ch4_outputs.zip")

    # ---------- Data masking ----------
    mask = df["sampling_recommendation"].isin(sel_lv)
    mask &= df.latitude.between(*lat_rng) & df.longitude.between(*lon_rng)
    if show_blind and "blind_spot_flag" in df:
        mask &= df.blind_spot_flag.eq("blind_zone")
    if kw:
        cols = ["grid_id", "sampling_recommendation", "blind_spot_flag"]
        mask &= df[cols].astype(str).apply(
            lambda s: s.str.contains(kw, case=False, regex=False)
        ).any(axis=1)
    view = df[mask].copy()

    # ---------- Title ----------
    st.title("Methane (CH₄) Scientific-Sampling Dashboard")

    # ---------- Map ----------
    st.subheader("Spatial Priority Map & Blind-spots")
    if view.empty:
        st.warning("当前筛选条件下没有数据，请调整过滤条件后再试！")
    else:
        view["std_ch4"] = pd.to_numeric(view["std_ch4"], errors="coerce").fillna(1.0)
        size_cap = np.nanpercentile(view["std_ch4"], 99) or 1.0

        # 转成 list，防止 Series 兼容问题
        lat = view.latitude.astype(float).tolist()
        lon = view.longitude.astype(float).tolist()
        color = view.sampling_recommendation.astype(str).tolist()
        size = view.std_ch4.astype(float).tolist()

        try:
            fig_map = px.scatter_mapbox(
                lat=lat, lon=lon, color=color,
                category_orders={"color": PRIO_LV},
                color_discrete_map=PRIO_COL,
                size=size, size_max=15, zoom=4, height=560,
                hover_name=view.grid_id.tolist(),
            )
            fig_map.update_traces(marker={"sizeref": size_cap / 20})
            if not blind_df.empty:
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=blind_df.latitude.tolist(),
                        lon=blind_df.longitude.tolist(),
                        name="Blind Zone", mode="markers",
                        marker=dict(color="black", size=14, symbol="x"),
                    )
                )
            fig_map.update_layout(
                mapbox_style="carto-positron",
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning(f"地图加载失败，降级普通底图：{e}")
            fig_map = px.scatter_mapbox(
                lat=lat, lon=lon, color=color,
                category_orders={"color": PRIO_LV},
                color_discrete_map=PRIO_COL,
                size=size, size_max=15, zoom=4, height=560,
            )
            fig_map.update_layout(mapbox_style="open-street-map",
                                  margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

    # ---------- Statistics ----------
    st.subheader("Priority Statistics")
    if view.empty:
        st.info("没有可统计的数据。")
    else:
        counts = view["sampling_recommendation"].value_counts() \
                      .reindex(PRIO_LV).fillna(0)
        labels = counts.index.astype(str).tolist()
        values = counts.values.astype(int).tolist()

        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Grids (filtered)", len(view))
            st.bar_chart(counts)
        with c2:
            if sum(values) == 0:
                st.info("暂无数据绘制饼图。")
            else:
                fig_pie = go.Figure(go.Pie(labels=labels, values=values))
                st.plotly_chart(fig_pie, use_container_width=True)

        if not hist_df.empty:
            st.bar_chart(
                hist_df.set_index(hist_df.priority_score_bins.astype(str))["count"],
                height=180, use_container_width=True,
            )

    # ---------- Top-N ----------
    st.subheader("Top-N Priority Grids & Trend")
    top_n = st.slider("Show Top-N", 10, 100, 20) if not view.empty else 0
    top_df = (view.sort_values("priority_score", ascending=False).head(top_n)
              if top_n else pd.DataFrame())
    if not top_df.empty:
        st.dataframe(
            top_df[["grid_id", "latitude", "longitude", "priority_score",
                    "sampling_recommendation", "mean_ch4", "std_ch4",
                    "obs_count", "outlier_count", "changepoint_day"]],
            use_container_width=True, hide_index=True,
        )

        idx = st.number_input("Select row (0-n)", 0, len(top_df)-1, 0)
        gid = top_df.iloc[int(idx)]["grid_id"]
        sub = ts_df[ts_df.grid_id == gid]
        if not sub.empty:
            try:
                dates, vals = zip(*[x.split(",") for x in
                                    sub.iloc[0].time_series.split("|")])
                dates = pd.to_datetime(dates).tolist()
                vals  = list(map(float, vals))
                fig_ts = go.Figure(go.Scatter(x=dates, y=vals,
                                              mode="lines+markers"))
                fig_ts.update_layout(title=f"Grid {gid} — Historical CH₄",
                                     yaxis_title="Flux")
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.error(f"趋势图绘制失败: {e}")
    else:
        st.info("No grids after filters.")

    # ---------- Blind-spot List ----------
    st.subheader("Blind-spot List")
    st.dataframe(blind_df if not blind_df.empty
                 else pd.DataFrame({"info": ["None"]}),
                 use_container_width=True, hide_index=True)

    # ---------- Report & README ----------
    with st.expander("📄 Auto-generated Report"):
        path = DIR / FILES["report"]
        st.code(path.read_text("utf-8"), language="markdown") \
            if path.exists() else st.info("Report missing.")

    with st.expander("ℹ️ Field Description"):
        path = DIR / FILES["readme"]
        st.code(path.read_text("utf-8"), language="markdown") \
            if path.exists() else st.info("README missing.")

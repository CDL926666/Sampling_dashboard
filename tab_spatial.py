# tab_spatial.py
import io, glob, zipfile, pathlib
import numpy as np
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
import os

from common import DIR, FILES, PRIO_LV, PRIO_COL

def render(df: pd.DataFrame, blind_df: pd.DataFrame,
           hist_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Filters")
        sel_lv = st.multiselect(
            "Priority Level", PRIO_LV, default=PRIO_LV,
            help="勾选想要显示的优先级，全部勾选显示全部"
        )
        show_blind = st.checkbox(
            "Show only blind-zones",
            value=False,
            disabled=blind_df.empty,
        )
        lat_rng = st.slider(
            "Latitude Range",
            float(df.latitude.min()), float(df.latitude.max()),
            (float(df.latitude.min()), float(df.latitude.max())),
        )
        lon_rng = st.slider(
            "Longitude Range",
            float(df.longitude.min()), float(df.longitude.max()),
            (float(df.longitude.min()), float(df.longitude.max())),
        )
        kw = st.text_input("Fuzzy search (visible cols)")

        st.divider()
        st.caption(f"🔧 Working dir: **{pathlib.Path.cwd()}**")

        # ---------- ZIP download ----------
        with st.expander("Download result files (.zip)"):
            if st.button("Build ZIP"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in glob.glob(f"{DIR}/*"):
                        zf.write(f, arcname=f.split('/')[-1])
                buf.seek(0)
                st.session_state["zip"] = buf.getvalue()
            if "zip" in st.session_state:
                st.download_button("⬇️ Download ZIP",
                                   st.session_state["zip"],
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

    # ---------- Debug 输出 ----------
    st.write("【DEBUG】筛选后数据：", view.shape)
    st.write("【DEBUG】分组内容 unique:", view["sampling_recommendation"].unique())
    st.write("【DEBUG】各优先级计数:", view["sampling_recommendation"].value_counts(dropna=False))
    st.write(view.head())

    # ---------- Title ----------
    st.title("Methane (CH₄) Scientific-Sampling Dashboard")

    # ---------- Map ----------
    st.subheader("Spatial Priority Map & Blind-spots")
    if view.empty:
        st.warning("当前筛选条件下没有数据，请在侧边栏调整优先级/经纬度/模糊搜索后再试！")
    else:
        # 确保 std_ch4 类型正确
        view["std_ch4"] = pd.to_numeric(view["std_ch4"], errors="coerce").fillna(1.0)
        size_cap = np.nanpercentile(view["std_ch4"], 99) or 1.0

        fig_map = px.scatter_mapbox(
            view, lat="latitude", lon="longitude",
            color="sampling_recommendation",
            category_orders={"sampling_recommendation": PRIO_LV},
            color_discrete_map=PRIO_COL,
            size="std_ch4", size_max=15, zoom=4, height=560,
            hover_data={
                "grid_id": True, "priority_score": ":.3f", "std_ch4": ":.3f",
                "trend_slope": ":.3f", "obs_count": True,
                "outlier_count": True, "blind_spot_flag": True,
            },
        )
        fig_map.update_traces(marker={"sizeref": size_cap / 20})
        if not blind_df.empty and len(blind_df) > 0:
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=blind_df.latitude, lon=blind_df.longitude,
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

    # ---------- Statistics ----------
    st.subheader("Priority Statistics")
    if view.empty:
        st.info("没有可统计的数据，请调整筛选条件。")
    else:
        counts = view["sampling_recommendation"].value_counts() \
                      .reindex(PRIO_LV).fillna(0)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Grids (filtered)", len(view))
            st.bar_chart(counts)
        with c2:
            st.plotly_chart(
                go.Figure(go.Pie(labels=counts.index,
                                 values=counts.values)),
                use_container_width=True,
            )
        if not hist_df.empty:
            st.bar_chart(
                hist_df.set_index(hist_df.priority_score_bins.astype(str))["count"],
                height=180,
                use_container_width=True,
            )

    # ---------- Top-N ----------
    st.subheader("Top-N Priority Grids & Trend")
    top_n = st.slider("Show Top-N", 10, 100, 20) if not view.empty else 0
    top_df = (view.sort_values("priority_score", ascending=False).head(top_n)
              if top_n else pd.DataFrame())
    if not top_df.empty:
        st.dataframe(
            top_df[[
                "grid_id", "latitude", "longitude", "priority_score",
                "sampling_recommendation", "mean_ch4", "std_ch4",
                "obs_count", "outlier_count", "changepoint_day",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        idx = st.number_input("Select row (0-n)", 0, len(top_df) - 1, 0)
        gid = top_df.iloc[int(idx)]["grid_id"]
        sub = ts_df[ts_df.grid_id == gid]
        if not sub.empty:
            try:
                dates, vals = zip(*[x.split(",") for x in
                                    sub.iloc[0].time_series.split("|")])
                fig_ts = go.Figure(
                    go.Scatter(x=pd.to_datetime(dates),
                               y=list(map(float, vals)),
                               mode="lines+markers"))
                fig_ts.update_layout(title=f"Grid {gid} — Historical CH₄")
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.error(f"趋势数据绘图失败: {e}")
    else:
        st.info("No grids after filters.")

    # ---------- Blind-spot List ----------
    st.subheader("Blind-spot List")
    st.dataframe(blind_df if not blind_df.empty
                 else pd.DataFrame({"info": ["None"]}),
                 use_container_width=True, hide_index=True)

    # ---------- Report & README ----------
    with st.expander("📄 Auto-generated Report"):
        path = f"{DIR}/{FILES['report']}"
        try:
            st.code(open(path, encoding="utf-8").read(), language="markdown")
        except Exception:
            st.info("Report missing or load error.")

    with st.expander("ℹ️ Field Description"):
        path = f"{DIR}/{FILES['readme']}"
        try:
            st.code(open(path, encoding="utf-8").read(), language="markdown")
        except Exception:
            st.info("README missing or load error.")


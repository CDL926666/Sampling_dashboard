# Y:\Bishe_project\tab_spatial.py

from __future__ import annotations
import io, glob, zipfile, pathlib, os
import numpy as np
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from common import DIR, FILES, PRIO_LV, PRIO_COL

# 地图显示模板配置
pio.templates.default = "plotly_dark"
px.set_mapbox_access_token(
    "pk.eyJ1IjoicGxvdGx5dXNlciIsImEiOiJjaWZ4dmFhOG8wMDU2dW9vY2dyd2Z6N3RjIn0.2dUAQKqKAF-6EMbbVfwn9w"
)

def _check_required_data(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        st.warning("Main data missing for spatial analysis. Please upload and process your data first.")
        return False
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("Key fields (latitude/longitude) missing from main data.")
        return False
    return True

def _spatial_filters(df: pd.DataFrame, blind_df: pd.DataFrame):
    with st.sidebar:
        st.header("Spatial Filters")
        sel_lv = st.multiselect(
            "Priority Level", PRIO_LV, default=PRIO_LV, key="spatial_sel_lv"
        )
        show_blind = st.checkbox(
            "Show only blind-zones", value=False,
            key="spatial_show_blind", disabled=blind_df.empty
        )
        lat_min, lat_max = float(df.latitude.min()), float(df.latitude.max())
        lon_min, lon_max = float(df.longitude.min()), float(df.longitude.max())
        lat_rng = st.slider(
            "Latitude Range", lat_min, lat_max, (lat_min, lat_max), key="spatial_lat_rng"
        )
        lon_rng = st.slider(
            "Longitude Range", lon_min, lon_max, (lon_min, lon_max), key="spatial_lon_rng"
        )
        kw = st.text_input("Fuzzy search (visible cols)", key="spatial_kw")
        st.caption(f"Working dir: {pathlib.Path.cwd()}")
        with st.expander("Download result files (.zip)"):
            if st.button("Build ZIP", key="spatial_zip_btn"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in glob.glob(f"{DIR}/*"):
                        zf.write(f, arcname=os.path.basename(f))
                buf.seek(0)
                st.session_state["spatial_zip"] = buf.getvalue()
            if "spatial_zip" in st.session_state:
                st.download_button(
                    "Download ZIP", st.session_state["spatial_zip"],
                    file_name="ch4_outputs.zip"
                )
    return sel_lv, show_blind, lat_rng, lon_rng, kw

def _filter_view(df: pd.DataFrame, blind_df: pd.DataFrame, sel_lv, show_blind, lat_rng, lon_rng, kw):
    mask = df["sampling_recommendation"].isin(sel_lv)
    mask &= df.latitude.between(*lat_rng) & df.longitude.between(*lon_rng)
    if show_blind and "blind_spot_flag" in df:
        mask &= df.blind_spot_flag.eq("blind_zone")
    if kw:
        cols = ["grid_id", "sampling_recommendation", "blind_spot_flag"]
        mask &= df[cols].astype(str).apply(
            lambda s: s.str.contains(kw, case=False, regex=False)
        ).any(axis=1)
    return df[mask].copy()

def _draw_priority_map(view, blind_df):
    view["std_ch4"] = pd.to_numeric(view["std_ch4"], errors="coerce").fillna(1.0)
    size_cap = np.nanpercentile(view["std_ch4"], 99) or 1.0
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
                    marker=dict(color="black", size=8, symbol="circle"),
                )
            )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Map failed, fallback: {e}")
        fig_map = px.scatter_mapbox(
            lat=lat, lon=lon, color=color,
            category_orders={"color": PRIO_LV},
            color_discrete_map=PRIO_COL,
            size=size, size_max=15, zoom=4, height=560,
        )
        fig_map.update_layout(mapbox_style="open-street-map",
                              margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

def _draw_priority_stats(view, hist_df):
    counts = view["sampling_recommendation"].value_counts().reindex(PRIO_LV).fillna(0)
    labels = counts.index.astype(str).tolist()
    values = counts.values.astype(int).tolist()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Grids (filtered)", len(view))
        st.bar_chart(counts)
    with c2:
        if sum(values) == 0:
            st.info("No data for pie chart.")
        else:
            fig_pie = go.Figure(go.Pie(labels=labels, values=values))
            st.plotly_chart(fig_pie, use_container_width=True)
    if not hist_df.empty:
        st.bar_chart(
            hist_df.set_index(hist_df.priority_score_bins.astype(str))["count"],
            height=180, use_container_width=True,
        )

def _draw_top_n_trend(view, ts_df):
    st.subheader("Top-N Priority Grids & Trend")
    top_n = st.slider("Show Top-N", 10, 100, 20, key="spatial_top_n") if not view.empty else 0
    top_df = (view.sort_values("priority_score", ascending=False).head(top_n)
              if top_n else pd.DataFrame())
    if not top_df.empty:
        st.dataframe(
            top_df[["grid_id", "latitude", "longitude", "priority_score",
                    "sampling_recommendation", "mean_ch4", "std_ch4",
                    "obs_count", "outlier_count", "changepoint_day"]],
            use_container_width=True, hide_index=True,
        )
        idx = st.number_input("Select row (0-n)", 0, len(top_df)-1, 0, key="spatial_row_idx")
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
                fig_ts.update_layout(title=f"Grid {gid} — Historical CH4",
                                     yaxis_title="Flux")
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.error(f"Trend plot failed: {e}")
    else:
        st.info("No grids after filters.")

def render(df: pd.DataFrame, blind_df: pd.DataFrame, hist_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    if not _check_required_data(df):
        return

    sel_lv, show_blind, lat_rng, lon_rng, kw = _spatial_filters(df, blind_df)
    view = _filter_view(df, blind_df, sel_lv, show_blind, lat_rng, lon_rng, kw)

    st.title("Methane (CH₄) Scientific-Sampling Dashboard")

    st.subheader("Spatial Priority Map & Blind-spots")
    if view.empty:
        st.warning("No data for current filters. Please adjust filters or check your data.")
    else:
        _draw_priority_map(view, blind_df)

    st.subheader("Priority Statistics")
    if view.empty:
        st.info("No data to count.")
    else:
        _draw_priority_stats(view, hist_df)

    _draw_top_n_trend(view, ts_df)

    st.subheader("Blind-spot List")
    st.dataframe(
        blind_df if not blind_df.empty else pd.DataFrame({"info": ["None"]}),
        use_container_width=True, hide_index=True
    )

    with st.expander("Auto-generated Report"):
        path = DIR / FILES["report"]
        st.code(path.read_text("utf-8"), language="markdown") \
            if path.exists() else st.info("Report missing.")

    with st.expander("Field Description"):
        path = DIR / FILES["readme"]
        st.code(path.read_text("utf-8"), language="markdown") \
            if path.exists() else st.info("README missing.")

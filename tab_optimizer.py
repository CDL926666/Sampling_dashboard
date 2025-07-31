# Y:\Bishe_project\tab_optimizer.py

from __future__ import annotations

import json, pathlib, textwrap
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from jinja2 import Template

from PY3.step5_sampling_engine import run_sampling
from common import OUT_SAMPLING

BOOT_N       = 1000
MIN_MONTHS   = 12
DEFAULT_EPS  = 5
SEASONS      = ("DJF", "MAM", "JJA", "SON")

try:
    import weasyprint
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

_PDF_TMPL = Template(textwrap.dedent("""
<style>
  body{font-family:Arial,Helvetica,sans-serif;font-size:14px}
  h2{color:#036;text-align:center;margin-bottom:4px}
  ul{line-height:1.4}
  code{background:#f4f4f4;padding:2px 4px;border-radius:4px}
</style>
<h2>CH₄ Sampling Recommendation</h2>
<p><b>Grid:</b> <code>{{ gid }}</code></p>
<p>Target CI95 ≤ <b>{{ eps }}</b> %</p>
<p>Recommended sampling&nbsp;<b>p★ = {{ pstar }} %</b></p>
<p>Expected CI95&nbsp;: {{ ci }} %</p>
<p>Cost-saving&nbsp;: {{ save }} %</p>
<h3>Calendar</h3>
<ul>{% for d in cal %}<li>{{ d }}</li>{% endfor %}</ul>
"""))

@dataclass
class SamplingContext:
    gid: str
    eps: float
    w: dict[str, float]

    @property
    def json_path(self) -> pathlib.Path:
        return OUT_SAMPLING / f"p_star_{self.gid}.json"

    @property
    def csv_path(self) -> pathlib.Path:
        return OUT_SAMPLING / f"bootstrap_{self.gid}.csv"

    def _file_valid(self) -> bool:
        if not (self.csv_path.exists() and self.json_path.exists()):
            return False
        try:
            csv_ok = sum(1 for _ in self.csv_path.open()) >= 2
            meta   = json.loads(self.json_path.read_text(encoding="utf-8"))
            json_ok = {"p_star_%", "expected_CI95_%", "calendar"} <= meta.keys()
            return csv_ok and json_ok
        except Exception:
            return False

    def need_run(self, force: bool = False) -> bool:
        if force or not self._file_valid():
            return True
        try:
            meta = json.loads(self.json_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        if abs(meta.get("target_error_%", 0) - self.eps) > 1e-6:
            return True
        w_old = meta.get("season_weight",
                         dict(zip(SEASONS, (0.25, 0.25, 0.25, 0.25))))
        return any(abs(self.w[s] - w_old.get(s, 0)) > 1e-4 for s in SEASONS)

@st.cache_data(show_spinner=False)
def _load_csv(path: pathlib.Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def render(df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    st.header("Sampling Optimizer — 95 % CI for Monthly Means")

    if df.empty or ts_df.empty:
        st.error("Main data or time series is empty. Optimizer not available.")
        return

    def _count_months(ts: str) -> int:
        return ts.count("|") + 1 if isinstance(ts, str) and ts else 0
    ts_df["months"] = ts_df["time_series"].apply(_count_months)
    months_map = ts_df.set_index("grid_id")["months"].to_dict()

    with st.sidebar:
        st.subheader("Settings")
        selectable = [g for g, m in months_map.items() if m >= MIN_MONTHS]
        if not selectable:
            st.error(f"No grid has ≥{MIN_MONTHS} months of data.")
            return
        gid = st.selectbox("Grid ID", sorted(selectable))
        eps = st.slider("Target CI95 % error", 1, 20, DEFAULT_EPS)
        st.markdown("*Season weights*")
        w_vals = [st.number_input(s, 0.0, 1.0, 0.25, 0.05, key=f"w_{s}") for s in SEASONS]
        w_sum = sum(w_vals) or 1e-9
        season_w = dict(zip(SEASONS, [w / w_sum for w in w_vals]))
        st.caption(f"Sum = {w_sum:.3f}")
        cols = st.columns(2)
        force_run     = cols[0].button("Re-calculate")
        refresh_only  = cols[1].button("Refresh result")

    ctx = SamplingContext(gid, eps, season_w)

    if ctx.need_run(force_run):
        with st.spinner("Running bootstrap ..."):
            try:
                run_sampling(grid_id=gid, eps=eps, n_boot=BOOT_N, season_weight=season_w)
            except Exception as e:
                st.error(f"Sampling engine failed: {e}")
                st.stop()

    if refresh_only:
        st.cache_data.clear()

    boot_df = _load_csv(ctx.csv_path)
    info    = _load_json(ctx.json_path)
    if boot_df.empty or not info:
        st.error("Result file invalid or loading failed, please click Re-calculate to retry.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("p★ (%)",          f"{info['p_star_%']}")
    c2.metric("Expected CI95 %", f"{info['expected_CI95_%']:.2f}")
    c3.metric("Cost-saving %",   f"{info['cost_saving_%']:.1f}")

    tab_ci, tab_bias, tab_ts, tab_cal = st.tabs(
        ["CI-p Curve", "Bias Violin", "Time-series", "Calendar / Export"]
    )

    with tab_ci:
        fig = go.Figure(go.Scatter(
            x=boot_df["p"], y=boot_df["CI95_%"],
            mode="lines+markers"))
        fig.add_hline(y=eps, line_dash="dash", line_color="#d62728")
        fig.add_vline(x=info['p_star_%']/100,
                      line_dash="dot", line_color="#2ca02c")
        fig.update_layout(xaxis_title="p", yaxis_title="CI95 %",
                          title="CI95 % vs sampling ratio p")
        st.plotly_chart(fig, use_container_width=True)

    with tab_bias:
        keep = sorted({.10, .30, info['p_star_%']/100, 1.00})
        vdf  = boot_df[boot_df["p"].isin(keep)].copy()
        if vdf.empty:
            st.info("No data to plot.")
        else:
            vdf["p_label"] = (vdf["p"]*100).round(1).astype(str)+" %"
            fig = px.violin(vdf, x="p_label", y="bias_%",
                            box=True, points="all",
                            title="Bootstrap bias distribution",
                            labels={"bias_%": "Bias %"})
            st.plotly_chart(fig, use_container_width=True)

    with tab_ts:
        row = ts_df.loc[ts_df.grid_id == gid]
        if row.empty:
            st.info("Time-series not found.")
        else:
            pairs = [s.split(",") for s in row.iloc[0].time_series.split("|")]
            dates = pd.to_datetime([p[0] for p in pairs])
            vals  = [float(p[1]) for p in pairs]
            star  = set(info["calendar"])
            vals_star = [v if d.strftime("%Y-%m-%d") in star else None for v, d in zip(vals, dates)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=vals,
                                     name="Full", line=dict(color="#8c8c8c")))
            fig.add_trace(go.Scatter(x=dates, y=vals_star,
                                     name="p★", mode="markers+lines",
                                     marker=dict(color="#2ca02c", size=7)))
            fig.update_layout(title="Time-series reconstruction",
                              yaxis_title="CH₄ flux")
            st.plotly_chart(fig, use_container_width=True)

    with tab_cal:
        st.write(info["calendar"])
        if info["calendar"]:
            st.download_button(
                "Download CSV",
                "\n".join(info["calendar"]),
                file_name=f"calendar_{gid}_{int(info['p_star_%'])}pct.csv"
            )
        with st.expander("Export PDF report"):
            if not HAVE_PDF:
                st.info("WeasyPrint not installed.")
            elif st.button("Generate PDF"):
                html = _PDF_TMPL.render(
                    gid=gid, eps=eps, pstar=info['p_star_%'],
                    ci=round(info['expected_CI95_%'], 2),
                    save=info['cost_saving_%'], cal=info['calendar']
                )
                pdf_bytes = weasyprint.HTML(string=html).write_pdf()
                st.download_button("Download PDF", pdf_bytes,
                                   file_name=f"sampling_{gid}.pdf",
                                   mime="application/pdf")

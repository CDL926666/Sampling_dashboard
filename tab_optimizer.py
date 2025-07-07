#!/usr/bin/env python3
# ======================================================================
#  tab_optimizer.py  ·  “Sampling Optimizer” 业务子页面
# ======================================================================
from __future__ import annotations
import json, pathlib, textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from jinja2 import Template

from PY3.step5_sampling_engine import run_sampling
from common import OUT_SAMPLING

# ───── 可选依赖 (PDF) ─────
try:
    import weasyprint
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

_PDF_TMPL = Template(textwrap.dedent("""
    <style>
        h2{color:#036;text-align:center;margin-bottom:0}
        body{font-family:Arial,Helvetica,sans-serif;font-size:14px}
        ul{line-height:1.4}
        code{background:#f4f4f4;padding:2px 4px;border-radius:4px}
    </style>
    <h2>CH₄ Sampling Recommendation</h2>
    <p><b>Grid:</b> <code>{{ gid }}</code></p>
    <p>Target CI95 ≤ <b>{{ eps }}</b> %</p>
    <p>Recommended sampling&nbsp;<b>p★ = {{ pstar }}%</b></p>
    <p>Expected CI95&nbsp;: {{ ci }} %</p>
    <p>Cost-saving&nbsp;: {{ save }} %</p>
    <h3>Calendar</h3>
    <ul>{% for d in cal %}<li>{{ d }}</li>{% endfor %}</ul>
"""))

@dataclass
class SamplingContext:
    gid:   str
    eps:   float
    w:     dict[str, float]

    @property
    def json_path(self) -> pathlib.Path:
        return OUT_SAMPLING / f"p_star_{self.gid}.json"

    @property
    def csv_path(self) -> pathlib.Path:
        return OUT_SAMPLING / f"bootstrap_{self.gid}.csv"

    def need_run(self, force: bool = False) -> bool:
        if force or (not self.json_path.exists()) or (not self.csv_path.exists()):
            return True
        META = _load_json_cached(self.json_path)
        if not META or "target_error_%" not in META:  # 防止损坏
            return True
        if abs(META["target_error_%"] - self.eps) > 1e-6:
            return True
        w_old = META.get("season_weight") or {"DJF": .25, "MAM": .25, "JJA": .25, "SON": .25}
        return any(abs(self.w[k] - w_old.get(k, 0)) > 1e-4 for k in self.w)

@st.cache_data(show_spinner=False)
def _load_json_cached(p: pathlib.Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"⚠️ 读取 JSON 失败: {p}  {e}")
        return {}

@st.cache_data(show_spinner=False)
def _load_csv_cached(p: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(p)
        if df.empty:
            st.warning(f"⚠️ CSV 为空: {p}")
        return df
    except Exception as e:
        st.warning(f"⚠️ 读取 CSV 失败: {p}  {e}")
        return pd.DataFrame()

def render(df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    st.header("🧮 Sampling Optimizer  —  monthly mean · 95 % CI")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.subheader("Optimizer Settings")
        if df.empty:
            st.error("没有主数据，无法优化抽样。")
            return
        grid_ids = sorted(df.grid_id.unique())
        if not grid_ids:
            st.error("当前主数据没有 grid_id，请检查数据。")
            return
        gid = st.selectbox("Grid ID", grid_ids, key="sel_gid")
        eps = st.slider("Target CI95 % error", 1, 20, 5, key="sel_eps")
        st.markdown("*Season weight (DJF / MAM / JJA / SON)*")
        w_vals = [st.number_input(s, 0.0, 1.0, 0.25, 0.05, key=f"w_{s}")
                  for s in ("DJF", "MAM", "JJA", "SON")]
        w_sum  = sum(w_vals) or 1e-9
        season_w = dict(zip(("DJF", "MAM", "JJA", "SON"),
                            [w / w_sum for w in w_vals]))
        st.caption(f"∑ weights = {w_sum:.3f}")
        force_run = st.button("↻  Re-calculate")

    ctx = SamplingContext(gid, eps, season_w)

    # ---------- Run engine (如需) ----------
    if ctx.need_run(force_run):
        with st.spinner("Bootstrap running … (may take 5-20 s)"):
            try:
                run_sampling(ctx.gid, ctx.eps, 1000, ctx.w)
            except Exception as e:
                st.error(f"Sampling engine failed: {e}")
                st.stop()

    # ---------- 读取结果 ----------
    boot_df = _load_csv_cached(ctx.csv_path)
    info    = _load_json_cached(ctx.json_path)
    if boot_df.empty or not info:
        st.error("❌ 该 grid 无可用结果数据，请尝试其它 grid_id 或重新运行采样。")
        return

    # ---------- Debug ----------
    st.caption(f"【DEBUG】当前 grid: {gid}, 结果数据行数: {len(boot_df)}")
    st.caption(f"【DEBUG】info 内容: {info}")

    # ---------- CI95 – p 曲线 ----------
    try:
        fig_ci = px.line(
            boot_df, x="p", y="CI95_%", markers=True,
            title="CI95 %  vs  sampling ratio (p)",
        )
        fig_ci.add_hline(y=eps, line_dash="dash", line_color="#d62728")
        fig_ci.add_vline(x=info["p_star_%"] / 100, line_dash="dot",
                         line_color="#2ca02c")
        fig_ci.update_traces(hovertemplate="p = %{x:.2f}<br>CI95 = %{y:.2f} %")
        st.plotly_chart(fig_ci, use_container_width=True)
    except Exception as e:
        st.warning(f"曲线图绘制失败: {e}")

    # ---------- Bias Violin ----------
    try:
        p_show = sorted(set([.10, .30, info["p_star_%"] / 100, 1.00]))
        viol_df = boot_df[boot_df["p"].isin(p_show)].copy()
        viol_df["p_label"] = (viol_df["p"] * 100).round(1).astype(str) + " %"
        if not viol_df.empty:
            fig_violin = px.violin(
                viol_df, x="p_label", y="bias_%", box=True, points="all",
                title="Bootstrap bias distribution",
                labels={"bias_%": "Bias %"},
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    except Exception as e:
        st.warning(f"Violin 图绘制失败: {e}")

    # ---------- Time-series Reconstruction ----------
    sub = ts_df[ts_df.grid_id == gid]
    if sub.empty:
        st.warning("⚠️ 原始时序缺失，无法重建序列。")
    else:
        try:
            dates, vals_full = zip(*[x.split(",") for x in
                                     sub.iloc[0].time_series.split("|")])
            dates = pd.to_datetime(dates)
            vals_full = list(map(float, vals_full))

            star_set = set(info.get("calendar", []))
            vals_star = [v if d.strftime("%Y-%m-%d") in star_set else None
                         for d, v in zip(dates, vals_full)]

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=dates, y=vals_full,
                                        name="Full series", line=dict(color="#8c8c8c")))
            fig_ts.add_trace(go.Scatter(x=dates, y=vals_star,
                                        name=f"p★ ({info.get('p_star_%', '?')} %)",
                                        mode="markers+lines",
                                        marker=dict(color="#2ca02c", size=7)))
            fig_ts.update_layout(title="Time-series reconstruction (full vs p★)",
                                 yaxis_title="CH₄ flux")
            st.plotly_chart(fig_ts, use_container_width=True)
        except Exception as e:
            st.warning(f"时序图绘制失败: {e}")

    # ---------- Summary & Calendar ----------
    if "p_star_%" in info and "expected_CI95_%" in info and "cost_saving_%" in info:
        st.success(
            f"**p★ = {info['p_star_%']} %**  "
            f"(CI95 ≈ {info['expected_CI95_%']:.2f} %, "
            f"Cost-saving ≈ {info['cost_saving_%']} %)"
        )

    with st.expander("📅  Sampling calendar"):
        st.write(info.get("calendar", []))
        if "calendar" in info:
            st.download_button(
                "Download CSV",
                "\n".join(info["calendar"]),
                file_name=f"calendar_{gid}_{int(info.get('p_star_%',0))}pct.csv",
            )

    # ---------- PDF Export ----------
    with st.expander("📝  Export PDF report"):
        if not HAVE_PDF:
            st.info("WeasyPrint 未安装，无法生成 PDF")
        else:
            if st.button("Generate PDF"):
                html = _PDF_TMPL.render(
                    gid=gid, eps=eps, pstar=info.get('p_star_%', '?'),
                    ci=round(info.get('expected_CI95_%', 0), 2),
                    save=info.get('cost_saving_%', 0), cal=info.get("calendar", []),
                )
                pdf_bytes = weasyprint.HTML(string=html).write_pdf()
                st.download_button(
                    "⬇️  Download PDF",
                    data=pdf_bytes,
                    file_name=f"sampling_{gid}.pdf",
                    mime="application/pdf",
                )

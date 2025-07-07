#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# Step-5 : Sampling Engine  (monthly bootstrap, 95 % CI)
# 生成：
#   sampling_engine/bootstrap_<grid>.csv
#   sampling_engine/p_star_<grid>.json
#
# CLI 示例（单行即可，Windows / Linux / macOS 均通用）：
#   python PY3/step5_sampling_engine.py --grid_id "49.500_-0.500" --eps 5 \
#          --boot 1000 --w "0.25,0.25,0.25,0.25"
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, pathlib, random, sys
import numpy as np
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# ────────── 路径与全局 ──────────
DATA_DIR = pathlib.Path("ch4_sampling_result")
OUT_DIR  = pathlib.Path("sampling_engine")
OUT_DIR.mkdir(exist_ok=True)

TS_DF: pd.DataFrame | None = None

def load_ts_df() -> pd.DataFrame:
    global TS_DF
    if TS_DF is None:
        path = DATA_DIR / "ts_df.csv"
        if not path.exists():
            raise FileNotFoundError(f"数据文件缺失: {path}")
        TS_DF = pd.read_csv(path, low_memory=False)
    return TS_DF

# ────────── Season 映射（北半球）──────────
_SEASON_MAP = {
    12: "DJF",  1: "DJF",  2: "DJF",
     3: "MAM",  4: "MAM",  5: "MAM",
     6: "JJA",  7: "JJA",  8: "JJA",
     9: "SON", 10: "SON", 11: "SON",
}
def _season(date_str: str) -> str:
    """'YYYY-MM-DD' → 'DJF/MAM/JJA/SON'"""
    return _SEASON_MAP[int(date_str.split("-")[1])]

# ────────── 辅助函数 ──────────
def _parse_weight(arg: str) -> dict[str, float]:
    """'0.2,0.3,0.3,0.2' → {'DJF':0.2, …}（自动归一）"""
    w = list(map(float, arg.split(",")))
    if len(w) != 4 or sum(w) <= 0:
        raise ValueError("--w 应给出 4 个权重，用逗号分隔")
    s = sum(w)
    return dict(zip(("DJF", "MAM", "JJA", "SON"), [x / s for x in w]))

def _parse_series(s: str) -> tuple[np.ndarray, list[str]]:
    """把 ts_df.time_series 解析为 (values, dates)"""
    pairs  = [p.split(",") for p in s.split("|") if p]
    values = np.fromiter((float(v) for _, v in pairs), float)
    dates  = [d for d, _ in pairs]
    return values, dates

# ────────────────────────────────────────────────────────────────
def run_sampling(grid_id: str,
                 eps: float = 5.0,
                 n_boot: int = 1_000,
                 season_weight: dict[str, float] | None = None,
                 rng_seed: int = 42) -> dict:
    """
    Bootstrap → 计算 95 % CI → 选最小采样比例 p★ → 生成采样日历
    写出 bootstrap_<grid>.csv 与 p_star_<grid>.json，并返回 metadata dict。
    """
    ts_df = load_ts_df()
    row = ts_df.loc[ts_df.grid_id == grid_id]
    if row.empty:
        raise ValueError(f"{grid_id=} 不存在于 ts_df.csv")

    y, dates = _parse_series(row.iloc[0]["time_series"])
    
    # 将 y 转换为 np.array，确保不带 Mask，避免警告
    y = np.asarray(y)
    
    n_month = len(y)
    mu = np.nanmean(y)  # 用 nanmean 防止潜在 NaN
    
    if n_month < 12:
        raise ValueError("序列不足 12 个月，无法评估年均误差")

    rng = np.random.default_rng(rng_seed)
    p_grid = np.arange(0.05, 1.01, 0.05)               # 5 % … 100 %
    records: list[dict[str, float]] = []

    for p in p_grid:
        k = max(1, round(p * n_month))
        means = np.array([
            y[rng.choice(n_month, k, replace=False)].mean()
            for _ in range(n_boot)
        ])
        ci95 = 1.96 * means.std()
        records.append({
            "p": p,
            "CI95_%": ci95 / mu * 100,
            "bias_%": (means.mean() - mu) / mu * 100,
        })

    boot_df = pd.DataFrame(records).sort_values("p")
    boot_df.to_csv(OUT_DIR / f"bootstrap_{grid_id}.csv", index=False)

    # —— 选 p★ —— #
    meet  = boot_df.loc[boot_df["CI95_%"] <= eps, "p"]
    p_star = float(meet.min()) if not meet.empty else 1.0
    exp_ci = float(boot_df.iloc[(boot_df["p"] - p_star).abs().idxmin()]["CI95_%"])

    # —— 生成采样日历 —— #
    k_need = max(1, round(p_star * n_month))
    if season_weight is None:                       # 等间隔
        sel_idx  = np.linspace(0, n_month - 1, k_need, dtype=int)
        calendar = [dates[i] for i in sel_idx]
    else:                                           # 分季节
        buckets: dict[str, list[str]] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        for d in dates:
            buckets[_season(d)].append(d)

        calendar: list[str] = []
        for s, w in season_weight.items():
            need = max(1, round(w * k_need)) if w > 0 else 0
            if need == 0 or not buckets[s]:
                continue
            pool = buckets[s]
            pick = (random.choices(pool, k=need)
                    if len(pool) < need else random.sample(pool, need))
            calendar.extend(pick)
        calendar.sort()

    info = {
        "grid_id": grid_id,
        "target_error_%": eps,
        "p_star_%": round(p_star * 100, 1),
        "expected_CI95_%": round(exp_ci, 2),
        "calendar": calendar,
        "cost_saving_%": round((1 - p_star) * 100, 1),
        "season_weight": season_weight or "equal",
        "bootstrap_n": n_boot,
    }
    with open(OUT_DIR / f"p_star_{grid_id}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info

# ─────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run sampling optimisation for one grid")
    ap.add_argument("--grid_id", required=True, help="grid_id in ts_df.csv")
    ap.add_argument("--eps",  type=float, default=5.0,  help="target CI95 %% error")
    ap.add_argument("--boot", type=int,   default=1000, help="bootstrap iterations")
    ap.add_argument("--w",    default="0.25,0.25,0.25,0.25",
                    help="season weights DJF,MAM,JJA,SON (sum needn't be 1; auto-norm)")
    args = ap.parse_args()

    try:
        season_w = _parse_weight(args.w) if args.w else None
        meta = run_sampling(args.grid_id, args.eps, args.boot, season_w)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"错误：{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"参数错误：{e}")
        sys.exit(2)

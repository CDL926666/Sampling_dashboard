# Y:\Bishe_project\PY3\step5_sampling_engine.py

from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

DEFAULT_DATA_DIR = pathlib.Path("ch4_sampling_result")
DEFAULT_OUT_DIR = pathlib.Path("sampling_engine")
DEFAULT_OUT_DIR.mkdir(exist_ok=True)

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_ERROR_TARGET = 5.0
DEFAULT_SEASON_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
DEFAULT_RNG_SEED = 42

TS_DF: pd.DataFrame | None = None

# 加载时序数据
def load_ts_df(data_dir=DEFAULT_DATA_DIR) -> pd.DataFrame:
    global TS_DF
    if TS_DF is None:
        path = data_dir / "ts_df.csv"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        TS_DF = pd.read_csv(path, low_memory=False)
    return TS_DF

# 月份到季节映射
_SEASON_MAP = {
    12: "DJF",  1: "DJF",  2: "DJF",
     3: "MAM",  4: "MAM",  5: "MAM",
     6: "JJA",  7: "JJA",  8: "JJA",
     9: "SON", 10: "SON", 11: "SON",
}
def _season(date_str: str) -> str:
    return _SEASON_MAP[int(date_str.split("-")[1])]

def _parse_weight(arg: str) -> dict[str, float]:
    w = list(map(float, arg.split(",")))
    if len(w) != 4 or sum(w) <= 0:
        raise ValueError("Weight string must have 4 positive numbers")
    s = sum(w)
    return dict(zip(("DJF", "MAM", "JJA", "SON"), [x / s for x in w]))

def _parse_series(s: str) -> tuple[np.ndarray, list[str]]:
    pairs = [p.split(",") for p in s.split("|") if p]
    values = np.fromiter((float(v) for _, v in pairs), float)
    dates = [d for d, _ in pairs]
    return values, dates

# 采样日历主函数
def run_sampling(
    grid_id: str,
    data_dir: pathlib.Path = DEFAULT_DATA_DIR,
    out_dir: pathlib.Path = DEFAULT_OUT_DIR,
    eps: float = DEFAULT_ERROR_TARGET,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    season_weight: dict[str, float] | None = None,
    rng_seed: int = DEFAULT_RNG_SEED
) -> dict:
    ts_df = load_ts_df(data_dir)
    row = ts_df.loc[ts_df.grid_id == grid_id]
    if row.empty:
        raise ValueError(f"{grid_id} not found in ts_df.csv")
    y, dates = _parse_series(row.iloc[0]["time_series"])
    y = np.asarray(y)
    n_month = len(y)
    mu = np.nanmean(y)
    if n_month < 12:
        raise ValueError("Too few months for error estimation (require >=12)")

    rng = np.random.default_rng(rng_seed)
    p_grid = np.arange(0.05, 1.01, 0.05)
    records = []
    for p in p_grid:
        k = max(1, round(p * n_month))
        means = np.array([
            y[rng.choice(n_month, k, replace=False)].mean()
            for _ in range(n_boot)
        ])
        ci95 = 1.96 * means.std()
        records.append({
            "p": p,
            "CI95_%": ci95 / mu * 100 if mu != 0 else np.nan,
            "bias_%": (means.mean() - mu) / mu * 100 if mu != 0 else np.nan,
        })

    boot_df = pd.DataFrame(records).sort_values("p")
    boot_df.to_csv(out_dir / f"bootstrap_{grid_id}.csv", index=False)

    meet = boot_df.loc[boot_df["CI95_%"] <= eps, "p"]
    p_star = float(meet.min()) if not meet.empty else 1.0
    exp_ci = float(boot_df.iloc[(boot_df["p"] - p_star).abs().idxmin()]["CI95_%"])

    k_need = max(1, round(p_star * n_month))
    if season_weight is None:
        sel_idx = np.linspace(0, n_month - 1, k_need, dtype=int)
        calendar = [dates[i] for i in sel_idx]
    else:
        buckets = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        for d in dates:
            buckets[_season(d)].append(d)
        calendar = []
        for s, w in season_weight.items():
            need = max(1, round(w * k_need)) if w > 0 else 0
            if need == 0 or not buckets[s]:
                continue
            pool = buckets[s]
            pick = (random.choices(pool, k=need) if len(pool) < need else random.sample(pool, need))
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
    with open(out_dir / f"p_star_{grid_id}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run sampling optimisation for one grid")
    ap.add_argument("--grid_id", required=True, help="grid_id in ts_df.csv")
    ap.add_argument("--eps", type=float, default=DEFAULT_ERROR_TARGET, help="target CI95 % error")
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOTSTRAP_N, help="bootstrap iterations")
    ap.add_argument("--w", default="0.25,0.25,0.25,0.25", help="season weights DJF,MAM,JJA,SON (sum auto-normed)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="input directory for ts_df.csv")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="output directory")
    args = ap.parse_args()
    try:
        season_w = _parse_weight(args.w) if args.w else None
        meta = run_sampling(
            args.grid_id,
            data_dir=pathlib.Path(args.data_dir),
            out_dir=pathlib.Path(args.out_dir),
            eps=args.eps,
            n_boot=args.boot,
            season_weight=season_w
        )
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Argument Error: {e}")
        sys.exit(2)

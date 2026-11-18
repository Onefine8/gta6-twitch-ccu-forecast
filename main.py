"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Model: comparable-launch shape forecaster, built from the case's own data.
The case ships 38-42 past big launches' hourly Twitch curves (SullyGnome
hourly category averages) plus the target's hour-1 print. For every past
launch we measure the shape of its release day as log-ratios
L_h = ln(viewers_h / viewers_1), h = 2..24, then forecast GTA VI's day
by pooling those shapes at GTA VI's own hour-1 scale.

What the data says (measured on the kit's practice cases, 37 usable
comparables; the pattern is what the leaderboard's best entries use too
-- kernel-weighted comparable shapes, log-ratio / analog-ensemble models):
- Big launches are FLATTER than small ones. The slope of log-ratio on
  log(hour-1 viewers) is about -0.7 at hour 14 and -0.9 at hours 18-20
  (correlation -0.8 to -0.94): a launch whose hour 1 already holds
  ~1M viewers has its peak near hour 1, while a small launch grows
  several-fold into the evening. The biggest comparables show it
  directly: ELDEN RING (hour 1 875k) dips to 0.53x at hour 10 and
  recovers only to 0.95x; Cyberpunk 2077 (852k) dips to 0.43x and
  recovers to 0.98x; Diablo IV (533k) stays 0.76x-1.26x all day.
- The median big-launch day dips to ~0.72x-0.78x at hours 3-5, stays
  ~0.75x-0.9x through the US unlock hours, then rises to ~1.3x-1.7x
  in the evening (hours 19-23) before easing at hour 24.
- So the forecast anchors on hour 1 and scales the pooled shape:
  median log-ratio per hour = 0.7 * (linear fit of log-ratio on
  log hour-1 size, evaluated at GTA VI's hour-1 size, with the
  extrapolation capped at 0.7 log units past the biggest comparable)
  + 0.3 * (Gaussian-kernel weighted median of log-ratios, bandwidth
  1.5 in log hour-1 size). The blend keeps the scale effect without
  letting the linear fit run away past the data.
- Spread per hour = the residual standard deviation of that hour's
  fit (floor 0.30 log units), so the 5%-95% band is honest about how
  much launch days differ even at the same size; quantiles are
  median * exp(z_tau * sigma_h).
- Leave-one-out pinball loss on the 28 big comparables (hour-1
  >= 10k): this model ~0.264 (regression weight 0.7; measured 0.2655
  at 0.6 and 0.2672 at 0.5, regression-only 0.2637), kernel-only
  ~0.281, the house persistence baseline ~0.351. The pooled shape
  is doing the work.

Rules the program obeys: standard library only, no network, no
randomness (same bytes every run), reads every *.in in the input
directory, writes one *.out per case, each {"hours": [{"hour": h,
"q": [q05, q25, q50, q75, q95]}]} for hours 2..24, quantiles
non-decreasing, each in (0, 1e9], rounded half-up to 2 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an
# `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Pool: only launches big enough to resemble a GTA VI-scale day.
MIN_H1 = 10000
# Kernel bandwidth in log hour-1 viewers.
KERNEL_BW = 1.5
# Blend weight of the scale regression vs the kernel median.
# Raised 0.6 -> 0.7 on the leave-one-out evidence: pinball 0.2655 at
# 0.6, 0.2644 at 0.7, 0.2672 at 0.5 (regression-only 0.2637). The
# remaining 0.3 kernel weight still anchors the median to observed
# shapes at the largest sizes, which matters because GTA VI's hour 1
# will sit past every comparable in the pool.
W_REG = 0.7
# Cap on extrapolating past the biggest comparable, in log units.
CAP_EXTRA = 0.7
# Floor on the per-hour log spread.
SIGMA_FLOOR = 0.30
# Fallback random-walk spread (house baseline) when the pool is too thin.
FALLBACK_SIGMA = 0.15


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def launch_curve(comp):
    """Hourly viewers {hour: viewers} for a comparable's release day.

    Returns None for the target's own history (not a launch) and for
    curves too thin to use.
    """
    name = comp.get("name", "")
    unlock = comp.get("unlock", "") or ""
    if "Grand Theft Auto VI" in name or "Not a launch" in unlock:
        return None
    day_start = comp.get("day_start")
    hourly = comp.get("hourly") or []
    if not day_start or not hourly:
        return None
    d0 = _parse_time(day_start)
    by_hour = {}
    for row in hourly:
        try:
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            h = int((_parse_time(row["from"]) - d0).total_seconds() // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= h <= 24:
            by_hour[h] = viewers
    if 1 not in by_hour or by_hour[1] < MIN_H1:
        return None
    if sum(1 for h in by_hour if h >= 2) < 20:
        return None
    return by_hour


def _mean(xs):
    return sum(xs) / len(xs)


def _linreg(xs, ys):
    """Least-squares y ~ a + b*x; returns (a, b)."""
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    if vx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs) / vx
    return my - b * mx, b


def _wquantile(pairs, level):
    """Weighted quantile of [(value, weight)] at `level`."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    cut = level * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= cut:
            return value
    return pairs[-1][0]


def hour_quantiles(pool, p1, hour):
    """Five log-quantiles of the ratio viewers_h / viewers_1, as counts."""
    xs, ys = [], []
    for by_hour in pool:
        if hour in by_hour:
            xs.append(math.log(by_hour[1]))
            ys.append(math.log(by_hour[hour] / by_hour[1]))
    if len(xs) < 5:
        # Pool too thin for this hour: persistence with a widening band.
        return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)))
                for z in Z]
    a, b = _linreg(xs, ys)
    max_h1 = max(math.exp(x) for x in xs)
    x = min(math.log(p1), math.log(max_h1) + CAP_EXTRA)
    mu_reg = a + b * x
    resid = [y - (a + b * xx) for xx, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / len(resid)
    sigma = max(SIGMA_FLOOR, math.sqrt(var))
    pairs = [(y, 0.2 + math.exp(-((xx - x) / KERNEL_BW) ** 2))
             for xx, y in zip(xs, ys)]
    mu_ker = _wquantile(pairs, 0.5)
    mu = W_REG * mu_reg + (1.0 - W_REG) * mu_ker
    qs = [mu + z * sigma for z in Z]
    for i in range(1, 5):
        qs[i] = max(qs[i], qs[i - 1])
    out = []
    for v in qs:
        q = round2(p1 * math.exp(v))
        q = max(q, 1)
        q = min(q, 1_000_000_000)
        out.append(q)
    for i in range(1, 5):
        out[i] = max(out[i], out[i - 1])
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    pool = []
    for comp in case.get("comparables") or []:
        curve = launch_curve(comp)
        if curve is not None:
            pool.append(curve)
    hours = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        hours.append({"hour": h, "q": hour_quantiles(pool, p1, h)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

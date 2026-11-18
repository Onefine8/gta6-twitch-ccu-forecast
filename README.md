# GTA VI launch-day Twitch CCU forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Onefine8/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Onefine8/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun `gta6-twitch-ccu-forecast-01`: a program (`main.py`,
Python 3.13, standard library only) that reads one case
(`/input/<id>.in`) and forecasts Twitch concurrent viewers in the
Grand Theft Auto VI category for hours 2–24 of release day
2026-11-19 UTC, as five quantiles (5/25/50/75/95%) per hour.

## Model

Comparable-launch shape forecaster, built entirely from the case's
own data (the 38–42 past big launches' hourly Twitch curves it ships):

- Anchor on the case's hour-1 print.
- For each past launch, measure its release-day shape as log-ratios
  `ln(viewers_h / viewers_1)`.
- Big launches are flatter than small ones: the slope of log-ratio on
  log hour-1 size is about −0.7 at hour 14 and −0.9 at hours 18–20
  (correlation −0.8 to −0.94 across the kit's comparables). The
  biggest launches show it directly — ELDEN RING (hour 1: 875k)
  dips to 0.53× at hour 10 and recovers only to 0.95×; Cyberpunk
  2077 (852k) dips to 0.43× and recovers to 0.98×.
- Median log-ratio per hour = 0.7 × (linear fit on log hour-1 size,
  evaluated at this case's hour-1 size, extrapolation capped 0.7 log
  units past the biggest comparable) + 0.3 × (Gaussian-kernel
  weighted median, bandwidth 1.5 in log size). The 0.7 weight is the
  leave-one-out optimum among the blends tried (0.2644, vs 0.2655 at
  0.6 and 0.2672 at 0.5); the remaining kernel weight anchors the
  median to observed shapes at the largest sizes.
- Spread per hour = residual standard deviation of that hour's fit
  (floor 0.30 log units); quantiles are median × exp(z × sigma).

Leave-one-out pinball loss on the kit's 28 big comparables
(hour-1 ≥ 10k): ~0.264, vs ~0.281 for the kernel-only variant and
~0.351 for the house persistence baseline.

## Files

- `main.py` — the entrypoint; reads every `*.in` in the input
  directory, writes one `*.out` per case.
- `LICENSE` — MIT.

## Run it

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=qgk71vks&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

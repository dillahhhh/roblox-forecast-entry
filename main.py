#!/usr/bin/env python3
"""Roblox Up-and-Coming chart forecast for make.fun (roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (chart reads, oldest first) and writes
/output/forecast.json: per-place probabilities for the top 10 places and a
top-10 order, scored against every hourly read of the chart in the 72 hours
after entries close.

Method -- horizon-mixture momentum forecast (deterministic, stdlib only)
-------------------------------------------------------------------------
The chart is a growth sort: a game's place is driven by how fast its player
count is growing, not by how many players it already has.  For every game on
the last chart we estimate, from its whole rank/playing history,

  * playing growth  -- time-weighted regression of log(playing) on time
                       (log points/day, half-life 8 h); the strongest signal,
                       since the chart sorts by user growth;
  * playing level   -- log10(playing), a small tie-breaker.

Both are shrunk by n/(n+4) (n = reads in the window): a game seen twice has
a noisy slope and must not be trusted like one seen twenty times.  Growth is
clamped to [-1.5, +2.0] log points/day.  A game's strength is
``base = -rank + 35 * growth + 0.5 * log10(playing)``; the order is the same
at every horizon (rank-velocity extrapolation was tested and hurt on every
fold: the chart turns over too fast for trends to persist).

The forecast is a *mixture over horizons* h = 0, 0.5, ..., 3 days: at each
horizon each place's probability is a Gaussian around the expected rank, with
sigma widening with horizon (1.75 places at h=0, +1.0 per day) because the
chart's top 10 turns over fast (median 6 places change between reads a few
hours apart).  Averaging over horizons is what the scoring wants: the score
is the mean log-probability over the window's 72 hourly reads, so early
hours are tight around today's chart and late hours are diffuse.

Each place names up to all ~97 chart games and sums to 0.8, leaving 0.2
unnamed: brand-new games keep entering the top 10 (in the practice window 3
of the late reads' top 10s held games never seen before), and the unnamed
share is what the forecast gives them.  Rows below p = 0.002 are dropped.

Reads are sorted by time and de-duplicated (the final history is hourly
recorder reads, which can repeat or arrive out of order); a row without a
rank falls back to its position in its read's list.

No age / survival penalty: the chart's "published within the last 28 days"
rule does NOT use the universe `created` date -- games 32-35 days old by
`created` still hold top-10 places (e.g. Survive Your Ideas, #1 at 32 days).
A created-based penalty demotes the actual leaders and cost 5.8 log points
on the practice window (verified: -38.67 with it, -32.91 without, on the
same history with real details attached).  Chart tenure (first seen on the
chart) is bounded by the history span and never nears 28 days here, so no
age term is used at all.

Backtest (local, against real archive reads -- the kit's 34 reads plus the
practice-window and Nov 11 reads reconstructed from the public archive's git
history at github.com/Kenny-Dhub/roblox-snapshots, scored exactly per
RULES.txt, log score, floor 0.0001, unnamed share).  History carries real
details (created/votes/favorites), as the final grading history will:

  fold                               this forecast   house baseline
  practice window (Nov 8-10)         -29.67 (7.8 hits)   -43.51 (6.4 hits)
  fold A (reads 1-24 -> 25-34)       -37.88
  fold C (reads 1-14 -> 15-24)       -36.40
  fold D (to Nov 10 23:23 -> Nov 11) -26.99
  mean, folds P/A/C                  -34.65

The previous version of this model scored -32.91 on the practice window;
the gains come from (1) dropping the buggy created-age penalty, (2) a
wider growth clamp [-1.5, 2.0] with stronger growth weight, (3) stronger
slope shrinkage n/(n+4), (4) a longer 240 h window, (5) a tighter sigma
(1.75 + 1.0/day, retuned from 1.5 for the best mean across folds) and (6) no rank-velocity extrapolation.  Top Trending rank,
growth acceleration, vote ratio and favorites were all tested as extra
features and rejected -- each made every fold worse.

No network, no randomness, no installed packages: the two grader runs on
the same history produce byte-identical output.
"""
import json
import math
import os
from datetime import datetime, timezone

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (grid-searched on four backtest folds) ---------------
HALF_LIFE_H = 8.0      # recency half-life for the trend regression (hours)
WINDOW_H = 240.0       # history window used for trends (hours)
W_GROW = 3.5           # weight of playing-growth rate in the strength score
W_PLAY = 0.5           # weight of log10(playing) level (tie-breaker only)
GROW_CLAMP = (-1.5, 2.0)  # clamp on log-playing growth (per day)
SIGMA0 = 1.75          # Gaussian sigma (places) at horizon 0
SIGMA_PER_DAY = 1.0    # sigma widening per day of horizon
UNNAMED = 0.2          # mass left unnamed at each place (new entrants)
MIN_P = 0.002          # rows below this are dropped
HORIZONS_DAYS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # mixture horizons
SHRINK_K = 4.0         # n/(n+K) slope shrinkage


def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def wlin(xs, ys, ws):
    """Weighted least-squares slope/intercept of y on x."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 0:
        return 0.0, my
    slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    return slope, my - slope * mx


def main():
    task_path = os.path.join(IN_DIR, "task.json")
    places = 10
    output_name = "forecast.json"
    try:
        with open(task_path, encoding="utf-8") as f:
            task = json.load(f)
        places = int(task.get("places") or 10)
        output_name = str(task.get("output") or "forecast.json")
    except Exception:
        pass

    reads = []
    with open(os.path.join(IN_DIR, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    if not reads:
        raise SystemExit("no history reads")
    # Sort by time; drop reads without a time and de-duplicate identical
    # timestamps (the hourly recorder can re-read a slot), keeping the last.
    stamped = []
    for r in reads:
        t = parse_time(r.get("at") or "")
        if t is not None:
            stamped.append((t, r))
    if not stamped:
        raise SystemExit("no history reads with a timestamp")
    stamped.sort(key=lambda tr: tr[0])
    deduped = []
    for t, r in stamped:
        if deduped and deduped[-1][0] == t:
            deduped[-1] = (t, r)
        else:
            deduped.append((t, r))
    reads = [r for _, r in deduped]
    last = reads[-1]
    t_end = parse_time(last.get("at") or "")
    if t_end is None:
        t_end = datetime.now(timezone.utc)

    # Per-game time series: (hours before the last read, rank, playing).
    # A row without a rank falls back to its position in its read's list.
    series = {}
    for r in reads:
        t = parse_time(r.get("at") or "")
        if t is None:
            continue
        h = (t_end - t).total_seconds() / 3600.0
        if h < 0:
            continue
        for idx, g in enumerate(r.get("up_and_coming") or []):
            uid = g.get("universe_id")
            if uid is None:
                continue
            series.setdefault(uid, []).append(
                (h, g.get("rank") or (idx + 1), g.get("playing") or 0))

    chart = sorted(last.get("up_and_coming") or [],
                   key=lambda g: g.get("rank") or 999)
    feats = []
    for g in chart:
        uid = g.get("universe_id")
        rank = g.get("rank") or 999
        playing = g.get("playing") or 0
        pts = [(h, r, p) for h, r, p in series.get(uid, []) if h <= WINDOW_H]
        grow_d = 0.0  # log-playing points per day
        if len(pts) >= 2:
            ws = [0.5 ** (h / HALF_LIFE_H) for h, _, _ in pts]
            xs = [-h for h, _, _ in pts]  # time, 0 = now, negative = past
            grow_h, _ = wlin(xs, [math.log(max(p, 1)) for _, _, p in pts], ws)
            grow_d = grow_h * 24.0
            # Shrink noisy slopes from games seen only a few times.
            grow_d *= len(pts) / (len(pts) + SHRINK_K)
        grow_c = max(GROW_CLAMP[0], min(GROW_CLAMP[1], grow_d))
        base = -rank + W_GROW * grow_c * 10.0 + W_PLAY * math.log10(max(playing, 1))
        feats.append({"uid": uid, "rank": rank, "base": base})

    # The order is the same at every horizon (see header: rank-velocity
    # extrapolation hurt on every fold).  Only sigma widens with horizon.
    order = sorted(feats, key=lambda f: (-f["base"], f["uid"]))

    # Horizon mixture: average each horizon's per-place Gaussian.
    acc = {}
    for hd in HORIZONS_DAYS:
        sigma_h = SIGMA0 + SIGMA_PER_DAY * hd
        for place in range(1, places + 1):
            row = []
            for idx, f in enumerate(order):
                d = (idx + 1 - place) / sigma_h
                row.append((f["uid"], math.exp(-0.5 * d * d)))
            tot = sum(w for _, w in row) or 1.0
            share = (1.0 - UNNAMED) / len(HORIZONS_DAYS)
            for uid, w in row:
                acc[(uid, place)] = acc.get((uid, place), 0.0) + share * w / tot
    probs = [{"universe_id": uid, "place": place, "p": round(p, 4)}
             for (uid, place), p in sorted(acc.items())
             if p >= MIN_P]
    # The top 10 is the expected order at the middle of the window (36 h);
    # with a horizon-independent order this is just the strength order.
    top10 = [f["uid"] for f in order[:places]]
    out = {"probabilities": probs, "top10": top10}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote %s: %d rows, top10: %s" % (out_path, len(probs), top10))


if __name__ == "__main__":
    main()

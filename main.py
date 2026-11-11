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

  * rank velocity   -- time-weighted regression of rank on time (ranks/day;
                       negative = climbing), half-life 8 h, window 72 h;
  * playing growth  -- time-weighted regression of log(playing) on time
                       (log points/day); the chart sorts by user growth, so
                       this is the single strongest signal;
  * age / survival  -- games leave the chart 28 days after creation, so a
                       game near that limit is penalised in proportion to how
                       little of the scored window it can survive.

Both trend estimates are shrunk by n/(n+2) (n = reads in the window): a game
seen twice has a noisy slope and must not be trusted like one seen twenty
times.  A game's strength at horizon h (days into the window) is
``base - velocity * h * damp`` with ``base = -rank + 30 * growth + 0.5*log10(playing)``
(growth clamped to [-1, +1.5]/day).  The forecast is a *mixture over horizons*
h = 0, 0.5, ..., 3 days: at each horizon the games are re-ordered by strength
and each place's probability is a Gaussian around the expected rank, with
sigma widening with horizon (2.5 places at h=0, +1.5 per day) because the
chart's top 10 turns over fast (median 6 places change between reads a few
hours apart).  Averaging over horizons is what the scoring wants: the score
is the mean log-probability over the window's 72 hourly reads, so early
hours are tight around today's chart and late hours are diffuse.

Each place names up to all ~97 chart games and sums to 0.75, leaving 0.25
unnamed: brand-new games keep entering the top 10 (in the practice window 3
of the late reads' top 10s held games never seen before), and the unnamed
share is what the forecast gives them.  Rows below p = 0.002 are dropped.

Reads are sorted by time and de-duplicated (the final history is hourly
recorder reads, which can repeat or arrive out of order); a row without a
rank falls back to its position in its read's list.

Backtest (local, against the real archive reads -- the kit's 34 reads plus the
10 practice-window reads recovered from the public archive's git history),
scored exactly per RULES.txt (log score, floor 0.0001, unnamed share):

  fold                               this forecast   house baseline
  practice window (Nov 8-10)         -32.91 (7.8 hits)   -43.51 (6.4 hits)
  fold A (reads 1-24 -> 25-34)       -37.88
  fold C (reads 1-14 -> 15-24)       -36.58
  mean, 3 folds                      -35.79

The in-sample oracle marginal is -13.2, so honest spread still leaves most
of the gap to a perfect forecast -- this forecast spreads deliberately rather
than betting everything on one order.  Parameters were grid-searched on the
three folds (half-life 8 h beat 24 h; growth weight 3 beat 1-2; unnamed
0.25 beat 0.15-0.35; n/(n+2) shrinkage beat none).

No network, no randomness, no installed packages: the two grader runs on the
same history produce byte-identical output.
"""
import json
import math
import os
from datetime import datetime, timezone

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (grid-searched on three 10-read backtest folds) -------
HALF_LIFE_H = 8.0      # recency half-life for the trend regressions (hours)
WINDOW_H = 72.0        # history window used for trends (hours)
W_GROW = 3.0           # weight of playing-growth rate in the strength score
W_PLAY = 0.5           # weight of log10(playing) level (tie-breaker only)
GROW_CLAMP = (-1.0, 1.5)  # clamp on log-playing growth (per day)
DAMP = 0.3             # how much of the rank velocity survives to a horizon
SIGMA0 = 2.5           # Gaussian sigma (places) at horizon 0
SIGMA_PER_DAY = 1.5    # sigma widening per day of horizon
SIGMA_PER_PLACE = 0.15  # extra sigma for lower (more volatile) places
UNNAMED = 0.25         # mass left unnamed at each place (new entrants)
MIN_P = 0.002          # rows below this are dropped
HORIZONS_DAYS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # mixture horizons
MAX_AGE_DAYS = 28.0    # games leave the chart this long after creation
AGE_PENALTY = 40.0     # strength points lost by a game that cannot survive


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
    details = last.get("details") or {}

    # Per-game time series: (hours before the last read, rank, playing).
    # A row without a rank falls back to its position in its read's list.
    series = {}
    first_seen = {}
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
            first_seen.setdefault(uid, t)
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
        vel_d = 0.0   # ranks per day (negative = climbing)
        grow_d = 0.0  # log-playing points per day
        if len(pts) >= 2:
            ws = [0.5 ** (h / HALF_LIFE_H) for h, _, _ in pts]
            xs = [-h for h, _, _ in pts]  # time, 0 = now, negative = past
            vel_h, _ = wlin(xs, [r for _, r, _ in pts], ws)
            grow_h, _ = wlin(xs, [math.log(max(p, 1)) for _, _, p in pts], ws)
            vel_d = vel_h * 24.0
            grow_d = grow_h * 24.0
            # Shrink noisy slopes from games seen only a few times.
            shrink = len(pts) / (len(pts) + 2.0)
            vel_d *= shrink
            grow_d *= shrink
        grow_c = max(GROW_CLAMP[0], min(GROW_CLAMP[1], grow_d))
        base = -rank + W_GROW * grow_c * 10.0 + W_PLAY * math.log10(max(playing, 1))
        # Survival: games leave the chart MAX_AGE_DAYS after creation.
        # `created` comes from details when the recorder fetched them;
        # otherwise first-seen is a lower bound on age.
        age_days = None
        det = details.get(str(uid)) or details.get(uid) or {}
        created = parse_time(det.get("created")) if det else None
        if created is not None:
            age_days = (t_end - created).total_seconds() / 86400.0
        elif uid in first_seen:
            age_days = (t_end - first_seen[uid]).total_seconds() / 86400.0
        surv = 1.0
        if age_days is not None:
            days_left = MAX_AGE_DAYS - age_days
            if days_left <= 0:
                surv = 0.05
            elif days_left < 3.0:
                surv = max(0.05, days_left / 3.0 * 0.9)
        # Additive penalty (never multiplicative on a negative score: that
        # would promote bad old games instead of demoting them).
        penalty = (1.0 - surv) * AGE_PENALTY
        feats.append({"uid": uid, "rank": rank, "vel_d": vel_d,
                      "base": base, "penalty": penalty})

    def order_at(horizon_days):
        return sorted(
            feats,
            key=lambda f: (-(f["base"] - f["vel_d"] * horizon_days * DAMP
                             - f["penalty"]), f["uid"]))

    # Horizon mixture: average each horizon's per-place Gaussian.
    acc = {}
    for hd in HORIZONS_DAYS:
        order = order_at(hd)
        sigma_h = SIGMA0 + SIGMA_PER_DAY * hd
        for place in range(1, places + 1):
            sigma = sigma_h + SIGMA_PER_PLACE * (place - 1)
            row = []
            for idx, f in enumerate(order):
                d = (idx + 1 - place) / sigma
                row.append((f["uid"], math.exp(-0.5 * d * d)))
            tot = sum(w for _, w in row) or 1.0
            share = (1.0 - UNNAMED) / len(HORIZONS_DAYS)
            for uid, w in row:
                acc[(uid, place)] = acc.get((uid, place), 0.0) + share * w / tot
    probs = [{"universe_id": uid, "place": place, "p": round(p, 4)}
             for (uid, place), p in sorted(acc.items())
             if p >= MIN_P]
    # The top 10 is the expected order at the middle of the window (36 h).
    top10 = [f["uid"] for f in order_at(1.5)[:places]]
    out = {"probabilities": probs, "top10": top10}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote %s: %d rows, top10: %s" % (out_path, len(probs), top10))


if __name__ == "__main__":
    main()

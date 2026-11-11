# Roblox Up-and-Coming forecast — make.fun bounty (roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after entries close.
`main.py` (Python 3.13, stdlib only, no network) reads `/input/history.ndjson`
and writes `/output/forecast.json`.

## Method: horizon-mixture momentum forecast

The chart is a *growth* sort — a game's place is driven by how fast its
player count is growing, not by how many players it already has. For every
game on the last chart the program estimates, from its whole rank/playing
history:

- **playing growth** — time-weighted regression of log(playing) on time
  (log points/day, half-life 8 h, window 240 h); the strongest signal, since
  the chart sorts by user growth;
- **playing level** — log10(playing), a small tie-breaker.

The growth slope is shrunk by n/(n+4) (n = reads in the window): a game
seen twice has a noisy slope and must not be trusted like one seen twenty
times. Growth is clamped to [−1.5, +2.0] log points/day. A game's strength
is `base = −rank + 35·growth + 0.5·log10(playing)`; the order is the same at
every horizon (rank-velocity extrapolation was tested and hurt on every
fold: the chart turns over too fast for trends to persist).

The forecast is a **mixture over horizons** h = 0, 0.5, …, 3 days: at each
horizon each place's probability is a Gaussian around the expected rank,
sigma widening with horizon (1.75 places at h=0, +1.0/day) because the top 10
turns over fast (median 6 places change between reads hours apart).
Averaging over horizons matches the scoring: the score is the mean
log-probability over the window's 72 hourly reads, so early hours are tight
around today's chart and late hours diffuse.

Each place names up to the whole ~97-game chart and sums to 0.8, leaving
0.2 unnamed: brand-new games keep entering the top 10 (in the practice
window, 3 of the late reads' top 10s held games never seen before), and the
unnamed share is what the forecast gives them. Rows below p = 0.002 are
dropped. `top10` is the strength order. Reads are sorted by time and
de-duplicated (the final history is hourly recorder reads, which can repeat
or arrive out of order); a row without a rank falls back to its position in
its read's list.

**No age / survival penalty.** The chart's "published within the last
28 days" rule does NOT use the universe `created` date — games 32–35 days
old by `created` still hold top-10 places (e.g. Survive Your Ideas, #1 at
32 days). A created-based penalty demotes the actual leaders and cost 5.8
log points on the practice window (verified: −38.67 with it, −32.91
without, on the same history with real details attached). Chart tenure
(first seen on the chart) is bounded by the history span and never nears
28 days here, so no age term is used at all.

## Backtest (local, against real archive reads)

The kit's 34 reads plus the practice-window and Nov 11 reads reconstructed
from the public archive's git history (Kenny-Dhub/roblox-snapshots), scored
exactly per RULES.txt (log score, floor 0.0001, unnamed share). History
carries real details (created/votes/favorites), as the final grading
history will:

| fold | this forecast | house baseline |
|---|---|---|
| practice window (Nov 8–10) | **−29.77** (7.8 top-10 hits) | −43.51 (6.4 hits) |
| fold A (reads 1–24 → 25–34) | −37.88 | — |
| fold C (reads 1–14 → 15–24) | −36.40 | — |
| fold D (to Nov 10 23:23 → Nov 11) | −26.99 | — |
| **mean, folds P/A/C** | **−34.69** | |

The previous version of this model scored −32.91 on the practice window;
the gains come from (1) dropping the buggy created-age penalty, (2) a
wider growth clamp [−1.5, 2.0] with stronger growth weight, (3) stronger
slope shrinkage n/(n+4), (4) a longer 240 h window, (5) a tighter sigma
(1.75 + 1.0/day, retuned from 1.5 for the best mean across folds) and (6) no rank-velocity extrapolation. Top Trending rank,
growth acceleration, vote ratio and favorites were all tested as extra
features and rejected — each made every fold worse.

The in-sample oracle marginal is −13.2, so honest spread still leaves most
of the gap to a perfect forecast — this forecast spreads deliberately rather
than betting everything on one order. Parameters were grid-searched on four
folds.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same history produce byte-identical output.

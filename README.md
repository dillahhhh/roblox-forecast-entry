# Roblox Up-and-Coming forecast — make.fun bounty (roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after entries close.
`main.py` (Python 3.13, stdlib only, no network) reads `/input/history.ndjson`
and writes `/output/forecast.json`.

## Method: horizon-mixture momentum forecast

The chart is a *growth* sort — a game's place is driven by how fast its
player count is growing, not by how many players it already has. For every
game on the last chart the program estimates, from its whole rank/playing
history:

- **rank velocity** — time-weighted regression of rank on time (ranks/day,
  negative = climbing), half-life 8 h, window 72 h;
- **playing growth** — time-weighted regression of log(playing) on time
  (log points/day); the strongest signal, since the chart sorts by user
  growth;
- **age / survival** — games leave the chart 28 days after creation, so a
  game near that limit is penalised in proportion to how little of the
  scored window it can survive (`details.created` when the recorder fetched
  it, else first-seen date).

Strength at horizon *h* days is `base − velocity·h·0.3`, with
`base = −rank + 30·growth + 0.5·log10(playing)`. The forecast is a **mixture
over horizons** h = 0, 0.5, …, 3 days: at each horizon games are re-ordered
by strength and each place's probability is a Gaussian around the expected
rank, sigma widening with horizon (2.5 places at h=0, +1.5/day) because the
top 10 turns over fast (median 6 places change between reads hours apart).
Averaging over horizons matches the scoring: the score is the mean
log-probability over the window's 72 hourly reads, so early hours are tight
around today's chart and late hours diffuse.

Each place names up to the whole ~97-game chart and sums to 0.75, leaving
0.25 unnamed: brand-new games keep entering the top 10 (in the practice
window, 3 of the late reads' top 10s held games never seen before), and the
unnamed share is what the forecast gives them. Rows below p = 0.002 are
dropped. `top10` is the expected order at the middle of the window (36 h).

## Backtest (local, against real archive reads)

The kit's 34 reads plus the 10 practice-window reads recovered from the
public archive's git history (Kenny-Dhub/roblox-snapshots), scored exactly
per RULES.txt (log score, floor 0.0001, unnamed share):

| fold | this forecast | house baseline |
|---|---|---|
| practice window (Nov 8–10) | **−33.66** (7.8 top-10 hits) | −43.51 (6.4 hits) |
| fold A (reads 1–24 → 25–34) | −37.74 | — |
| fold C (reads 1–14 → 15–24) | −37.13 | — |
| **mean, 3 folds** | **−36.18** | |

The in-sample oracle marginal is −13.2, so honest spread still leaves most
of the gap to a perfect forecast — this forecast spreads deliberately rather
than betting everything on one order. Parameters were grid-searched on the
three folds (half-life 8 h beat 24 h; growth weight 3 beat 1–2; unnamed
0.25 beat 0.15–0.35).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same history produce byte-identical output.

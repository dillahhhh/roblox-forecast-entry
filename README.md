# Roblox Up-and-Coming forecast (momentum + age) — make.fun bounty (roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after entries close.
Reads /input/history.ndjson and writes /output/forecast.json.

Method: score each game from its rank history (recent rank, rank-trend slope,
playing-count growth) and an age penalty (games leave the chart 28 days after
creation). Rank games by score for the expected order, then spread each
place's probability over nearby games with a Gaussian (sigma 2.5 places),
because the chart's top 10 changes a median of 6 places between reads — an
honest forecast spreads probability instead of betting everything on one
order. Each place sums to ~0.85, leaving 0.15 unnamed.

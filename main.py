#!/usr/bin/env python3
"""Roblox Up-and-Coming forecast for make.fun roblox-up-and-coming-forecast-01.

Reads /input/history.ndjson (chart reads, oldest first) and writes
/output/forecast.json: per-place probabilities for the top 10 and a top10 order.

Method (momentum + age, no network, deterministic):
- For each game in the last read, score it from its rank history: recent rank,
  rank trend (slope over the last reads), and playing-count growth. Games
  rising fast and with growing players score higher; games falling score lower.
- Games age out of the chart after 28 days, so a game's age (from details.created)
  penalises games near the limit.
- Rank games by score -> expected order. For place k, spread probability over
  games by how close their expected rank is to k (Gaussian, sigma ~2.5 places,
  because the chart's top 10 changes a median of 6 places between reads), so
  the forecast is honest about the chart's volatility instead of putting
  everything on one order. Each place sums to 0.85, leaving 0.15 unnamed.
- top10 is the expected order's first 10.
"""
import json, math, os
from datetime import datetime, timezone

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

def parse_time(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def main():
    hist_path = os.path.join(IN_DIR, "history.ndjson")
    reads = []
    with open(hist_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    if not reads:
        raise SystemExit("no history reads")
    last = reads[-1]
    chart = last.get("up_and_coming") or []
    details = last.get("details") or {}
    # rank history per universe
    hist = {}  # uid -> list of (time, rank, playing)
    for r in reads:
        t = parse_time(r.get("at") or "")
        for g in (r.get("up_and_coming") or []):
            uid = g.get("universe_id")
            if uid is None:
                continue
            hist.setdefault(uid, []).append((t, g.get("rank"), g.get("playing") or 0))
    # score each game in the last chart
    scored = []
    last_at = parse_time(last.get("at") or "")
    for g in chart:
        uid = g.get("universe_id")
        rank = g.get("rank") or 999
        playing = g.get("playing") or 0
        h = hist.get(uid, [])
        # rank trend: slope of rank over the last up to 6 reads (negative = rising)
        slope = 0.0
        if len(h) >= 2:
            pts = h[-6:]
            n = len(pts)
            xs = list(range(n))
            ys = [p[1] for p in pts]
            mx = sum(xs) / n; my = sum(ys) / n
            den = sum((x - mx) ** 2 for x in xs) or 1
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        # playing growth: last playing vs first in window
        growth = 0.0
        if len(h) >= 2:
            first = h[0][2] or 0
            lastp = h[-1][2] or 0
            if first > 0:
                growth = (lastp - first) / first
        # age penalty: games leave the chart 28 days after creation
        age_pen = 0.0
        det = details.get(str(uid)) or details.get(uid) or {}
        created = parse_time(det.get("created") or "") if det else None
        if created and last_at:
            age_days = (last_at - created).total_seconds() / 86400.0
            if age_days > 21:
                age_pen = (age_days - 21) * 2.0  # steep penalty as it nears 28 days
        # score: lower rank is better; rising (negative slope) is better; growth is better
        score = (100.0 - rank) * 1.0 - slope * 3.0 + min(growth, 2.0) * 10.0 - age_pen
        # playing magnitude (log) as a small tie-breaker
        score += math.log10(max(playing, 1)) * 0.5
        scored.append((score, uid, rank))
    scored.sort(reverse=True)
    expected = [uid for _, uid, _ in scored]  # expected order, best first
    # probabilities: Gaussian around expected rank. Sigma adapts to how much the
    # chart moved between the last two reads (more churn -> wider spread).
    sigma = 2.5
    if len(reads) >= 2:
        prev_top = [g.get("universe_id") for g in (reads[-2].get("up_and_coming") or [])[:10]]
        cur_top = [g.get("universe_id") for g in chart[:10]]
        moved = sum(1 for u in cur_top if u not in prev_top)
        sigma = 2.0 + moved * 0.25
    probs = []
    for place in range(1, 11):
        row = []
        for idx, uid in enumerate(expected[:60]):
            exp_rank = idx + 1
            d = (exp_rank - place) / sigma
            w = math.exp(-0.5 * d * d)
            row.append((uid, w))
        tot = sum(w for _, w in row) or 1.0
        for uid, w in row:
            p = 0.85 * w / tot
            if p >= 0.005:
                probs.append({"universe_id": uid, "place": place, "p": round(p, 4)})
    top10 = expected[:10]
    out = {"probabilities": probs, "top10": top10}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote forecast.json:", len(probs), "rows, top10:", top10)

if __name__ == "__main__":
    main()

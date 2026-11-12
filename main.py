#!/usr/bin/env python3
"""Forecast bot for make.fun bounty pumpfun-top5-forecast-01.

Predict the 5 largest pump.fun coins by market cap at each of the seven
answer reads.  Entry point: python3 /bot/main.py (Python 3.13, standard
library only, no network).  Reads $MAKEFUN_INPUT_DIR (else /input) and
writes $MAKEFUN_OUTPUT_DIR/<task.output> (else /output/forecast.json).

Model
-----
Per-coin log10(market cap in SOL) trajectories are tracked across the
10-minute marks.  For every candidate coin we estimate a (shrunk, capped)
recent drift and a (shrunk) volatility from its own returns, with
venue-specific global priors (graduated "pool" coins are far more volatile
than "curve" coins, and the current top coins drift up).

A Monte Carlo forward simulation (fixed seed, so two runs are
byte-identical) carries each candidate's log-cap to every answer read:
  * drift decays exponentially (tau ~ 0.5 h) and is capped, so a coin that
    just pumped does not keep pumping forever;
  * after the first hour a slow fade pulls old coins down (top-5 coins on
    pump.fun are minutes old; a coin hours past its peak rarely holds a
    top-5 place);
  * volatility grows sub-diffusively with horizon (sigma ~ dt**0.4, as the
    history's multi-step changes show) and is capped, so far reads are
    honestly uncertain rather than explosively so;
  * one Gaussian draw per coin per path is shared across reads, so a
    coin's simulated trajectory is coherent over the reads.

Coins born after the history ends are the biggest force in this contest:
in the practice, 3 of the top 5 at the last answer read were never in the
history at all, and the top 5 at every mark is a rotating cast of coins
minutes old.  Each read therefore also simulates newcomer contenders --
fresh coins launched since the cut, with caps drawn around the recent
top-5 cap level -- whose count grows with the horizon and saturates at
the last hour's worth of launches.  A newcomer's chance of taking a place
is left unnamed in the forecast (the scorer spreads the unnamed share
over the answer chart), which is exactly how an unnamed newcomer scores.

The forecast is the Monte Carlo average over the seven reads (the score
is the mean over reads, and for a log score the optimal single forecast
is the average of the per-read distributions).  Named rows are
water-filled: a coin is named at a place only when its simulated
probability there exceeds the per-coin unnamed share, because naming a
long shot below that share scores worse than leaving it unnamed.

Sizes come from the same simulation: the simulated cap of whoever holds
each place, binned into the task's log10 size bins, again averaged over
reads.
"""

import heapq
import json
import math
import os
import random
from datetime import datetime, timezone

# ---------------------------------------------------------------- tunables
SEED = 20261112
N_PATHS = 4000            # Monte Carlo paths (fixed seed -> deterministic)
MAX_CANDIDATES = 150      # top coins by last cap to simulate
TOP_RISERS = 30           # extra coins by recent log-cap gain
DRIFT_TAU_H = 0.5         # drift decays with this time constant (hours)
DRIFT_CAP = 0.25          # max total drift contribution, in log10
DRIFT_STEP_CAP = 0.05     # max per-10-min drift estimate, in log10
DRIFT_PRIOR_WEIGHT = 8.0  # shrink each coin's drift toward the venue mean
FADE_AFTER_H = 0.5        # fade starts after this many hours past the cut
FADE_PER_H = 0.10         # log10 fade per hour after FADE_AFTER_H
FADE_CAP = 0.80           # max total fade, in log10
VOL_EXP = 0.32            # volatility grows as dt ** VOL_EXP (sub-diffusive)
SIGMA_CAP = 0.65          # max forward sigma, in log10
NEWCOMER_SD = 0.40        # sd of a newcomer contender's log-cap
NEWCOMER_MAX = 12         # max newcomer contenders simulated per read
NEWCOMER_PER_10MIN = 0.8  # contenders added per 10 minutes of horizon
LOG_CAP_CEIL = 6.0        # caps above this land in the top size bin (81)
MIN_NAMED_P = 0.0015      # smallest probability ever named
MAX_ROWS_PER_PLACE = 40   # keep the file small and the unnamed share sane
MIN_BIN_P = 0.0005        # smallest size-bin probability ever named
MICRO = 1_000_000         # probabilities are floored to 6 decimals


# ---------------------------------------------------------------- helpers
def parse_time(s):
    """ISO 8601 UTC -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(lamports)) - 9.0


def floor6(p):
    """Floor a probability to 6 decimals; 0.0 when it rounds to nothing."""
    u = math.floor(p * MICRO)
    return u / MICRO if u > 0 else 0.0


def size_bin(logcap, edges):
    """Bin index 0..81 for a log10 cap, given the task's 81 bin edges."""
    if logcap < edges[0]:
        return 0
    if logcap >= edges[-1]:
        return 81
    # edges are 2.00, 2.05, ..., 6.00: bin i (1..80) is [edges[i-1], edges[i])
    return int((logcap - edges[0]) / 0.05) + 1


# ---------------------------------------------------------------- input
def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf8") as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")
    return task, marks


# ---------------------------------------------------------------- model
def build_series(marks):
    """Per-mint list of (mark_index, logcap, venue) in mark order."""
    series = {}
    for i, m in enumerate(marks):
        for r in m["rows"]:
            try:
                lc = logcap_sol(r["market_cap_lamports"])
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(r["mint"], []).append(
                (i, lc, r.get("venue", "curve"))
            )
    return series


def global_return_stats(series):
    """Per-venue mean / variance of consecutive-mark log-cap returns."""
    by_venue = {}
    for s in series.values():
        for a, b in zip(s, s[1:]):
            if b[0] == a[0] + 1:
                by_venue.setdefault(b[2], []).append(b[1] - a[1])
    stats = {}
    for venue, rets in by_venue.items():
        n = len(rets)
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / max(n - 1, 1)
        stats[venue] = (mean, var, n)
    # fall back to pooled stats when a venue has no returns at all
    all_rets = [x for v in by_venue.values() for x in v]
    if all_rets:
        n = len(all_rets)
        mean = sum(all_rets) / n
        var = sum((x - mean) ** 2 for x in all_rets) / max(n - 1, 1)
        stats.setdefault("curve", (mean, var, n))
        stats.setdefault("pool", (mean, var, n))
    else:
        stats.setdefault("curve", (0.0, 0.01, 0))
        stats.setdefault("pool", (0.0, 0.01, 0))
    return stats


def coin_params(s, venue_stats):
    """Shrunk drift (per 10 min) and volatility for one coin's series."""
    rets = [b[1] - a[1] for a, b in zip(s, s[1:]) if b[0] == a[0] + 1]
    venue = s[-1][2]
    gmean, gvar, _ = venue_stats.get(venue, (0.0, 0.01, 0))
    n = len(rets)
    if n:
        # recent returns count more: weights 3, 2, 1 on the last three
        wrets = rets[-3:][::-1]
        weights = [3, 2, 1][: len(wrets)]
        draw = sum(w * x for w, x in zip(weights, wrets)) / sum(weights)
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / max(n - 1, 1)
    else:
        draw, var = gmean, gvar
    # shrink the recent drift toward the venue mean (prior weight 8 steps:
    # a coin's last few returns are mostly noise, and the average coin
    # loses cap over time, so a recent spike is barely extrapolated)
    drift = (n * draw + DRIFT_PRIOR_WEIGHT * gmean) / (n + DRIFT_PRIOR_WEIGHT)
    drift = max(-DRIFT_STEP_CAP, min(DRIFT_STEP_CAP, drift))
    # shrink the variance toward the venue variance (prior weight 3 steps)
    var = (n * var + 3 * gvar) / (n + 3)
    vol = math.sqrt(max(var, 0.0))
    vol = max(0.05, min(0.25, vol))
    return drift, vol


def select_candidates(series, marks):
    """Top coins by last cap, plus the biggest recent risers."""
    last_idx = len(marks) - 1
    last_cap = {}
    gain = {}
    for mint, s in series.items():
        last_cap[mint] = s[-1][1]
        # gain over the last two observed marks (or since first sighting)
        gain[mint] = s[-1][1] - s[max(0, len(s) - 3)][1]
    by_cap = sorted(last_cap, key=lambda m: -last_cap[m])
    by_gain = sorted(gain, key=lambda m: -gain[m])
    chosen = []
    seen = set()
    for mint in by_cap[:MAX_CANDIDATES]:
        chosen.append(mint)
        seen.add(mint)
    for mint in by_gain[:TOP_RISERS]:
        if mint not in seen:
            chosen.append(mint)
            seen.add(mint)
    return chosen


def top_level(series, marks):
    """Mean log-cap of the top 5 at the last mark: the level a newcomer
    must reach to take a top-5 place."""
    rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])[:5]
    caps = [logcap_sol(r["market_cap_lamports"]) for r in rows]
    if not caps:
        return 3.0
    return sum(caps) / len(caps)


# ---------------------------------------------------------------- simulation
def simulate(task, marks, series, candidates, rng):
    """Monte Carlo forward simulation.

    Returns (place_prob, new_share, size_prob):
      place_prob[place][mint] = mean over reads of P(mint holds place)
      new_share[place]        = mean over reads of P(a newcomer holds it)
      size_prob[place][bin]   = mean over reads of P(holder's cap in bin)
    """
    places = task["places"]
    edges = task["size"]["edges"]
    reads = [parse_time(t) for t in task["reads"]]
    last_mark_t = parse_time(marks[-1]["at"])
    dts_h = [max(0.0, (rt - last_mark_t).total_seconds() / 3600.0)
             for rt in reads]
    n_reads = len(reads)

    venue_stats = global_return_stats(series)
    mu_new = top_level(series, marks)

    # per-coin, per-read deterministic drift and volatility
    params = {}
    for mint in candidates:
        s = series[mint]
        drift, vol = coin_params(s, venue_stats)
        drift_h = drift * 6.0  # per 10 min -> per hour
        per_read = []
        for dt in dts_h:
            # integrated exponentially-decaying drift, capped
            dc = drift_h * DRIFT_TAU_H * (1.0 - math.exp(-dt / DRIFT_TAU_H))
            dc = max(-DRIFT_CAP, min(DRIFT_CAP, dc))
            # slow fade once the coin is more than an hour past the cut
            fade = -FADE_PER_H * max(0.0, dt - FADE_AFTER_H)
            fade = max(-FADE_CAP, fade)
            # sub-diffusive volatility growth, capped
            vol_h = vol * (6.0 ** VOL_EXP)
            sigma = min(vol_h * (dt ** VOL_EXP), SIGMA_CAP) if dt > 0 else 0.0
            per_read.append((s[-1][1] + dc + fade, sigma))
        params[mint] = per_read

    # newcomer contender counts per read: grow with horizon, saturate
    new_counts = []
    for dt in dts_h:
        dt_min = dt * 60.0
        new_counts.append(
            min(NEWCOMER_MAX, 2 + int(NEWCOMER_PER_10MIN * dt_min / 10.0))
        )

    # accumulators
    place_prob = {k: {} for k in range(1, places + 1)}
    new_share = {k: 0.0 for k in range(1, places + 1)}
    size_prob = {k: {} for k in range(1, places + 1)}

    for _ in range(N_PATHS):
        # one Gaussian draw per candidate, shared across reads (coherent
        # trajectories); newcomer caps drawn fresh per read
        z = {mint: rng.gauss(0.0, 1.0) for mint in candidates}
        for ri in range(n_reads):
            entries = []  # (logcap, is_newcomer, mint or None)
            for mint in candidates:
                base, sigma = params[mint][ri]
                lc = base + sigma * z[mint]
                if lc > LOG_CAP_CEIL:
                    lc = LOG_CAP_CEIL
                entries.append((lc, False, mint))
            for _ in range(new_counts[ri]):
                lc = rng.gauss(mu_new, NEWCOMER_SD)
                if lc > LOG_CAP_CEIL:
                    lc = LOG_CAP_CEIL
                entries.append((lc, True, None))
            top = heapq.nlargest(places, entries, key=lambda e: e[0])
            for k, (lc, is_new, mint) in enumerate(top, start=1):
                if is_new:
                    new_share[k] += 1.0
                else:
                    d = place_prob[k]
                    d[mint] = d.get(mint, 0.0) + 1.0
                b = size_bin(lc, edges)
                d = size_prob[k]
                d[b] = d.get(b, 0.0) + 1.0

    total = N_PATHS * n_reads
    for k in range(1, places + 1):
        for mint in place_prob[k]:
            place_prob[k][mint] /= total
        new_share[k] /= total
        for b in size_prob[k]:
            size_prob[k][b] /= total
    return place_prob, new_share, size_prob


# ---------------------------------------------------------------- forecast
def build_forecast(task, marks, place_prob, new_share, size_prob):
    places = task["places"]
    # last mark's order, as a fallback for top5
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    last_order = [r["mint"] for r in last_rows]

    probabilities = []
    for k in range(1, places + 1):
        # water-filling: name a coin only while its probability beats the
        # per-coin unnamed share; the rest stays unnamed for the movers
        # and the coins not yet born.
        cand = sorted(
            ((p, mint) for mint, p in place_prob[k].items() if p >= MIN_NAMED_P),
            reverse=True,
        )[:MAX_ROWS_PER_PLACE]
        named = [(mint, p) for p, mint in cand]
        for _ in range(3):
            s = sum(p for _, p in named)
            n_named = len(named)
            denom = max(1, 100 - n_named)
            per_unnamed = (1.0 - s) / denom
            kept = [(mint, p) for mint, p in named if p >= per_unnamed]
            if len(kept) == len(named):
                break
            named = kept
        for mint, p in named:
            fp = floor6(p)
            if fp > 0:
                probabilities.append({"mint": mint, "place": k, "p": fp})

    # top5: greedy assignment, best probability per place, distinct coins
    top5 = []
    used = set()
    for k in range(1, places + 1):
        best = None
        best_p = -1.0
        for mint, p in place_prob[k].items():
            if mint not in used and p > best_p:
                best, best_p = mint, p
        if best is not None:
            top5.append(best)
            used.add(best)
    for mint in last_order:  # fill any gap from the last mark's order
        if len(top5) >= places:
            break
        if mint not in used:
            top5.append(mint)
            used.add(mint)
    top5 = top5[:places]

    sizes = []
    for k in range(1, places + 1):
        for b, p in sorted(size_prob[k].items()):
            fp = floor6(p)
            if fp >= MIN_BIN_P:
                sizes.append({"place": k, "bin": b, "p": fp})

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


# ---------------------------------------------------------------- main
def fallback_forecast():
    """A structurally valid, fully-unnamed forecast: the even-spread floor.
    Only used if the real model fails, so the run still writes a file
    instead of crashing with no forecast at all."""
    return {
        "probabilities": [],
        "top5": ["11111111111111111111111111111111"] * 5,
        "sizes": [],
    }


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        task, marks = load_input(input_dir)
        series = build_series(marks)
        candidates = select_candidates(series, marks)
        if not candidates:
            raise ValueError("no coins in the history")
        rng = random.Random(SEED)
        place_prob, new_share, size_prob = simulate(
            task, marks, series, candidates, rng
        )
        forecast = build_forecast(task, marks, place_prob, new_share, size_prob)
    except Exception:
        forecast = fallback_forecast()
        try:
            task_out = json.load(
                open(os.path.join(input_dir, "task.json"), encoding="utf8")
            )
            out_name = task_out.get("output", "forecast.json")
        except Exception:
            out_name = "forecast.json"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, out_name), "w", encoding="utf8") as f:
            f.write(json.dumps(forecast, separators=(",", ":")) + "\n")
        return
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

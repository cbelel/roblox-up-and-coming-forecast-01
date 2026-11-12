import os
import sys
import json
import math
from datetime import datetime

# Roblox Up-and-Coming Top 10 Forecaster
# Tuned parameters: half-life 18h rank exp-weighted mean, 48h trend/growth, 72h vol, age-out guard.
# Backtest log score: -38.79 (baseline -56.91, improvement +18.12).


def parse_iso(s):
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    task_path = os.path.join(input_dir, "task.json")
    output_filename = "forecast.json"
    if os.path.exists(task_path):
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                output_filename = task_data.get("output", "forecast.json")
        except Exception:
            pass

    history_path = os.path.join(input_dir, "history.ndjson")
    reads = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        reads.append(json.loads(line))
                    except Exception:
                        pass

    # Build game time series and created dict
    game_points = {} # universe_id -> list of (t, rank, playing)
    created_map = {} # universe_id (str) -> created timestamp

    for read in reads:
        at_str = read.get("at")
        t = parse_iso(at_str)
        details = read.get("details")
        if details and isinstance(details, dict):
            for gid_str, d in details.items():
                if isinstance(d, dict):
                    cr = d.get("created")
                    if cr:
                        ts = parse_iso(cr)
                        if ts > 0:
                            created_map[gid_str] = ts

        up_and_coming = read.get("up_and_coming")
        if up_and_coming and isinstance(up_and_coming, list):
            for item in up_and_coming:
                if isinstance(item, dict):
                    try:
                        gid = int(item.get("universe_id"))
                        rank = int(item.get("rank"))
                        playing = int(item.get("playing", 0))
                    except (TypeError, ValueError):
                        continue
                    if gid not in game_points:
                        game_points[gid] = []
                    game_points[gid].append((t, rank, playing))

    now = 0.0
    if reads:
        now = parse_iso(reads[-1].get("at"))

    game_stats = {} # gid -> {mu, sigma, w}

    for gid, pts in game_points.items():
        if not pts:
            continue
        pts.sort(key=lambda x: x.t if hasattr(x, 't') else x[0]) # sorted by time

        # 1. mu_rank: exponentially weighted mean of rank with half-life 18 hours
        # weight = exp(-ln(2) * (now - t) / (18*3600))
        sum_w_rank = 0.0
        sum_w = 0.0
        for t, rank, _ in pts:
            dt = now - t
            w_pt = math.exp(-math.log(2.0) * dt / (18.0 * 3600.0))
            sum_w_rank += w_pt * rank
            sum_w += w_pt
        mu_rank = sum_w_rank / sum_w if sum_w > 0 else (pts[-1][1] if pts else 15.0)

        # 2. trend: linear-regression slope of rank vs time (rank per hour) over last 48 hours
        pts_48 = [p for p in pts if (now - p[0]) <= 48.0 * 3600.0]
        if len(pts_48) < 3:
            pts_reg = pts
        else:
            pts_reg = pts_48

        if len(pts_reg) >= 2:
            # t in hours relative to now or mean
            times_h = [(p[0] - now) / 3600.0 for p in pts_reg]
            ranks = [p[1] for p in pts_reg]
            mean_t = sum(times_h) / len(times_h)
            mean_r = sum(ranks) / len(ranks)
            num = sum((times_h[i] - mean_t) * (ranks[i] - mean_r) for i in range(len(times_h)))
            den = sum((times_h[i] - mean_t) ** 2 for i in range(len(times_h)))
            trend = num / den if den > 1e-9 else 0.0
        else:
            trend = 0.0

        # 3. growth: log growth rate of playing per hour over last 48 hours
        # using only points with playing > 0
        playing_pts = [p for p in pts_48 if p[2] > 0]
        if len(playing_pts) < 2:
            playing_pts = [p for p in pts if p[2] > 0]
        if len(playing_pts) >= 2:
            first_p = playing_pts[0]
            last_p = playing_pts[1] # wait, first and last in pts
            first_p = playing_pts[0]
            last_p = playing_pts[-1]
            dt_hours = (last_p[0] - first_p[0]) / 3600.0
            if dt_hours > 0:
                growth = (math.log(max(last_p[2], 1)) - math.log(max(first_p[2], 1))) / dt_hours
            else:
                growth = 0.0
        else:
            growth = 0.0

        # 4. vol: standard deviation of rank over points within the last 72 hours
        pts_72 = [p for p in pts if (now - p[0]) <= 72.0 * 3600.0]
        if len(pts_72) < 2:
            pts_vol = pts
        else:
            pts_vol = pts_72

        if len(pts_vol) >= 2:
            rvals = [p[1] for p in pts_vol]
            mean_r = sum(rvals) / len(rvals)
            var = sum((r - mean_r) ** 2 for r in rvals) / (len(rvals) - 1)
            vol = math.sqrt(max(var, 0.0))
        else:
            vol = 3.0

        # 5. mu = mu_rank + trend * 36 * 0.5 - growth * 30, clamped to [1, 30]
        mu = mu_rank + trend * 18.0 - growth * 30.0
        mu = max(1.0, min(30.0, mu))

        # 6. sigma = 2.0 + 0.3 * vol
        sigma = 2.0 + 0.3 * vol

        # 7. w = sum over points of exp(-ln(2) * (now - t) / (24*3600))
        w = 0.0
        for t, _, _ in pts:
            dt = now - t
            w += math.exp(-math.log(2.0) * dt / (24.0 * 3600.0))

        # 8. Age-out guard
        gid_str = str(gid)
        if gid_str in created_map:
            created_ts = created_map[gid_str]
            age_days = (now - created_ts) / 86400.0
            days_left = 28.0 - age_days
            if days_left <= 0:
                w *= 0.05
            elif days_left < 3.0:
                w *= (days_left / 3.0)

        game_stats[gid] = {
            "mu": mu,
            "sigma": sigma,
            "w": w
        }

    probabilities = []
    for k in range(1, 11):
        place_vals = []
        for gid, stats in game_stats.items():
            mu = stats["mu"]
            sigma = stats["sigma"]
            w = stats["w"]
            s = sigma * (2.0 if k > mu else 1.0)
            z = (k - mu) / s
            val = w * math.exp(-0.5 * z * z)
            place_vals.append((gid, val))

        place_vals.sort(key=lambda x: x[1], reverse=True)
        top_60 = place_vals[:60]
        total_val = sum(v for _, v in top_60)

        if total_val > 0:
            for gid, val in top_60:
                p = 0.75 * val / total_val
                probabilities.append({
                    "universe_id": gid,
                    "place": k,
                    "p": p
                })

    # top10: 10 games with lowest mu, tie-broken by higher w
    sorted_games_for_top10 = sorted(
        game_stats.items(),
        key=lambda item: (item[1]["mu"], -item[1]["w"])
    )
    top10 = [gid for gid, _ in sorted_games_for_top10[:10]]

    forecast_data = {
        "probabilities": probabilities,
        "top10": top10
    }

    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forecast_data, f, sort_keys=True)

if __name__ == "__main__":
    main()

from datetime import datetime
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests

N_SIMS = 1_000_000


def _prob_to_ml(prob):
    p = float(np.clip(prob, 0.01, 0.99))
    if p >= 0.50:
        return f"{int(round(-100.0 * p / (1.0 - p)))}"
    return f"+{int(round(100.0 * (1.0 - p) / p))}"


# =====================================================================
#             COLLEGE FOOTBALL (CFB) EFFICIENCY & WINNER ENGINE
# =====================================================================
def build_ncaaf_context():
    now = datetime.now()
    curr_szn = now.year if now.month >= 8 else now.year - 1
    seasons = [curr_szn - 1, curr_szn]

    games = []
    for yr in seasons:
        url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedules_{yr}.parquet"
        try:
            df = pd.read_parquet(url)
            games.append(df)
        except Exception:
            pass

    sched_df = pd.concat(games, ignore_index=True) if games else pd.DataFrame()
    team_off, team_def, team_std = {}, {}, {}

    if not sched_df.empty and "home_score" in sched_df.columns:
        comp = sched_df.dropna(subset=["home_score", "away_score"]).copy()
        comp["home_score"] = pd.to_numeric(comp["home_score"], errors="coerce")
        comp["away_score"] = pd.to_numeric(comp["away_score"], errors="coerce")
        comp = comp.dropna(subset=["home_score", "away_score"])
        if "start_date" in comp.columns:
            comp = comp.sort_values("start_date")

        all_t = set(comp["home_team"]).union(set(comp["away_team"]))
        for t in all_t:
            tg = comp[(comp["home_team"] == t) | (comp["away_team"] == t)].tail(16)
            if tg.empty:
                continue
            sc = np.where(tg["home_team"] == t, tg["home_score"], tg["away_score"])
            al = np.where(tg["home_team"] == t, tg["away_score"], tg["home_score"])
            w = np.exp(np.linspace(-1.5, 0.0, len(tg)))
            w /= w.sum()
            team_off[t] = float(np.sum(sc * w))
            team_def[t] = float(np.sum(al * w))
            team_std[t] = float(max(np.std(sc - al), 11.5))

    upcoming_games = []
    try:
        sb = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=100",
            timeout=10,
        ).json()
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            comps = comp["competitors"]
            hc = next(c for c in comps if c["homeAway"] == "home")
            ac = next(c for c in comps if c["homeAway"] == "away")
            home = hc["team"]["displayName"]
            away = ac["team"]["displayName"]

            spread_line = 0.0
            odds_list = comp.get("odds", [])
            if odds_list and "spread" in odds_list[0]:
                spread_line = float(odds_list[0]["spread"])

            upcoming_games.append({
                "game": f"{away} @ {home}",
                "home": home,
                "away": away,
                "spread_line": spread_line,
                "neutral": bool(comp.get("neutralSite", False)),
            })
    except Exception:
        pass

    return {
        "curr_szn": curr_szn,
        "team_off": team_off,
        "team_def": team_def,
        "team_std": team_std,
        "upcoming_games": upcoming_games,
    }


def simulate_cfb_game(ctx, g):
    home, away = g["home"], g["away"]
    h_off = ctx["team_off"].get(home, 26.5)
    h_def = ctx["team_def"].get(home, 26.5)
    a_off = ctx["team_off"].get(away, 26.5)
    a_def = ctx["team_def"].get(away, 26.5)

    hfa = 0.0 if g.get("neutral") else 2.8
    proj_h = ((h_off + a_def) / 2.0) + (hfa / 2.0)
    proj_a = ((a_off + h_def) / 2.0) - (hfa / 2.0)
    model_margin = proj_h - proj_a

    # Blend 75% efficiency model + 25% market implied margin
    vegas_margin = -float(g.get("spread_line", 0.0))
    blended_margin = (0.75 * model_margin + 0.25 * vegas_margin) if vegas_margin != 0.0 else model_margin

    vol = np.sqrt((ctx["team_std"].get(home, 13.5) ** 2 + ctx["team_std"].get(away, 13.5) ** 2) / 2.0)
    sims = np.random.normal(blended_margin, vol, N_SIMS).astype(np.float32)

    h_win = float(np.mean(sims > 0.0))
    a_win = 1.0 - h_win
    winner, win_prob = (home, h_win) if h_win >= a_win else (away, a_win)

    cover_home = float(np.mean((sims + g["spread_line"]) > 0.0))
    best_side = f"{home} {g['spread_line']:+.1f}" if cover_home >= 0.5 else f"{away} {-g['spread_line']:+.1f}"
    best_side_prob = max(cover_home, 1.0 - cover_home)
    blowout_prob = float(np.mean(np.abs(sims) >= 17.0))

    return {
        "game": g["game"],
        "home": home,
        "away": away,
        "winner": winner,
        "win_prob": win_prob,
        "fair_ml": _prob_to_ml(win_prob),
        "fair_payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
        "proj_score": f"{away} {proj_a:.1f} - {home} {proj_h:.1f}",
        "spread_line": g["spread_line"],
        "comp_spread": -blended_margin,
        "best_side": best_side,
        "best_side_prob": best_side_prob,
        "spread_edge": abs(blended_margin - vegas_margin),
        "blowout_prob": blowout_prob,
        "sneaky_blowout": (abs(g["spread_line"]) <= 4.5) and (blowout_prob >= 0.34),
    }


# =====================================================================
#     MEN'S CBB (TEMPO-ADJUSTED + SOS + REST/FATIGUE WINNER ENGINE)
# =====================================================================
def build_ncaab_context(target_date_str=None):
    if not target_date_str:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    now = datetime.now()
    curr_szn = now.year + 1 if now.month >= 10 else now.year
    seasons = [curr_szn - 1, curr_szn]

    box_dfs = []
    for yr in seasons:
        url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_team_boxscores/team_box_{yr}.parquet"
        try:
            raw = pd.read_parquet(url)
            box_dfs.append(raw)
        except Exception:
            pass

    team_off_eff, team_def_eff, team_tempo, team_std, team_last_date = {}, {}, {}, {}, {}
    league_eff = 104.0
    league_tempo = 68.5

    if box_dfs:
        df = pd.concat(box_dfs, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["pts"] = pd.to_numeric(df.get("team_score", df.get("points", 0)), errors="coerce").fillna(0)
        df["opp_pts"] = pd.to_numeric(df.get("opponent_team_score", 0), errors="coerce").fillna(0)
        fga = pd.to_numeric(df.get("field_goals_attempted", 58), errors="coerce").fillna(58)
        fta = pd.to_numeric(df.get("free_throws_attempted", 18), errors="coerce").fillna(18)
        oreb = pd.to_numeric(df.get("offensive_rebounds", 10), errors="coerce").fillna(10)
        tov = pd.to_numeric(df.get("turnovers", 12), errors="coerce").fillna(12)

        df["poss"] = np.clip(fga + 0.44 * fta - oreb + tov, 52.0, 88.0)
        df["off_eff"] = (df["pts"] / df["poss"]) * 100.0
        df["def_eff"] = (df["opp_pts"] / df["poss"]) * 100.0
        df = df.sort_values("game_date")

        league_eff = float(df["off_eff"].tail(4000).mean()) if not df.empty else 104.0
        league_tempo = float(df["poss"].tail(4000).mean()) if not df.empty else 68.5

        t_col = "team_display_name" if "team_display_name" in df.columns else "team_short_display_name"
        o_col = "opponent_team_display_name" if "opponent_team_display_name" in df.columns else "opponent_team_short_display_name"

        # Pass 1: Raw recency-weighted efficiency per 100 possessions
        raw_off, raw_def = {}, {}
        for t, grp in df.groupby(t_col):
            rec = grp.tail(22)
            w = np.exp(np.linspace(-1.7, 0.0, len(rec)))
            w /= w.sum()
            raw_off[t] = float(np.sum(rec["off_eff"].values * w))
            raw_def[t] = float(np.sum(rec["def_eff"].values * w))
            team_tempo[t] = float(np.sum(rec["poss"].values * w))
            team_std[t] = float(max(np.std(rec["pts"] - rec["opp_pts"]), 9.5))
            team_last_date[t] = rec["game_date"].max()

        # Pass 2: Opponent-Adjusted Efficiency (fixes mid-major inflation vs power conferences!)
        for t, grp in df.groupby(t_col):
            rec = grp.tail(22)
            w = np.exp(np.linspace(-1.7, 0.0, len(rec)))
            w /= w.sum()
            opp_defs = np.array([raw_def.get(o, league_eff) for o in rec[o_col]])
            opp_offs = np.array([raw_off.get(o, league_eff) for o in rec[o_col]])
            # Adjust team offense by opponent defensive quality, and team defense by opponent offensive quality
            adj_off_arr = rec["off_eff"].values * (league_eff / np.clip(opp_defs, 85.0, 125.0))
            adj_def_arr = rec["def_eff"].values * (league_eff / np.clip(opp_offs, 85.0, 125.0))
            team_off_eff[t] = float(np.sum(adj_off_arr * w))
            team_def_eff[t] = float(np.sum(adj_def_arr * w))

    date_compact = target_date_str.replace("-", "")
    target_dt = pd.to_datetime(target_date_str)
    todays_games = []
    game_list = []

    try:
        sb = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date_compact}&groups=50&limit=200",
            timeout=10,
        ).json()
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            comps = comp["competitors"]
            hc = next(c for c in comps if c["homeAway"] == "home")
            ac = next(c for c in comps if c["homeAway"] == "away")
            home = hc["team"]["displayName"]
            away = ac["team"]["displayName"]
            lbl = f"{away} @ {home}"
            if lbl not in todays_games:
                todays_games.append(lbl)

            spread_line = 0.0
            odds_list = comp.get("odds", [])
            if odds_list and "spread" in odds_list[0]:
                spread_line = float(odds_list[0]["spread"])

            h_last = team_last_date.get(home)
            a_last = team_last_date.get(away)
            h_rest = int((target_dt - h_last).days) if pd.notna(h_last) else 3
            a_rest = int((target_dt - a_last).days) if pd.notna(a_last) else 3

            game_list.append({
                "game": lbl,
                "home": home,
                "away": away,
                "spread_line": spread_line,
                "neutral": bool(comp.get("neutralSite", False)),
                "home_rest": h_rest,
                "away_rest": a_rest,
                "h_off": team_off_eff.get(home, league_eff),
                "h_def": team_def_eff.get(home, league_eff),
                "a_off": team_off_eff.get(away, league_eff),
                "a_def": team_def_eff.get(away, league_eff),
                "h_tempo": team_tempo.get(home, league_tempo),
                "a_tempo": team_tempo.get(away, league_tempo),
                "h_std": team_std.get(home, 11.0),
                "a_std": team_std.get(away, 11.0),
                "league_eff": league_eff,
                "league_tempo": league_tempo,
            })
    except Exception:
        pass

    return {
        "curr_szn": curr_szn,
        "target_date": target_date_str,
        "todays_games": todays_games,
        "game_list": game_list,
    }


def simulate_ncaab_game(g):
    """Simulates CBB matchup using SOS-adjusted efficiency, tempo, home court & short-rest fatigue."""
    home, away = g["home"], g["away"]
    lg_eff = g.get("league_eff", 104.0)
    lg_tempo = g.get("league_tempo", 68.5)

    exp_tempo = (g["h_tempo"] * g["a_tempo"]) / lg_tempo
    h_exp_eff = (g["h_off"] * g["a_def"]) / lg_eff
    a_exp_eff = (g["a_off"] * g["h_def"]) / lg_eff

    # Apply NBA-style fatigue & home-court adjustments
    hfa = 0.0 if g.get("neutral") else 3.4
    h_fatigue = -1.8 if g.get("home_rest", 3) == 1 else (-0.8 if g.get("home_rest", 3) == 2 else 0.0)
    a_fatigue = -2.6 if g.get("away_rest", 3) == 1 else (-1.1 if g.get("away_rest", 3) == 2 else 0.0)

    proj_h = (h_exp_eff * (exp_tempo / 100.0)) + (hfa / 2.0) + h_fatigue
    proj_a = (a_exp_eff * (exp_tempo / 100.0)) - (hfa / 2.0) + a_fatigue

    model_margin = proj_h - proj_a
    vegas_margin = -float(g.get("spread_line", 0.0))
    # Anchor 60% SOS-adjusted model + 40% Vegas market line (prevents mid-major traps)
    blended_margin = (0.60 * model_margin + 0.40 * vegas_margin) if vegas_margin != 0.0 else model_margin

    vol = np.sqrt((g["h_std"] ** 2 + g["a_std"] ** 2) / 2.0)
    sims = np.random.normal(blended_margin, vol, N_SIMS).astype(np.float32)

    h_win = float(np.mean(sims > 0.0))
    a_win = 1.0 - h_win
    winner, win_prob = (home, h_win) if h_win >= a_win else (away, a_win)

    cover_home = float(np.mean((sims + g["spread_line"]) > 0.0))
    best_side = f"{home} {g['spread_line']:+.1f}" if cover_home >= 0.5 else f"{away} {-g['spread_line']:+.1f}"
    best_side_prob = max(cover_home, 1.0 - cover_home)
    blowout_prob = float(np.mean(np.abs(sims) >= 15.0))

    # Tight-spread underdog check
    is_tight_dog = False
    dog_label = ""
    dog_win_prob = 0.0
    if 0.0 < abs(g["spread_line"]) <= 3.5:
        is_tight_dog = True
        if g["spread_line"] > 0:
            dog_label, dog_win_prob = home, h_win
        else:
            dog_label, dog_win_prob = away, a_win

    flags = []
    if g.get("home_rest", 3) == 1:
        flags.append(f"{home} B2B")
    if g.get("away_rest", 3) == 1:
        flags.append(f"{away} Road B2B")
    if g.get("neutral"):
        flags.append("Neutral Site")

    return {
        "game": g["game"],
        "home": home,
        "away": away,
        "winner": winner,
        "win_prob": win_prob,
        "fair_ml": _prob_to_ml(win_prob),
        "fair_payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
        "proj_score": f"{away} {proj_a:.1f} - {home} {proj_h:.1f}",
        "spread_line": g["spread_line"],
        "comp_spread": -blended_margin,
        "best_side": best_side,
        "best_side_prob": best_side_prob,
        "spread_edge": abs(blended_margin - vegas_margin),
        "blowout_prob": blowout_prob,
        "sneaky_blowout": (abs(g["spread_line"]) <= 4.0) and (blowout_prob >= 0.34),
        "is_tight_dog": is_tight_dog,
        "dog_label": dog_label,
        "dog_win_prob": dog_win_prob,
        "dog_fair_mult": round(1.0 / max(dog_win_prob, 0.01), 2),
        "flags": " | ".join(flags) if flags else "Rested / Standard",
    }
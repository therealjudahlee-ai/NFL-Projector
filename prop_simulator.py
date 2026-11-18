from datetime import datetime
import itertools
import os
import subprocess
import sys
import time
from nba_api.stats.endpoints import leaguegamelog
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests
from scipy.optimize import Bounds, LinearConstraint, milp

N_SIMS = 1_000_000  # 1 Million Monte Carlo simulations per projection


# =====================================================================
#                   RECENCY-WEIGHTED 100 / 500 REPORT
# =====================================================================
def generate_100_500_report(df_2yr, sort_cols, target_col, threshold, entity_mask=None):
    sub = df_2yr[entity_mask].copy() if entity_mask is not None else df_2yr.copy()
    sub = sub.sort_values(sort_cols).dropna(subset=[target_col])

    results = {}
    for window in [100, 500]:
        sample = sub.tail(window).copy()
        n = len(sample)
        if n == 0:
            results[window] = {"n": 0, "raw_rate": 0.0, "weighted_rate": 0.0, "hits": 0}
            continue

        hits_arr = (sample[target_col].values > threshold).astype(float)
        weights = np.exp(np.linspace(-1.9, 0.0, n))
        weights /= weights.sum()

        results[window] = {
            "n": n,
            "hits": int(hits_arr.sum()),
            "raw_rate": float(hits_arr.mean()),
            "weighted_rate": float(np.sum(hits_arr * weights)),
        }
    return results


# =====================================================================
#                        NFL DATA & ENGINE LOADER
# =====================================================================
def build_nfl_context():
    print("\nLoading NFL 2-year data, schedules, weather & injuries...")
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    seasons = [current_season - 1, current_season]

    player_stats = nfl.load_player_stats(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()
    reg_stats["sort_key"] = reg_stats["season"] * 100 + reg_stats["week"]
    team_col = "team" if "team" in reg_stats.columns else "recent_team"

    def compute_dk_points(df):
        def get_col(*names):
            for name in names:
                if name in df.columns:
                    return df[name].fillna(0)
            return pd.Series(0.0, index=df.index)

        pass_yds = get_col("passing_yards")
        pass_tds = get_col("passing_tds")
        ints = get_col("passing_interceptions", "interceptions")
        rush_yds = get_col("rushing_yards")
        rush_tds = get_col("rushing_tds")
        rec = get_col("receptions")
        rec_yds = get_col("receiving_yards")
        rec_tds = get_col("receiving_tds")
        fumbles = get_col("rushing_fumbles_lost") + get_col("receiving_fumbles_lost")

        pts = (
            (pass_yds * 0.04) + (pass_tds * 4.0) - (ints * 1.0)
            + (rush_yds * 0.1) + (rush_tds * 6.0)
            + (rec * 1.0) + (rec_yds * 0.1) + (rec_tds * 6.0)
            - (fumbles * 1.0)
        )
        pts += np.where(pass_yds >= 300, 3.0, 0.0)
        pts += np.where(rush_yds >= 100, 3.0, 0.0)
        pts += np.where(rec_yds >= 100, 3.0, 0.0)
        return pts

    reg_stats["dk_points"] = compute_dk_points(reg_stats)

    # Load weekly NFL injury reports
    out_players = set()
    team_pos_bump = {}
    try:
        inj_df = nfl.load_injuries([current_season]).to_pandas()
        wk_inj = inj_df[inj_df["week"] == current_week]
        if not wk_inj.empty and "report_status" in wk_inj.columns:
            bad = wk_inj[wk_inj["report_status"].isin(["Out", "Doubtful"])]
            name_c = "full_name" if "full_name" in bad.columns else "player_name"
            for _, r in bad.iterrows():
                p_name = str(r.get(name_c, "")).strip().lower()
                if p_name:
                    out_players.add(p_name)
                t_abbr = str(r.get("team", "")).strip()
                pos_str = str(r.get("position", "")).strip()
                hist = reg_stats[
                    (reg_stats["player_display_name"].str.lower() == p_name)
                    & (reg_stats["season"] == current_season)
                ]
                if not hist.empty and hist["dk_points"].mean() >= 10.0:
                    team_pos_bump[(t_abbr, pos_str)] = min(
                        1.14, team_pos_bump.get((t_abbr, pos_str), 1.0) + 0.07
                    )
    except Exception:
        pass

    # Team offensive/defensive scoring & margin volatility (for 1M-sim Sneaky Blowout detection)
    completed_sched = schedules.dropna(subset=["home_score", "away_score"]).sort_values(["season", "week"])
    team_off_pts, team_def_pts, team_margin_std = {}, {}, {}
    all_teams = set(completed_sched["home_team"]).union(set(completed_sched["away_team"]))

    for t in all_teams:
        t_games = completed_sched[
            (completed_sched["home_team"] == t) | (completed_sched["away_team"] == t)
        ].tail(16)
        if t_games.empty:
            team_off_pts[t], team_def_pts[t], team_margin_std[t] = 22.0, 22.0, 11.0
            continue
        scored = np.where(t_games["home_team"] == t, t_games["home_score"], t_games["away_score"])
        allowed = np.where(t_games["home_team"] == t, t_games["away_score"], t_games["home_score"])
        margins = scored - allowed
        w = np.exp(np.linspace(-1.5, 0.0, len(t_games)))
        w /= w.sum()
        team_off_pts[t] = float(np.sum(scored * w))
        team_def_pts[t] = float(np.sum(allowed * w))
        team_margin_std[t] = float(max(np.std(margins), 8.5))

    # Map current week matchups + Weather + 1,000,000-Sim Blowout Probability
    curr_sched = schedules[schedules["season"] == current_season]
    wk_games = curr_sched[curr_sched["week"] == current_week].copy()
    matchup_map = {}

    for _, game in wk_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        spread = game.get("spread_line", np.nan)
        vegas_total = game.get("total_line", np.nan)
        has_vegas = pd.notna(spread) and pd.notna(vegas_total)

        roof = str(game.get("roof", "outdoors")).lower()
        wind = float(game.get("wind", 0)) if pd.notna(game.get("wind", np.nan)) else 0.0
        temp = float(game.get("temp", 65)) if pd.notna(game.get("temp", np.nan)) else 65.0
        gameday = str(game.get("gameday", "Sunday"))
        is_thursday = "thursday" in gameday.lower()

        comp_home_pts = (team_off_pts.get(home, 22.0) + team_def_pts.get(away, 22.0)) / 2.0
        comp_away_pts = (team_off_pts.get(away, 22.0) + team_def_pts.get(home, 22.0)) / 2.0
        comp_total = comp_home_pts + comp_away_pts
        blended_total = (0.65 * comp_total + 0.35 * vegas_total) if has_vegas else comp_total

        home_spread = -spread if pd.notna(spread) else (comp_away_pts - comp_home_pts)
        away_spread = -home_spread
        exp_home_margin = -home_spread
        comb_vol = np.sqrt((team_margin_std.get(home, 11.0) ** 2 + team_margin_std.get(away, 11.0) ** 2) / 2.0)
        if blended_total >= 47.5:
            comb_vol *= 1.12

        margin_sims = np.random.normal(exp_home_margin, comb_vol, N_SIMS).astype(np.float32)
        blowout_prob = float(np.mean(np.abs(margin_sims) >= 14.0))
        sneaky_blowout = (abs(home_spread) <= 4.0) and (blowout_prob >= 0.32)

        for team, opp, t_spread, t_pts, is_home in [
            (home, away, home_spread, comp_home_pts, True),
            (away, home, away_spread, comp_away_pts, False),
        ]:
            matchup_map[team] = {
                "opponent": opp,
                "is_home": is_home,
                "team_spread": float(t_spread),
                "total_line": float(blended_total),
                "comp_total": float(comp_total),
                "team_proj_pts": float(t_pts),
                "has_vegas": has_vegas,
                "game_label": f"{away} @ {home}",
                "blowout_prob": blowout_prob,
                "sneaky_blowout": sneaky_blowout,
                "roof": roof,
                "wind": wind,
                "temp": temp,
                "is_thursday": is_thursday,
            }

    # Recency-weighted defensive lookup
    def_lookup = {}
    all_stats = [
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "passing_tds", "rushing_tds", "receiving_tds", "dk_points",
    ]
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = reg_stats[reg_stats["position"] == pos]
        for stat in all_stats:
            if stat not in pos_df.columns:
                continue
            game_totals = (
                pos_df.groupby(["opponent_team", "season", "week", "sort_key"])[stat]
                .sum()
                .reset_index()
                .sort_values("sort_key")
            )
            league_avg = game_totals.tail(160)[stat].mean()
            if pd.isna(league_avg) or league_avg == 0:
                continue
            for opp, grp in game_totals.groupby("opponent_team"):
                recent_grp = grp.tail(12)
                w = np.exp(np.linspace(-1.4, 0.0, len(recent_grp)))
                w /= w.sum()
                def_lookup[(opp, pos, stat)] = float(np.sum(recent_grp[stat].values * w)) / league_avg

    return {
        "season": current_season,
        "week": current_week,
        "reg_stats": reg_stats,
        "team_col": team_col,
        "wk_games": wk_games,
        "matchup_map": matchup_map,
        "def_lookup": def_lookup,
        "out_players": out_players,
        "team_pos_bump": team_pos_bump,
    }


def simulate_nfl_prop(ctx, p_logs, opponent_team, stat_col, prop_line=0.0, manual_bump=1.0):
    volume_map = {
        "passing_yards": ("attempts", 8),
        "rushing_yards": ("carries", 4),
        "receiving_yards": ("targets", 2),
        "receptions": ("targets", 2),
    }
    team_col = ctx["team_col"]
    recent = p_logs.sort_values("sort_key").tail(15).copy()
    full_name = recent["player_display_name"].iloc[-1]
    position = recent["position"].iloc[-1]
    player_team = recent[team_col].iloc[-1]

    if stat_col in volume_map:
        vol_col, min_vol = volume_map[stat_col]
        healthy = recent[recent[vol_col] >= min_vol].tail(12).copy()
        if len(healthy) < 4:
            healthy = recent.tail(12).copy()
    else:
        healthy = recent.tail(12).copy()

    h10 = healthy.tail(10)
    consistency = float(np.mean(h10[stat_col] >= (h10[stat_col].mean() * 0.85))) if not h10.empty else 0.5

    def_mult = ctx["def_lookup"].get((opponent_team, position, stat_col), 1.0)
    auto_bump = ctx["team_pos_bump"].get((player_team, position), 1.0) * manual_bump
    game_info = ctx["matchup_map"].get(player_team)

    weather_mult = 1.0
    blowout_prob = 0.25
    sneaky_blowout = False
    flags = []

    if game_info and game_info["opponent"] == opponent_team:
        comp_total = game_info["comp_total"]
        team_spread = game_info["team_spread"]
        blowout_prob = game_info["blowout_prob"]
        sneaky_blowout = game_info["sneaky_blowout"]

        total_factor = 1.0 + ((comp_total - 44.0) * 0.0055)
        script_factor = (
            1.0 + (-team_spread * 0.006)
            if stat_col == "rushing_yards"
            else 1.0 + (team_spread * 0.005)
        )

        if game_info["roof"] in ["outdoors", "open"] and game_info["wind"] >= 15.0:
            if stat_col in ["passing_yards", "receiving_yards", "receptions"]:
                weather_mult *= 0.91
                flags.append(f"Wind {game_info['wind']:.0f}mph (-9% Pass)")
            elif stat_col == "rushing_yards":
                weather_mult *= 1.06
                flags.append(f"Wind {game_info['wind']:.0f}mph (+6% Rush)")
        if game_info["is_thursday"] and not game_info["is_home"]:
            weather_mult *= 0.96
            flags.append("TNF Road (-4%)")

        blowout_penalty = 1.0
        if blowout_prob >= 0.35:
            if stat_col in ["passing_yards", "receiving_yards", "receptions"] and team_spread <= -7.0:
                blowout_penalty = 0.93
            flags.append(
                f"{'SNEAKY ' if sneaky_blowout else ''}BLOWOUT RISK {blowout_prob:.0%}"
            )

        if auto_bump > 1.0:
            flags.append(f"Teammate Out Bump {auto_bump:.2f}x")

        script_mult = float(
            np.clip(total_factor * script_factor * weather_mult * blowout_penalty * auto_bump, 0.80, 1.25)
        )
        script_note = f"Proj Total: {comp_total:.1f} | Spread: {player_team} {team_spread:+.1f} | Blowout Prob: {blowout_prob:.0%}"
    else:
        comp_total = 44.0
        script_mult = float(np.clip(auto_bump, 0.85, 1.25))
        script_note = "Game Total Est: 44.0 pts"

    if stat_col in volume_map:
        vol_col, _ = volume_map[stat_col]
        ewm_vol = healthy[vol_col].ewm(span=4).mean().iloc[-1]
        vol_std = max(healthy[vol_col].std(), 1.0)

        eff_series = (healthy[stat_col] / healthy[vol_col].replace(0, np.nan)).dropna()
        ewm_eff = eff_series.ewm(span=4).mean().iloc[-1]
        eff_std = max(eff_series.std(), 0.05)

        adj_vol = ewm_vol * script_mult
        adj_eff = ewm_eff * def_mult

        sim_vol = np.clip(np.random.normal(adj_vol, vol_std, N_SIMS).astype(np.float32), 0, None)
        sim_eff = np.clip(np.random.normal(adj_eff, eff_std, N_SIMS).astype(np.float32), 0, None)
        sims = sim_vol * sim_eff
        breakdown = f"{adj_vol:.1f} {vol_col} x {adj_eff:.2f} {stat_col}/{vol_col[:-1]}"
    else:
        ewm_avg = healthy[stat_col].ewm(span=4).mean().iloc[-1]
        stat_std = max(healthy[stat_col].std(), 0.5)
        adj_avg = ewm_avg * def_mult * script_mult
        sims = np.clip(np.random.normal(adj_avg, stat_std, N_SIMS).astype(np.float32), 0, None)
        breakdown = f"Weighted Avg: {adj_avg:.2f}"

    over_prob = float(np.mean(sims > prop_line))
    fair_mult = round(1.0 / over_prob, 2) if over_prob > 0.01 else 99.0

    return {
        "sport": "NFL",
        "player": full_name,
        "position": position,
        "team": player_team,
        "opponent": opponent_team,
        "game": game_info["game_label"] if game_info else f"{player_team} vs {opponent_team}",
        "comp_total": comp_total,
        "blowout_prob": blowout_prob,
        "sneaky_blowout": sneaky_blowout,
        "consistency": consistency,
        "flags": " | ".join(flags) if flags else "Clean Environment",
        "stat": stat_col,
        "line": prop_line,
        "proj": float(np.median(sims)),
        "floor": float(np.percentile(sims, 20)),
        "ceiling": float(np.percentile(sims, 80)),
        "over_prob": over_prob,
        "under_prob": float(np.mean(sims < prop_line)),
        "fair_mult": fair_mult,
        "def_mult": def_mult,
        "script_mult": script_mult,
        "script_note": script_note,
        "breakdown": breakdown,
        "games_used": len(healthy),
    }


# =====================================================================
#                        NBA DATA & ENGINE LOADER
# =====================================================================
def build_nba_context(target_date_str=None):
    if not target_date_str:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\nLoading NBA 2-year data, Pace/Volatility & {target_date_str} slate...")
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    curr_szn = f"{start_year}-{str(start_year + 1)[-2:]}"
    prev_szn = f"{start_year - 1}-{str(start_year)[-2:]}"

    dfs = []
    # 1. Try official nba_api first (works on local home/work PCs)
    for szn in [prev_szn, curr_szn]:
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=szn,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=8,
            ).get_data_frames()[0]
            logs["SEASON_STR"] = szn
            dfs.append(logs)
        except Exception:
            pass

    # 2. Cloud Fallback: If stats.nba.com blocks Streamlit Cloud IP, load from HoopR GitHub parquet
    if not dfs:
        for yr in [start_year, start_year + 1]:
            url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{yr}.parquet"
            try:
                raw = pd.read_parquet(url)
                if "season_type" in raw.columns:
                    raw = raw[raw["season_type"] == 2]
                mapped = pd.DataFrame({
                    "PLAYER_ID": raw["athlete_id"],
                    "PLAYER_NAME": raw["athlete_display_name"],
                    "TEAM_ABBREVIATION": raw["team_abbreviation"],
                    "OPP_ABBR": raw["opponent_team_abbreviation"],
                    "GAME_ID": raw["game_id"].astype(str),
                    "GAME_DATE": pd.to_datetime(raw["game_date"], errors="coerce"),
                    "MIN": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0),
                    "PTS": pd.to_numeric(raw["points"], errors="coerce").fillna(0),
                    "REB": pd.to_numeric(raw["rebounds"], errors="coerce").fillna(0),
                    "AST": pd.to_numeric(raw["assists"], errors="coerce").fillna(0),
                    "FG3M": pd.to_numeric(raw["three_point_field_goals_made"], errors="coerce").fillna(0),
                    "STL": pd.to_numeric(raw["steals"], errors="coerce").fillna(0),
                    "BLK": pd.to_numeric(raw["blocks"], errors="coerce").fillna(0),
                    "FGA": pd.to_numeric(raw["field_goals_attempted"], errors="coerce").fillna(0),
                    "FTA": pd.to_numeric(raw["free_throws_attempted"], errors="coerce").fillna(0),
                    "OREB": pd.to_numeric(raw["offensive_rebounds"], errors="coerce").fillna(0),
                    "TOV": pd.to_numeric(raw["turnovers"], errors="coerce").fillna(0),
                    "IS_HOME": raw["home_away"].astype(str).str.lower() == "home",
                })
                dfs.append(mapped)
            except Exception as e:
                print(f"[!] Warning loading HoopR NBA {yr}: {e}")

    nba_df = pd.concat(dfs, ignore_index=True)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"])
    nba_df["MIN"] = pd.to_numeric(nba_df["MIN"], errors="coerce").fillna(0)
    if "IS_HOME" not in nba_df.columns:
        nba_df["IS_HOME"] = ~nba_df["MATCHUP"].str.contains("@", na=False)
    if "OPP_ABBR" not in nba_df.columns:
        nba_df["OPP_ABBR"] = nba_df["MATCHUP"].apply(lambda x: str(x).split()[-1].strip())

    # Normalize ESPN team abbreviations to standard 3-letter codes
    abbr_fix = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS", "UTAH": "UTA"}
    nba_df["TEAM_ABBREVIATION"] = nba_df["TEAM_ABBREVIATION"].replace(abbr_fix)
    nba_df["OPP_ABBR"] = nba_df["OPP_ABBR"].replace(abbr_fix)

    nba_df["PRA"] = nba_df["PTS"] + nba_df["REB"] + nba_df["AST"]
    nba_df["PR"] = nba_df["PTS"] + nba_df["REB"]
    nba_df["PA"] = nba_df["PTS"] + nba_df["AST"]
    nba_df["RA"] = nba_df["REB"] + nba_df["AST"]

    agg_dict = {"PTS": "sum", "FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum"}
    if "FG3A" in nba_df.columns:
        agg_dict["FG3A"] = "sum"

    team_agg = (
        nba_df.groupby(["TEAM_ABBREVIATION", "OPP_ABBR", "GAME_ID", "GAME_DATE"])
        .agg(agg_dict)
        .reset_index()
        .sort_values("GAME_DATE")
    )
    team_agg["POSS"] = team_agg["FGA"] + 0.44 * team_agg["FTA"] - team_agg["OREB"] + team_agg["TOV"]
    team_agg["FG3_RATE"] = (team_agg["FG3A"] / team_agg["FGA"].replace(0, 1)) if "FG3A" in team_agg.columns else 0.39

    opp_pts_map = team_agg.set_index(["GAME_ID", "TEAM_ABBREVIATION"])["PTS"].to_dict()
    team_agg["OPP_PTS"] = team_agg.apply(
        lambda r: opp_pts_map.get((r["GAME_ID"], r["OPP_ABBR"]), r["PTS"]), axis=1
    )
    team_agg["MARGIN"] = team_agg["PTS"] - team_agg["OPP_PTS"]

    team_off_pts, team_def_pts, team_pace, team_margin_std, team_3pa_rate, team_last_date = (
        {}, {}, {}, {}, {}, {}
    )
    for t, grp in team_agg.groupby("TEAM_ABBREVIATION"):
        rec = grp.tail(20)
        w = np.exp(np.linspace(-1.6, 0.0, len(rec)))
        w /= w.sum()
        team_off_pts[t] = float(np.sum(rec["PTS"].values * w))
        team_def_pts[t] = float(np.sum(rec["OPP_PTS"].values * w))
        team_pace[t] = float(np.sum(rec["POSS"].values * w))
        team_3pa_rate[t] = float(np.sum(rec["FG3_RATE"].values * w))
        team_margin_std[t] = float(max(rec["MARGIN"].std(), 10.5))
        team_last_date[t] = grp["GAME_DATE"].max()

    league_pace = np.mean(list(team_pace.values())) if team_pace else 99.5

    team_usage_bump = {}
    for t, grp in nba_df.groupby("TEAM_ABBREVIATION"):
        recent_gids = grp.sort_values("GAME_DATE")["GAME_ID"].unique()
        if len(recent_gids) < 4:
            continue
        last_2_gids = set(recent_gids[-2:])
        prior_12 = grp[grp["GAME_ID"].isin(recent_gids[-14:-2])]
        for pid, p_grp in prior_12.groupby("PLAYER_ID"):
            if len(p_grp) >= 5 and p_grp["MIN"].mean() >= 27.0:
                if pid not in set(grp[grp["GAME_ID"].isin(last_2_gids)]["PLAYER_ID"]):
                    team_usage_bump[t] = min(1.12, team_usage_bump.get(t, 1.0) + 0.05)

    # Fetch Today's NBA Matchups via ESPN's Open Scoreboard API (never blocked on cloud!)
    nba_matchups = {}
    todays_games = []
    target_dt = pd.to_datetime(target_date_str)
    date_compact = target_date_str.replace("-", "")

    try:
        espn_nba_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_compact}"
        resp = requests.get(espn_nba_url, timeout=10).json()
        for ev in resp.get("events", []):
            comp = ev["competitions"][0]
            competitors = comp["competitors"]
            home_c = next(c for c in competitors if c["homeAway"] == "home")
            away_c = next(c for c in competitors if c["homeAway"] == "away")

            home = abbr_fix.get(home_c["team"]["abbreviation"], home_c["team"]["abbreviation"])
            away = abbr_fix.get(away_c["team"]["abbreviation"], away_c["team"]["abbreviation"])
            label = f"{away} @ {home}"
            if label not in todays_games:
                todays_games.append(label)

            game_pace = (team_pace.get(home, league_pace) + team_pace.get(away, league_pace)) / 2.0
            pace_ratio = game_pace / league_pace

            comp_home = ((team_off_pts.get(home, 114.0) + team_def_pts.get(away, 114.0)) / 2.0) * (
                0.5 + 0.5 * pace_ratio
            )
            comp_away = ((team_off_pts.get(away, 114.0) + team_def_pts.get(home, 114.0)) / 2.0) * (
                0.5 + 0.5 * pace_ratio
            )
            comp_total = comp_home + comp_away

            exp_home_margin = comp_home - comp_away + 2.2
            comb_vol = np.sqrt(
                (team_margin_std.get(home, 12.0) ** 2 + team_margin_std.get(away, 12.0) ** 2) / 2.0
            )
            avg_3pa_rate = (team_3pa_rate.get(home, 0.39) + team_3pa_rate.get(away, 0.39)) / 2.0
            if avg_3pa_rate >= 0.41:
                comb_vol *= 1.12

            margin_sims = np.random.normal(exp_home_margin, comb_vol, N_SIMS).astype(np.float32)
            blowout_prob = float(np.mean(np.abs(margin_sims) >= 15.0))
            sneaky_blowout = (abs(exp_home_margin) <= 4.0) and (blowout_prob >= 0.32)

            for team, opp, is_home, t_pts, o_pts in [
                (home, away, True, comp_home, comp_away),
                (away, home, False, comp_away, comp_home),
            ]:
                last_dt = team_last_date.get(team)
                days_rest = int((target_dt - last_dt).days) if pd.notna(last_dt) else 2
                nba_matchups[team] = {
                    "opponent": opp,
                    "is_home": is_home,
                    "game_label": label,
                    "comp_total": float(comp_total),
                    "team_proj_pts": float(t_pts),
                    "proj_margin": float(t_pts - o_pts + (2.2 if is_home else -2.2)),
                    "pace_ratio": float(pace_ratio),
                    "days_rest": days_rest,
                    "is_b2b": days_rest == 1,
                    "blowout_prob": blowout_prob,
                    "sneaky_blowout": sneaky_blowout,
                }
    except Exception as e:
        print(f"[!] Could not fetch NBA schedule for {target_date_str}: {e}")

    def_lookup = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA", "PR", "PA", "RA"]:
        game_totals = (
            nba_df.groupby(["OPP_ABBR", "GAME_ID", "GAME_DATE"])[stat]
            .sum()
            .reset_index()
            .sort_values("GAME_DATE")
        )
        league_avg = game_totals.tail(400)[stat].mean()
        if pd.isna(league_avg) or league_avg == 0:
            continue
        for opp, grp in game_totals.groupby("OPP_ABBR"):
            rec_grp = grp.tail(20)
            w = np.exp(np.linspace(-1.5, 0.0, len(rec_grp)))
            w /= w.sum()
            def_lookup[(opp, stat)] = float(np.sum(rec_grp[stat].values * w)) / league_avg

    return {
        "curr_szn": curr_szn,
        "target_date": target_date_str,
        "nba_df": nba_df,
        "todays_games": todays_games,
        "matchup_map": nba_matchups,
        "def_lookup": def_lookup,
        "team_usage_bump": team_usage_bump,
    }


def simulate_nba_prop(ctx, p_logs, opponent_abbr, stat_col, prop_line=0.0, manual_bump=1.0):
    rec_logs = p_logs.sort_values("GAME_DATE").tail(22).copy()
    full_name = rec_logs["PLAYER_NAME"].iloc[-1]
    team_abbr = rec_logs["TEAM_ABBREVIATION"].iloc[-1]

    healthy = rec_logs[rec_logs["MIN"] >= 12.0].tail(15).copy()
    if len(healthy) < 4:
        healthy = rec_logs.tail(15).copy()

    h10 = healthy.tail(10)
    consistency = float(np.mean(h10[stat_col] >= (h10[stat_col].mean() * 0.85))) if not h10.empty else 0.5

    def_mult = ctx["def_lookup"].get((opponent_abbr, stat_col), 1.0)
    auto_bump = ctx["team_usage_bump"].get(team_abbr, 1.0) * manual_bump
    game_info = ctx["matchup_map"].get(team_abbr)

    ewm_min = healthy["MIN"].ewm(span=5).mean().iloc[-1]
    min_std = max(healthy["MIN"].std(), 1.5)

    rate_series = (healthy[stat_col] / healthy["MIN"].replace(0, np.nan)).dropna()
    ewm_rate = rate_series.ewm(span=5).mean().iloc[-1]
    rate_std = max(rate_series.std(), 0.02)

    flags = []
    blowout_prob = 0.25
    sneaky_blowout = False
    pace_mult = 1.0
    fatigue_mult = 1.0
    min_adj_mult = 1.0

    if game_info and game_info["opponent"] == opponent_abbr:
        comp_total = game_info["comp_total"]
        blowout_prob = game_info["blowout_prob"]
        sneaky_blowout = game_info["sneaky_blowout"]
        pace_mult = float(np.clip(game_info["pace_ratio"], 0.92, 1.10))

        split_logs = healthy[healthy["IS_HOME"] == game_info["is_home"]]
        if len(split_logs) >= 4:
            split_rate = (split_logs[stat_col] / split_logs["MIN"].replace(0, np.nan)).dropna().mean()
            ewm_rate = 0.75 * ewm_rate + 0.25 * split_rate

        if game_info["is_b2b"]:
            fatigue_mult = 0.95 if not game_info["is_home"] else 0.975
            flags.append(f"{'Road' if not game_info['is_home'] else 'Home'} B2B")

        if blowout_prob >= 0.34 and ewm_min >= 26.0:
            min_adj_mult = 0.92
            flags.append(
                f"{'SNEAKY ' if sneaky_blowout else ''}BLOWOUT RISK {blowout_prob:.0%} (-8% Min)"
            )

        if auto_bump > 1.0:
            flags.append(f"Teammate Out Bump {auto_bump:.2f}x")

        script_note = (
            f"Proj Total: {comp_total:.1f} | Pace: {pace_mult:.2f}x | Blowout Prob: {blowout_prob:.0%}"
        )
    else:
        comp_total = 228.0
        script_note = "Comp Game Total Est: 228.0 pts"

    adj_min = ewm_min * min_adj_mult
    adj_rate = ewm_rate * def_mult * pace_mult * fatigue_mult * auto_bump

    sim_min = np.clip(np.random.normal(adj_min, min_std, N_SIMS).astype(np.float32), 0, 48.0)
    sim_rate = np.clip(np.random.normal(adj_rate, rate_std, N_SIMS).astype(np.float32), 0, None)
    sims = sim_min * sim_rate

    over_prob = float(np.mean(sims > prop_line))
    fair_mult = round(1.0 / over_prob, 2) if over_prob > 0.01 else 99.0

    return {
        "sport": "NBA",
        "player": full_name,
        "position": "NBA",
        "team": team_abbr,
        "opponent": opponent_abbr,
        "game": game_info["game_label"] if game_info else f"{team_abbr} vs {opponent_abbr}",
        "comp_total": comp_total,
        "blowout_prob": blowout_prob,
        "sneaky_blowout": sneaky_blowout,
        "consistency": consistency,
        "flags": " | ".join(flags) if flags else "Clean Environment",
        "stat": stat_col,
        "line": prop_line,
        "proj": float(np.median(sims)),
        "floor": float(np.percentile(sims, 20)),
        "ceiling": float(np.percentile(sims, 80)),
        "over_prob": over_prob,
        "under_prob": float(np.mean(sims < prop_line)),
        "fair_mult": fair_mult,
        "def_mult": def_mult,
        "script_mult": float(pace_mult * fatigue_mult * auto_bump),
        "script_note": script_note,
        "breakdown": f"{adj_min:.1f} mins x {adj_rate:.2f} {stat_col}/min",
        "games_used": len(healthy),
    }


# =====================================================================
#        SHARED DISPLAY HELPER FOR 2x–3x VALUE + 100/500 REPORT
# =====================================================================
def display_2x_3x_report(top_plays, min_prob, max_prob, sport_label, df_source, sort_cols, player_col):
    if not top_plays:
        print(f"[!] No {sport_label} props fell between {min_prob:.0%} and {max_prob:.0%}. Try widening your % range.")
        return

    print("\n" + "=" * 86)
    print(f" TOP {sport_label} 2x–3x VALUE PLAYS ({min_prob:.0%}–{max_prob:.0%} HIT PROBABILITY)")
    print("=" * 86)

    for idx, play in enumerate(top_plays[:10], start=1):
        rep = generate_100_500_report(
            df_source, sort_cols, play["stat"], play["line"],
            entity_mask=(df_source[player_col] == play["player"]),
        )
        if sport_label == "NFL":
            def_rep = generate_100_500_report(
                df_source, sort_cols, play["stat"], play["line"],
                entity_mask=(df_source["position"] == play["position"]),
            )
        else:
            def_rep = generate_100_500_report(
                df_source, sort_cols, play["stat"], play["line"],
                entity_mask=(df_source["OPP_ABBR"] == play["opponent"]),
            )

        r100, r500 = rep[100], rep[500]
        d100, d500 = def_rep[100], def_rep[500]

        print(f"\n#{idx} [{sport_label}] {play['player']} ({play['team']}) OVER {play['line']} {play['stat']}")
        print(
            f"    Sim Hit Prob: {play['over_prob']:.1%}  |  Fair Payout: {play['fair_mult']:.2f}x  |  "
            f"Median Proj: {play['proj']:.1f} (80th Ceil: {play['ceiling']:.1f}) | Consistency: {play['consistency']:.0%}"
        )
        print(f"    Matchup: {play['game']}  |  {play['script_note']}")
        print(f"    Context Flags: {play['flags']}  |  Def Trend: {play['def_mult']:.2f}x | Form: {play['trend_ratio']:.2f}x")
        print(
            f"    Player Last {r100['n']} Games (2-Yr Cap): Raw {r100['raw_rate']:.1%} ({r100['hits']}/{r100['n']}) -> "
            f"Recency-Weighted: {r100['weighted_rate']:.1%}"
        )
        print(
            f"    Player Last {r500['n']} Games (2-Yr Cap): Raw {r500['raw_rate']:.1%} ({r500['hits']}/{r500['n']}) -> "
            f"Recency-Weighted: {r500['weighted_rate']:.1%}"
        )
        print(
            f"    League/Opp Sample (Last {d100['n']} / {d500['n']}): "
            f"Weighted Hit Rate {d100['weighted_rate']:.1%} (100g) / {d500['weighted_rate']:.1%} (500g)"
        )

    if len(top_plays) >= 2:
        print("\n" + "=" * 86)
        print(f" SUGGESTED {sport_label} 2-LEG & 3-LEG HIGH-MULTIPLIER PARLAYS")
        print("=" * 86)
        for n_legs in [2, 3]:
            if len(top_plays) < n_legs:
                continue
            best_combo, best_score = None, -1.0
            for combo in itertools.combinations(top_plays[:10], n_legs):
                if len({c["game"] for c in combo}) < n_legs:
                    continue
                score = sum(c["value_score"] for c in combo)
                if score > best_score:
                    best_score = score
                    best_combo = combo

            if best_combo:
                comb_prob = float(np.prod([c["over_prob"] for c in best_combo]))
                comb_payout = 1.0 / comb_prob if comb_prob > 0 else 0
                print(
                    f"\nBest {n_legs}-Leg {sport_label} Value Slip | Combined Prob: {comb_prob:.1%} "
                    f"(Est. Payout Multiplier: {comb_payout:.1f}x)"
                )
                for leg in best_combo:
                    print(
                        f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                        f"[{leg['over_prob']:.1%} | {leg['fair_mult']:.2f}x | {leg['game']}]"
                    )


# =====================================================================
#                         NFL INTERACTIVE MENU
# =====================================================================
def run_nfl_menu(nfl_ctx=None):
    if nfl_ctx is None:
        nfl_ctx = build_nfl_context()

    reg_stats = nfl_ctx["reg_stats"]
    team_col = nfl_ctx["team_col"]
    matchup_map = nfl_ctx["matchup_map"]
    current_season = nfl_ctx["season"]
    current_week = nfl_ctx["week"]
    out_players = nfl_ctx["out_players"]

    volume_map = {
        "passing_yards": ("attempts", 8),
        "rushing_yards": ("carries", 4),
        "receiving_yards": ("targets", 2),
        "receptions": ("targets", 2),
    }
    pos_props = {
        "QB": ["passing_yards"],
        "RB": ["rushing_yards", "receiving_yards"],
        "WR": ["receiving_yards", "receptions"],
        "TE": ["receiving_yards", "receptions"],
    }

    while True:
        print(f"\nNFL Simulator Menu (Season {current_season} | Week {current_week}):")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan This Week's Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] NFL DFS Lineup Projector & Optimizer ($50k Salary Cap)")
        print("  [4] NFL 2x–3x Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [5] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4/5]: ").strip()

        if choice == "1":
            print("\n--- NFL SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
            print("Stats: passing_yards, rushing_yards, receiving_yards, receptions, dk_points, passing_tds, rushing_tds, receiving_tds")
            player = input("\nPlayer Name (e.g., Josh Allen): ").strip()
            p_logs = reg_stats[reg_stats["player_display_name"].str.contains(player, case=False, na=False)]
            if p_logs.empty:
                print(f"[!] No NFL player found matching '{player}'.")
                continue

            player_team = p_logs.sort_values("sort_key")[team_col].iloc[-1]
            auto_opp = matchup_map[player_team]["opponent"] if player_team in matchup_map else ""
            opp = (
                input(f"Opponent Team Abbr [Press Enter for Week {current_week} opp: {auto_opp}]: ").strip().upper()
                or auto_opp
            )
            stat = input("Stat (e.g., passing_yards, rushing_yards, receiving_yards, dk_points): ").strip().lower()
            if stat not in reg_stats.columns:
                print(f"[!] Invalid stat '{stat}'.")
                continue
            try:
                line = float(input("Prop Line (e.g., 245.5): ").strip())
            except ValueError:
                print("[!] Invalid number for prop line.")
                continue

            star_out = input("Is a key teammate OUT tonight? [y/N]: ").strip().lower() == "y"
            manual_bump = 1.08 if star_out else 1.0

            t0 = time.time()
            res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat, line, manual_bump=manual_bump)
            elapsed = time.time() - t0

            print("\n" + "=" * 70)
            print(f" {res['player']} ({res['position']} - {res['team']}) vs. {res['opponent']} | {res['stat'].upper()} {res['line']}")
            print("=" * 70)
            print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
            print(f" Healthy Games:     {res['games_used']} | Consistency Score: {res['consistency']:.0%}")
            print(f" Matchup & Script:  {res['script_note']}")
            print(f" Context Flags:     {res['flags']}")
            print(f" Engine Breakdown:  {res['breakdown']} (Def: {res['def_mult']:.2f}x)")
            print("-" * 70)
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
            print("=" * 70)

        elif choice == "2":
            print(f"\n--- WEEK {current_week} NFL AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
            try:
                rate_in = float(input("Minimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
                min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
                parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
                discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
                skip_blowouts = input("Skip high Blowout-Risk games (including sneaky -2.5 traps)? [Y/n]: ").strip().lower() != "n"
            except ValueError:
                min_hit_rate, parlay_legs, discount, skip_blowouts = 0.75, 3, 0.85, True

            t0 = time.time()
            recent_pool = reg_stats[
                (reg_stats["season"] == current_season) & (reg_stats["week"] >= max(1, current_week - 3))
            ]
            active_players = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(
                subset=["player_id"]
            )

            qualifying_legs = []
            sims_count = 0
            for _, p in active_players.iterrows():
                p_name = str(p["player_display_name"]).strip()
                if p_name.lower() in out_players:
                    continue
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in pos_props:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.35:
                    continue

                opp = matchup_map[team]["opponent"]
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue
                for stat_col in pos_props[pos]:
                    vol_col, min_vol = volume_map[stat_col]
                    healthy = p_logs[p_logs[vol_col] >= min_vol].tail(10)
                    if len(healthy) < 4:
                        continue
                    raw_avg = healthy[stat_col].mean()
                    if (stat_col != "receptions" and raw_avg < 18) or (stat_col == "receptions" and raw_avg < 3.5):
                        continue
                    target_line = np.floor(raw_avg * discount) + 0.5
                    res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat_col, target_line)
                    sims_count += N_SIMS
                    if res["over_prob"] >= min_hit_rate:
                        qualifying_legs.append(res)

            elapsed = time.time() - t0
            qualifying_legs.sort(key=lambda x: (x["over_prob"] * (0.85 + 0.15 * x["consistency"])), reverse=True)
            print(f"\nCompleted {sims_count:,} sims in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
            for leg in qualifying_legs[:12]:
                print(
                    f"  [{leg['over_prob']:.2%} | Cons:{leg['consistency']:.0%}] {leg['player']} ({leg['team']}) "
                    f"OVER {leg['line']} {leg['stat']} (Proj: {leg['proj']:.1f} | {leg['game']} | {leg['flags']})"
                )

            if len(qualifying_legs) >= parlay_legs:
                print(f"\n================ TOP {parlay_legs}-LEG NFL PARLAYS (1 LEG PER GAME) ================")
                valid_parlays = []
                for combo in itertools.combinations(qualifying_legs[:14], parlay_legs):
                    if len({leg["game"] for leg in combo}) < parlay_legs:
                        continue
                    valid_parlays.append((float(np.prod([leg["over_prob"] for leg in combo])), combo))
                valid_parlays.sort(key=lambda x: x[0], reverse=True)
                for idx, (prob, combo) in enumerate(valid_parlays[:5], start=1):
                    fair_odds = int(-100 * prob / (1 - prob)) if prob > 0.5 else int(100 * (1 - prob) / prob)
                    print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.2%} (Fair Odds: {fair_odds:+d})")
                    for leg in combo:
                        print(
                            f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                            f"[{leg['over_prob']:.2%} | Proj: {leg['proj']:.1f} | {leg['game']}]"
                        )

        elif choice == "3":
            print(f"\n--- WEEK {current_week} NFL DFS LINEUP PROJECTOR & OPTIMIZER ($50,000 CAP) ---")
            is_gpp = input("Contest Mode — [1] Cash (Median) or [2] GPP (Ceiling + QB Stack): ").strip() == "2"

            dk_salaries = {}
            if os.path.exists("DKSalaries.csv"):
                dk_df = pd.read_csv("DKSalaries.csv")
                for _, row in dk_df.iterrows():
                    dk_salaries[str(row.get("Name", "")).strip().lower()] = int(row.get("Salary", 5000))

            t0 = time.time()
            recent_pool = reg_stats[
                (reg_stats["season"] == current_season) & (reg_stats["week"] >= max(1, current_week - 3))
            ]
            active_players = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(
                subset=["player_id"]
            )

            pool = []
            for _, p in active_players.iterrows():
                if str(p["player_display_name"]).strip().lower() in out_players:
                    continue
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in ["QB", "RB", "WR", "TE"]:
                    continue
                opp = matchup_map[team]["opponent"]
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 3 or p_logs["dk_points"].tail(6).mean() < 4.5:
                    continue

                res = simulate_nfl_prop(nfl_ctx, p_logs, opp, "dk_points", 15.0)
                name_lower = res["player"].lower()
                if name_lower in dk_salaries:
                    salary = dk_salaries[name_lower]
                else:
                    base_sal = {
                        "QB": 4600 + (res["proj"] * 175),
                        "RB": 4000 + (res["proj"] * 240),
                        "WR": 3500 + (res["proj"] * 235),
                        "TE": 2800 + (res["proj"] * 245),
                    }[pos]
                    salary = int(np.clip(round(base_sal / 100) * 100, 3000, 9500))

                pool.append(
                    {
                        "name": res["player"],
                        "pos": pos,
                        "team": team,
                        "opp": opp,
                        "salary": salary,
                        "proj": res["proj"],
                        "ceiling": res["ceiling"],
                        "opt_pts": res["ceiling"] if is_gpp else res["proj"],
                        "value": (res["proj"] / salary) * 1000,
                    }
                )

            for team, info in matchup_map.items():
                opp = info["opponent"]
                opp_implied = (info["total_line"] / 2.0) + (info["team_spread"] / 2.0)
                dst_proj = float(np.clip(13.5 - (opp_implied * 0.32), 3.0, 12.0))
                dst_ceil = dst_proj + 5.5
                dst_name = f"{team} DST"
                sal = dk_salaries.get(
                    dst_name.lower(), int(np.clip(round((2200 + dst_proj * 210) / 100) * 100, 2200, 4400))
                )
                pool.append(
                    {
                        "name": dst_name,
                        "pos": "DST",
                        "team": team,
                        "opp": opp,
                        "salary": sal,
                        "proj": dst_proj,
                        "ceiling": dst_ceil,
                        "opt_pts": dst_ceil if is_gpp else dst_proj,
                        "value": (dst_proj / sal) * 1000,
                    }
                )

            df_pool = pd.DataFrame(pool)
            n = len(df_pool)
            constraints = [
                LinearConstraint(df_pool["salary"].values, 0, 50000),
                LinearConstraint((df_pool["pos"] == "QB").astype(float).values, 1, 1),
                LinearConstraint((df_pool["pos"] == "DST").astype(float).values, 1, 1),
                LinearConstraint((df_pool["pos"] == "RB").astype(float).values, 2, 3),
                LinearConstraint((df_pool["pos"] == "WR").astype(float).values, 3, 4),
                LinearConstraint((df_pool["pos"] == "TE").astype(float).values, 1, 2),
                LinearConstraint(df_pool["pos"].isin(["RB", "WR", "TE"]).astype(float).values, 7, 7),
            ]
            if is_gpp:
                for t in df_pool["team"].unique():
                    qb_t = ((df_pool["team"] == t) & (df_pool["pos"] == "QB")).astype(float).values
                    pc_t = ((df_pool["team"] == t) & (df_pool["pos"].isin(["WR", "TE"]))).astype(float).values
                    constraints.append(LinearConstraint(pc_t - qb_t, 0, np.inf))

            res_opt = milp(c=-df_pool["opt_pts"].values, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=constraints)
            elapsed = time.time() - t0
            if not res_opt.success:
                print("[!] Could not find a valid NFL DFS lineup under $50,000.")
                continue

            chosen = df_pool[res_opt.x > 0.5].copy()
            chosen["order"] = chosen["pos"].map({"QB": 1, "RB": 2, "WR": 3, "TE": 4, "DST": 5})
            chosen = chosen.sort_values(["order", "proj"], ascending=[True, False])

            print("\n" + "=" * 78)
            mode_label = "GPP TOURNAMENT (CEILING + QB STACK)" if is_gpp else "CASH GAME (MEDIAN PROJECTION)"
            print(f" OPTIMAL WEEK {current_week} NFL DFS LINEUP | {mode_label} ({elapsed:.1f}s)")
            print("=" * 78)
            print(f" {'POS':<4} {'PLAYER':<22} {'MATCHUP':<11} {'SALARY':>7} {'MEDIAN':>7} {'CEIL':>7} {'VAL':>5}")
            print("-" * 78)
            counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0}
            limits = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
            for _, row in chosen.iterrows():
                p = row["pos"]
                counts[p] += 1
                slot = "FLEX" if counts[p] > limits[p] else p
                print(
                    f" {slot:<4} {row['name']:<22} {row['team'] + ' vs ' + row['opp']:<11} "
                    f"${row['salary']:>6,} {row['proj']:>7.1f} {row['ceiling']:>7.1f} {row['value']:>4.1f}x"
                )
            print("-" * 78)
            print(
                f" TOTAL SALARY: ${chosen['salary'].sum():,} / $50,000  |  "
                f"PROJ MEDIAN: {chosen['proj'].sum():.1f} pts  |  CEILING: {chosen['ceiling'].sum():.1f} pts"
            )
            print("=" * 78)

        elif choice == "4":
            print(f"\n--- WEEK {current_week} NFL 2x–3x VALUE FINDER (30%–50% HIT RATE + 100/500 REPORT) ---")
            try:
                min_pct = float(input("Minimum Hit Rate % (default 30 for 30%): ").strip() or "30")
                max_pct = float(input("Maximum Hit Rate % (default 50 for 50%): ").strip() or "50")
                skip_blowouts = input("Filter out high Blowout-Risk games? [Y/n]: ").strip().lower() != "n"
                min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
            except ValueError:
                min_prob, max_prob, skip_blowouts = 0.30, 0.50, True

            t0 = time.time()
            recent_pool = reg_stats[
                (reg_stats["season"] == current_season) & (reg_stats["week"] >= max(1, current_week - 3))
            ]
            active_nfl = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(
                subset=["player_id"]
            )

            nfl_props_ext = {
                "QB": [("passing_yards", "attempts", 10)],
                "RB": [("rushing_yards", "carries", 5), ("receiving_yards", "targets", 2)],
                "WR": [("receiving_yards", "targets", 3), ("receptions", "targets", 3)],
                "TE": [("receiving_yards", "targets", 2), ("receptions", "targets", 2)],
            }

            candidates = []
            sims_count = 0
            print("\nScanning Week matchups for 2x–3x alt-line upside plays...")

            for _, p in active_nfl.iterrows():
                if str(p["player_display_name"]).strip().lower() in out_players:
                    continue
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in nfl_props_ext:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.35:
                    continue

                opp = matchup_map[team]["opponent"]
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue

                for stat_col, vol_col, min_vol in nfl_props_ext[pos]:
                    healthy = p_logs[p_logs[vol_col] >= min_vol].tail(10)
                    if len(healthy) < 4:
                        continue

                    recent_trend_avg = healthy[stat_col].tail(4).mean()
                    two_yr_avg = healthy[stat_col].mean()
                    if (stat_col != "receptions" and recent_trend_avg < 20) or (
                        stat_col == "receptions" and recent_trend_avg < 3.5
                    ):
                        continue

                    for mult in [1.15, 1.28]:
                        alt_line = np.floor(recent_trend_avg * mult) + 0.5
                        res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat_col, alt_line)
                        sims_count += N_SIMS

                        if min_prob <= res["over_prob"] <= max_prob:
                            trend_boost = recent_trend_avg / max(two_yr_avg, 1.0)
                            res["trend_ratio"] = trend_boost
                            res["value_score"] = float(
                                res["over_prob"] * res["def_mult"] * res["script_mult"] * np.clip(trend_boost, 0.85, 1.30)
                            )
                            candidates.append(res)

            elapsed = time.time() - t0
            candidates.sort(key=lambda x: x["value_score"], reverse=True)
            seen, top_plays = set(), []
            for c in candidates:
                key = (c["player"], c["stat"])
                if key not in seen:
                    seen.add(key)
                    top_plays.append(c)

            print(f"\nCompleted {sims_count:,} NFL simulations in {elapsed:.1f}s!")
            display_2x_3x_report(
                top_plays, min_prob, max_prob, "NFL", reg_stats, ["season", "week"], "player_display_name"
            )

        elif choice == "5" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, 4, or 5.")

    return nfl_ctx


# =====================================================================
#                         NBA INTERACTIVE MENU
# =====================================================================
def run_nba_menu(nba_ctx=None):
    if nba_ctx is None:
        nba_ctx = build_nba_context()

    nba_df = nba_ctx["nba_df"]
    matchup_map = nba_ctx["matchup_map"]
    todays_games = nba_ctx["todays_games"]

    while True:
        print(f"\nNBA Simulator Menu (Season {nba_ctx['curr_szn']} | Date: {nba_ctx['target_date']} | Games: {len(todays_games)}):")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan Today's NBA Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] NBA 2x–3x Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [4] Change Slate Date (e.g., Tomorrow's Games)")
        print("  [5] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4/5]: ").strip()

        if choice == "1":
            print("\n--- NBA SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
            print("Stats: PTS, REB, AST, FG3M, STL, BLK, PRA, PR, PA, RA")
            player = input("\nPlayer Name (e.g., Nikola Jokic, Jayson Tatum): ").strip()
            p_logs = nba_df[nba_df["PLAYER_NAME"].str.contains(player, case=False, na=False)]
            if p_logs.empty:
                print(f"[!] No NBA player found matching '{player}'.")
                continue

            team_abbr = p_logs.sort_values("GAME_DATE")["TEAM_ABBREVIATION"].iloc[-1]
            auto_opp = matchup_map[team_abbr]["opponent"] if team_abbr in matchup_map else ""
            opp = (
                input(f"Opponent Team Abbr [Press Enter for {nba_ctx['target_date']} opp: {auto_opp}]: ").strip().upper()
                or auto_opp
            )
            stat = input("Stat (PTS, REB, AST, FG3M, PRA, PR, PA, RA): ").strip().upper()
            if stat not in nba_df.columns:
                print(f"[!] Invalid stat '{stat}'.")
                continue
            try:
                line = float(input("Prop Line (e.g., 26.5): ").strip())
            except ValueError:
                print("[!] Invalid number for prop line.")
                continue

            star_out = input("Is a star teammate OUT tonight? [y/N]: ").strip().lower() == "y"
            manual_bump = 1.07 if star_out else 1.0

            t0 = time.time()
            res = simulate_nba_prop(nba_ctx, p_logs, opp, stat, line, manual_bump=manual_bump)
            elapsed = time.time() - t0

            print("\n" + "=" * 70)
            print(f" {res['player']} ({res['team']}) vs. {res['opponent']} | {res['stat']} {res['line']}")
            print("=" * 70)
            print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
            print(f" Healthy Games:     {res['games_used']} | Consistency Score: {res['consistency']:.0%}")
            print(f" Matchup & Script:  {res['script_note']}")
            print(f" Context Flags:     {res['flags']}")
            print(f" Engine Breakdown:  {res['breakdown']} (Def: {res['def_mult']:.2f}x)")
            print("-" * 70)
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
            print("=" * 70)

        elif choice == "2":
            print(f"\n--- {nba_ctx['target_date']} NBA AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
            if not matchup_map:
                print("[!] No NBA games scheduled for this date.")
                continue
            print("Games on Slate:")
            for g in todays_games:
                home_t = g.split("@")[-1].strip()
                b_prob = matchup_map[home_t]["blowout_prob"]
                s_tag = " [!] SNEAKY CLOSE-SPREAD BLOWOUT RISK" if matchup_map[home_t]["sneaky_blowout"] else ""
                print(f"  • {g} (15+ pt Blowout Prob: {b_prob:.0%}{s_tag})")

            try:
                rate_in = float(input("\nMinimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
                min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
                parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
                discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
                skip_blowouts = input("Skip high Blowout-Risk games (including sneaky -2.5 traps)? [Y/n]: ").strip().lower() != "n"
            except ValueError:
                min_hit_rate, parlay_legs, discount, skip_blowouts = 0.75, 3, 0.85, True

            t0 = time.time()
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()

            qualifying_legs = []
            sims_count = 0
            for pid in active_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                team = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if team not in matchup_map or len(p_logs) < 5:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.34:
                    continue

                healthy = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(healthy) < 5 or healthy["MIN"].mean() < 20.0:
                    continue
                opp = matchup_map[team]["opponent"]
                for stat in ["PTS", "REB", "AST", "PRA"]:
                    raw_avg = healthy[stat].mean()
                    if (stat in ["REB", "AST"] and raw_avg < 4.5) or (stat == "PTS" and raw_avg < 10.0):
                        continue
                    target_line = np.floor(raw_avg * discount) + 0.5
                    res = simulate_nba_prop(nba_ctx, p_logs, opp, stat, target_line)
                    sims_count += N_SIMS
                    if res["over_prob"] >= min_hit_rate:
                        qualifying_legs.append(res)

            elapsed = time.time() - t0
            qualifying_legs.sort(key=lambda x: (x["over_prob"] * (0.85 + 0.15 * x["consistency"])), reverse=True)
            print(f"\nCompleted {sims_count:,} sims in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
            for leg in qualifying_legs[:12]:
                print(
                    f"  [{leg['over_prob']:.2%} | Cons:{leg['consistency']:.0%}] {leg['player']} ({leg['team']}) "
                    f"OVER {leg['line']} {leg['stat']} (Proj: {leg['proj']:.1f} | {leg['game']} | {leg['flags']})"
                )

            if len(qualifying_legs) >= parlay_legs:
                print(f"\n================ TOP {parlay_legs}-LEG NBA PARLAYS (1 LEG PER GAME) ================")
                valid_parlays = []
                for combo in itertools.combinations(qualifying_legs[:14], parlay_legs):
                    if len({leg["game"] for leg in combo}) < parlay_legs:
                        continue
                    valid_parlays.append((float(np.prod([leg["over_prob"] for leg in combo])), combo))
                valid_parlays.sort(key=lambda x: x[0], reverse=True)
                for idx, (prob, combo) in enumerate(valid_parlays[:5], start=1):
                    fair_odds = int(-100 * prob / (1 - prob)) if prob > 0.5 else int(100 * (1 - prob) / prob)
                    print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.2%} (Fair Odds: {fair_odds:+d})")
                    for leg in combo:
                        print(
                            f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                            f"[{leg['over_prob']:.2%} | Proj: {leg['proj']:.1f} | {leg['game']}]"
                        )

        elif choice == "3":
            print(f"\n--- {nba_ctx['target_date']} NBA 2x–3x VALUE FINDER (30%–50% HIT RATE + 100/500 REPORT) ---")
            if not matchup_map:
                print("[!] No NBA games scheduled for this date.")
                continue

            try:
                min_pct = float(input("Minimum Hit Rate % (default 30 for 30%): ").strip() or "30")
                max_pct = float(input("Maximum Hit Rate % (default 50 for 50%): ").strip() or "50")
                skip_blowouts = input("Filter out high Blowout-Risk games (including sneaky -2.5 traps)? [Y/n]: ").strip().lower() != "n"
                min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
            except ValueError:
                min_prob, max_prob, skip_blowouts = 0.30, 0.50, True

            t0 = time.time()
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_nba_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()

            candidates = []
            sims_count = 0
            print(f"\nScanning {len(todays_games)} NBA games for 2x–3x alt-line upside plays...")

            for pid in active_nba_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                team = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if team not in matchup_map or len(p_logs) < 5:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.34:
                    continue

                healthy = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(healthy) < 5 or healthy["MIN"].mean() < 22.0:
                    continue

                opp = matchup_map[team]["opponent"]
                for stat in ["PTS", "REB", "AST", "PRA"]:
                    recent_trend_avg = healthy[stat].tail(5).mean()
                    two_yr_avg = healthy[stat].mean()
                    if (stat in ["REB", "AST"] and recent_trend_avg < 4.5) or (stat == "PTS" and recent_trend_avg < 11.0):
                        continue

                    for mult in [1.12, 1.24]:
                        alt_line = np.floor(recent_trend_avg * mult) + 0.5
                        res = simulate_nba_prop(nba_ctx, p_logs, opp, stat, alt_line)
                        sims_count += N_SIMS

                        if min_prob <= res["over_prob"] <= max_prob:
                            trend_boost = recent_trend_avg / max(two_yr_avg, 1.0)
                            res["trend_ratio"] = trend_boost
                            res["value_score"] = float(
                                res["over_prob"] * res["def_mult"] * res["script_mult"] * np.clip(trend_boost, 0.85, 1.30)
                            )
                            candidates.append(res)

            elapsed = time.time() - t0
            candidates.sort(key=lambda x: x["value_score"], reverse=True)
            seen, top_plays = set(), []
            for c in candidates:
                key = (c["player"], c["stat"])
                if key not in seen:
                    seen.add(key)
                    top_plays.append(c)

            print(f"\nCompleted {sims_count:,} NBA simulations in {elapsed:.1f}s!")
            display_2x_3x_report(
                top_plays, min_prob, max_prob, "NBA", nba_df, ["GAME_DATE"], "PLAYER_NAME"
            )

        elif choice == "4":
            new_date = input("Enter NBA Slate Date (YYYY-MM-DD, e.g. 2026-11-19): ").strip()
            if new_date:
                nba_ctx = build_nba_context(new_date)
                nba_df = nba_ctx["nba_df"]
                matchup_map = nba_ctx["matchup_map"]
                todays_games = nba_ctx["todays_games"]

        elif choice == "5" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, 4, or 5.")

    return nba_ctx


# =====================================================================
#                        MAIN MULTI-SPORT LAUNCHER
# =====================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("   MULTI-SPORT 1,000,000-SIM PROP, PARLAY & VALUE ENGINE         ")
    print("=================================================================")
    cached_nfl = None
    cached_nba = None

    while True:
        print("\nSelect a Sport:")
        print("  [1] NFL (Props, Parlays, DFS Optimizer & 2x–3x Value Finder)")
        print("  [2] NBA (Props, Parlays & 2x–3x Value Finder)")
        print("  [3] College Sports (Men's CBB Props & College Football Spreads)")
        print("  [4] Quit")
        sport = input("Select [1/2/3/4]: ").strip()

        if sport == "1":
            cached_nfl = run_nfl_menu(cached_nfl)
        elif sport == "2":
            cached_nba = run_nba_menu(cached_nba)
        elif sport == "3":
            if os.path.exists("college_simulator.py"):
                subprocess.run([sys.executable, "college_simulator.py"])
            else:
                print("[!] college_simulator.py not found in this folder yet. Create it first!")
        elif sport == "4" or sport.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")
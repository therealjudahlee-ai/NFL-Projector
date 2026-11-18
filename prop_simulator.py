from datetime import datetime
import itertools
import os
import re
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


def _norm_name(name):
    """Normalizes player names so 'C.J. Stroud' matches 'CJ Stroud'."""
    s = re.sub(r"[^a-z0-9\s]", "", str(name).lower()).strip()
    for suffix in [" jr", " sr", " ii", " iii", " iv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


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
    print("\nLoading NFL 2-year data, live rosters, depth charts, weather & IR/injuries...")
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    seasons = [current_season - 1, current_season]

    player_stats = nfl.load_player_stats(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()
    reg_stats["sort_key"] = reg_stats["season"] * 100 + reg_stats["week"]
    team_col = "team" if "team" in reg_stats.columns else "recent_team"

    def compute_dfs_points(df):
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

        # DraftKings: Full PPR + 300/100 yd 3-pt bonuses
        dk = (
            (pass_yds * 0.04) + (pass_tds * 4.0) - (ints * 1.0)
            + (rush_yds * 0.1) + (rush_tds * 6.0)
            + (rec * 1.0) + (rec_yds * 0.1) + (rec_tds * 6.0)
            - (fumbles * 1.0)
        )
        dk += np.where(pass_yds >= 300, 3.0, 0.0)
        dk += np.where(rush_yds >= 100, 3.0, 0.0)
        dk += np.where(rec_yds >= 100, 3.0, 0.0)

        # FanDuel: Half PPR, no yardage bonuses
        fd = (
            (pass_yds * 0.04) + (pass_tds * 4.0) - (ints * 1.0)
            + (rush_yds * 0.1) + (rush_tds * 6.0)
            + (rec * 0.5) + (rec_yds * 0.1) + (rec_tds * 6.0)
            - (fumbles * 2.0)
        )
        return dk, fd

    reg_stats["dk_points"], reg_stats["fd_points"] = compute_dfs_points(reg_stats)

    # Load live current-week depth charts to verify true starting QBs (QB1)
    starting_qbs = set()
    try:
        dc_df = nfl.load_depth_charts([current_season]).to_pandas()
        if not dc_df.empty:
            if "dt" in dc_df.columns:
                dc_df = dc_df[dc_df["dt"] == dc_df["dt"].max()]
            elif "week" in dc_df.columns:
                dc_df = dc_df[dc_df["week"] == dc_df["week"].max()]
            rank_col = (
                "pos_rank"
                if "pos_rank" in dc_df.columns
                else ("depth_team" if "depth_team" in dc_df.columns else None)
            )
            pos_col = (
                "pos_abb"
                if "pos_abb" in dc_df.columns
                else ("position" if "position" in dc_df.columns else None)
            )
            name_dc = (
                "player_name"
                if "player_name" in dc_df.columns
                else ("full_name" if "full_name" in dc_df.columns else None)
            )
            if rank_col and pos_col and name_dc:
                qb1 = dc_df[
                    (dc_df[pos_col] == "QB")
                    & (pd.to_numeric(dc_df[rank_col], errors="coerce") == 1)
                ]
                starting_qbs = {_norm_name(x) for x in qb1[name_dc].dropna() if _norm_name(x)}
    except Exception:
        pass

    out_players = set()
    active_roster_team = {}

    # 1. Load nflreadpy weekly rosters & block IR / PUP / NFI / SUS
    try:
        rost_df = nfl.load_rosters_weekly([current_season]).to_pandas()
        if not rost_df.empty:
            rost_df = rost_df[rost_df["week"] == rost_df["week"].max()]
            r_name = "full_name" if "full_name" in rost_df.columns else "player_name"
            for _, r in rost_df.iterrows():
                nm = _norm_name(r.get(r_name, ""))
                if not nm:
                    continue
                st_val = str(r.get("status", "ACT")).strip().upper()
                if st_val in ["IR", "RES", "PUP", "NFI", "SUS", "CUT", "RET", "WAV", "INACTIVE", "OUT"]:
                    out_players.add(nm)
                elif st_val in ["ACT", "ACTIVE", ""]:
                    active_roster_team[nm] = str(r.get("team", "")).strip()
    except Exception:
        pass

    # 2. Load nflreadpy weekly injury report
    team_pos_bump = {}
    try:
        inj_df = nfl.load_injuries([current_season]).to_pandas()
        wk_inj = inj_df[inj_df["week"] == current_week]
        if not wk_inj.empty and "report_status" in wk_inj.columns:
            bad = wk_inj[wk_inj["report_status"].isin(["Out", "Doubtful"])]
            name_c = "full_name" if "full_name" in bad.columns else "player_name"
            for _, r in bad.iterrows():
                p_name = _norm_name(r.get(name_c, ""))
                if p_name:
                    out_players.add(p_name)
                t_abbr = str(r.get("team", "")).strip()
                pos_str = str(r.get("position", "")).strip()
                hist = reg_stats[
                    (reg_stats["player_display_name"].apply(_norm_name) == p_name)
                    & (reg_stats["season"] == current_season)
                ]
                if not hist.empty and hist["dk_points"].mean() >= 10.0:
                    team_pos_bump[(t_abbr, pos_str)] = min(
                        1.14, team_pos_bump.get((t_abbr, pos_str), 1.0) + 0.07
                    )
    except Exception:
        pass

    # 3. Pull LIVE ESPN NFL injuries & 32-team active rosters (catches IR like Malik Willis immediately)
    espn_abbr_map = {"WSH": "WAS", "LAR": "LA"}
    try:
        espn_inj = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", timeout=8
        ).json()
        for team_blk in espn_inj.get("injuries", []):
            for item in team_blk.get("injuries", []):
                st_txt = str(item.get("status", "")).lower()
                type_desc = str(item.get("type", {}).get("description", "")).lower()
                if any(k in st_txt or k in type_desc for k in ["out", "doubtful", "injured reserve", "ir", "pup", "suspension"]):
                    ath = item.get("athlete", {})
                    nm_norm = _norm_name(ath.get("displayName", ""))
                    if nm_norm:
                        out_players.add(nm_norm)
    except Exception:
        pass

    try:
        teams_resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams?limit=40", timeout=8
        ).json()
        for sp in teams_resp.get("sports", []):
            for lg in sp.get("leagues", []):
                for t_obj in lg.get("teams", []):
                    t_info = t_obj.get("team", {})
                    t_id = t_info.get("id")
                    t_abbr = espn_abbr_map.get(t_info.get("abbreviation", ""), t_info.get("abbreviation", ""))
                    if not t_id:
                        continue
                    r_resp = requests.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{t_id}/roster",
                        timeout=6,
                    ).json()
                    for grp in r_resp.get("athletes", []):
                        for ath in grp.get("items", []):
                            nm_norm = _norm_name(ath.get("displayName", ""))
                            if not nm_norm:
                                continue
                            st_type = str(ath.get("status", {}).get("type", "active")).lower()
                            if st_type != "active":
                                out_players.add(nm_norm)
                                active_roster_team.pop(nm_norm, None)
                            else:
                                active_roster_team[nm_norm] = t_abbr
    except Exception:
        pass

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

    def_lookup = {}
    all_stats = [
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "passing_tds", "rushing_tds", "receiving_tds", "dk_points", "fd_points",
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
        "starting_qbs": starting_qbs,
        "active_roster_team": active_roster_team,
    }


def simulate_nfl_prop(ctx, p_logs, opponent_team, stat_col, prop_line=0.0, manual_bump=1.0, override_team=None):
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
    player_team = override_team or recent[team_col].iloc[-1]

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
            flags.append(f"{'SNEAKY ' if sneaky_blowout else ''}BLOWOUT RISK {blowout_prob:.0%}")

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
#             ADVANCED NFL DFS POOL & MULTI-LINEUP SOLVER
# =====================================================================
def build_nfl_dfs_pool(nfl_ctx, site="DK", custom_csv_df=None):
    """Builds a projected NFL DFS player pool with live ESPN IR/Roster filtering & 1-QB-per-team."""
    reg_stats = nfl_ctx["reg_stats"]
    team_col = nfl_ctx["team_col"]
    matchup_map = nfl_ctx["matchup_map"]
    out_players = nfl_ctx["out_players"]
    starting_qbs = nfl_ctx.get("starting_qbs", set())
    active_roster_team = nfl_ctx.get("active_roster_team", {})
    stat_col = "dk_points" if site == "DK" else "fd_points"

    csv_salaries = {}
    csv_ids = {}
    if custom_csv_df is not None and not custom_csv_df.empty:
        for _, row in custom_csv_df.iterrows():
            nm = str(row.get("Name", row.get("Nickname", ""))).strip()
            if nm:
                norm_nm = _norm_name(nm)
                csv_salaries[norm_nm] = int(pd.to_numeric(row.get("Salary", 5000), errors="coerce") or 5000)
                if "Name + ID" in row:
                    csv_ids[norm_nm] = str(row["Name + ID"])
                elif "Id" in row:
                    csv_ids[norm_nm] = f"{nm} ({row['Id']})"

    recent_pool = reg_stats[
        (reg_stats["season"] == nfl_ctx["season"]) & (reg_stats["week"] >= max(1, nfl_ctx["week"] - 4))
    ]
    active_players = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(
        subset=["player_id"]
    )

    pool = []
    for _, p in active_players.iterrows():
        p_name_raw = str(p["player_display_name"]).strip()
        p_norm = _norm_name(p_name_raw)
        if p_norm in out_players:
            continue

        # Sync to live active roster team if available (blocks IR/PUP/inactive players)
        if active_roster_team:
            if p_norm not in active_roster_team:
                continue
            team = active_roster_team[p_norm]
        else:
            team = p[team_col]

        pos = p["position"]
        if team not in matchup_map or pos not in ["QB", "RB", "WR", "TE"]:
            continue

        p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
        if len(p_logs) < 3:
            continue

        # Require starter-level passing volume (20+ pass att/game) AND activity last week
        if pos == "QB":
            recent_att = p_logs["attempts"].tail(4).mean() if "attempts" in p_logs.columns else 0
            szn_logs = p_logs[p_logs["season"] == nfl_ctx["season"]]
            last_wk = int(szn_logs["week"].max()) if not szn_logs.empty else 0
            if recent_att < 20.0 or last_wk < (nfl_ctx["week"] - 1):
                continue
        else:
            if p_logs[stat_col].tail(6).mean() < 5.5:
                continue

        opp = matchup_map[team]["opponent"]
        res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat_col, 15.0, override_team=team)

        if csv_salaries and p_norm not in csv_salaries:
            continue

        if p_norm in csv_salaries:
            salary = csv_salaries[p_norm]
        else:
            if site == "DK":
                base_sal = {
                    "QB": 4800 + (res["proj"] * 170),
                    "RB": 4200 + (res["proj"] * 235),
                    "WR": 3600 + (res["proj"] * 230),
                    "TE": 3000 + (res["proj"] * 240),
                }[pos]
                salary = int(np.clip(round(base_sal / 100) * 100, 3200, 9500))
            else:
                base_sal = {
                    "QB": 6200 + (res["proj"] * 180),
                    "RB": 5200 + (res["proj"] * 255),
                    "WR": 4900 + (res["proj"] * 250),
                    "TE": 4300 + (res["proj"] * 255),
                }[pos]
                salary = int(np.clip(round(base_sal / 100) * 100, 4500, 10500))

        val = (res["proj"] / salary) * 1000.0
        team_pts = matchup_map[team]["team_proj_pts"]
        est_own = float(np.clip(((val - 1.6) * 8.5) + ((team_pts - 21.0) * 1.1), 1.5, 38.0))
        leverage_pts = float(res["ceiling"] * (1.0 - (est_own / 100.0) * 0.45))

        pool.append(
            {
                "name": res["player"],
                "dk_id": csv_ids.get(p_norm, res["player"]),
                "pos": pos,
                "team": team,
                "opp": opp,
                "game": res["game"],
                "salary": salary,
                "proj": round(res["proj"], 2),
                "ceiling": round(res["ceiling"], 2),
                "est_own": round(est_own, 1),
                "leverage": round(leverage_pts, 2),
                "value": round(val, 2),
                "flags": res["flags"],
                "_is_dc_qb1": 1 if (pos == "QB" and p_norm in starting_qbs) else 0,
            }
        )

    # Strictly keep ONLY 1 true starting QB per NFL team
    df_temp = pd.DataFrame(pool)
    if not df_temp.empty and "QB" in df_temp["pos"].values:
        qbs = df_temp[df_temp["pos"] == "QB"].sort_values(["_is_dc_qb1", "proj"], ascending=[False, False])
        qbs = qbs.drop_duplicates(subset=["team"], keep="first")
        non_qbs = df_temp[df_temp["pos"] != "QB"]
        df_temp = pd.concat([qbs, non_qbs], ignore_index=True)

    if not df_temp.empty and "_is_dc_qb1" in df_temp.columns:
        df_temp = df_temp.drop(columns=["_is_dc_qb1"])
    pool = df_temp.to_dict("records") if not df_temp.empty else []

    for team, info in matchup_map.items():
        opp = info["opponent"]
        opp_implied = (info["total_line"] / 2.0) + (info["team_spread"] / 2.0)
        dst_proj = float(np.clip(13.5 - (opp_implied * 0.32), 3.0, 12.0))
        dst_ceil = dst_proj + 5.5
        dst_name = f"{team} DST"
        dst_norm = _norm_name(dst_name)
        t_norm = _norm_name(team)
        if csv_salaries and dst_norm not in csv_salaries and t_norm not in csv_salaries:
            continue
        sal = csv_salaries.get(
            dst_norm,
            csv_salaries.get(
                t_norm,
                int(np.clip(round(((2200 if site == "DK" else 3200) + dst_proj * 210) / 100) * 100, 2200, 5200)),
            ),
        )
        val = (dst_proj / sal) * 1000.0
        est_own = float(np.clip((val - 1.5) * 6.5, 2.0, 24.0))
        pool.append(
            {
                "name": dst_name,
                "dk_id": csv_ids.get(dst_norm, dst_name),
                "pos": "DST",
                "team": team,
                "opp": opp,
                "game": info["game_label"],
                "salary": sal,
                "proj": round(dst_proj, 2),
                "ceiling": round(dst_ceil, 2),
                "est_own": round(est_own, 1),
                "leverage": round(dst_ceil * (1.0 - (est_own / 100.0) * 0.45), 2),
                "value": round(val, 2),
                "flags": f"Opp Implied: {opp_implied:.1f}",
            }
        )

    return pd.DataFrame(pool)


def optimize_nfl_dfs_lineups(
    df_pool,
    site="DK",
    mode="gpp",
    num_lineups=1,
    min_unique=2,
    max_exposure=0.65,
    stack_type="qb_1_bringback",
    no_opp_vs_dst=True,
    max_1_rb_per_team=True,
    locked_names=None,
    excluded_names=None,
):
    """Solves 1 to 20 optimal NFL DFS lineups with stacking, bring-backs, anti-cannibalization & exposure."""
    if df_pool.empty:
        return []

    locked_names = {_norm_name(x) for x in (locked_names or [])}
    excluded_names = {_norm_name(x) for x in (excluded_names or [])}

    salary_cap = 50000 if site == "DK" else 60000
    score_col = {"cash": "proj", "gpp": "ceiling", "leverage": "leverage"}[mode]

    n = len(df_pool)
    base_constraints = [
        LinearConstraint(df_pool["salary"].values, 0, salary_cap),
        LinearConstraint((df_pool["pos"] == "QB").astype(float).values, 1, 1),
        LinearConstraint((df_pool["pos"] == "DST").astype(float).values, 1, 1),
        LinearConstraint((df_pool["pos"] == "RB").astype(float).values, 2, 3),
        LinearConstraint((df_pool["pos"] == "WR").astype(float).values, 3, 4),
        LinearConstraint((df_pool["pos"] == "TE").astype(float).values, 1, 2),
        LinearConstraint(df_pool["pos"].isin(["RB", "WR", "TE"]).astype(float).values, 7, 7),
    ]

    lb = np.zeros(n)
    ub = np.ones(n)
    for idx, row in df_pool.iterrows():
        nm = _norm_name(row["name"])
        if nm in excluded_names:
            ub[idx] = 0.0
        elif nm in locked_names:
            lb[idx] = 1.0

    teams = df_pool["team"].unique()
    for t in teams:
        qb_t = ((df_pool["team"] == t) & (df_pool["pos"] == "QB")).astype(float).values
        pc_t = ((df_pool["team"] == t) & (df_pool["pos"].isin(["WR", "TE"]))).astype(float).values
        opp_teams = df_pool[df_pool["team"] == t]["opp"].unique()
        opp_skill = ((df_pool["team"].isin(opp_teams)) & (df_pool["pos"].isin(["WR", "TE", "RB"]))).astype(float).values

        if stack_type in ["qb_1", "qb_1_bringback"]:
            base_constraints.append(LinearConstraint(pc_t - qb_t, 0, np.inf))
        elif stack_type in ["qb_2", "qb_2_bringback"]:
            base_constraints.append(LinearConstraint(pc_t - (2.0 * qb_t), 0, np.inf))

        if "bringback" in stack_type:
            base_constraints.append(LinearConstraint(opp_skill - qb_t, 0, np.inf))

        if no_opp_vs_dst:
            dst_t = ((df_pool["team"] == t) & (df_pool["pos"] == "DST")).astype(float).values
            if dst_t.sum() > 0 and opp_skill.sum() > 0:
                base_constraints.append(LinearConstraint(opp_skill + (7.0 * dst_t), 0, 7.0))

        if max_1_rb_per_team:
            rb_t = ((df_pool["team"] == t) & (df_pool["pos"] == "RB")).astype(float).values
            if rb_t.sum() > 1:
                base_constraints.append(LinearConstraint(rb_t, 0, 1))

    lineups = []
    appearance_counts = np.zeros(n)
    max_appearances = max(1, int(np.ceil(num_lineups * max_exposure)))

    for lu_idx in range(num_lineups):
        cur_ub = ub.copy()
        if lu_idx > 0:
            for i in range(n):
                if lb[i] < 0.5 and appearance_counts[i] >= max_appearances:
                    cur_ub[i] = 0.0

        res_opt = milp(
            c=-df_pool[score_col].values,
            integrality=np.ones(n),
            bounds=Bounds(lb, cur_ub),
            constraints=base_constraints,
        )
        if not res_opt.success:
            break

        sel_mask = res_opt.x > 0.5
        appearance_counts[sel_mask] += 1

        chosen = df_pool[sel_mask].copy()
        chosen["order"] = chosen["pos"].map({"QB": 1, "RB": 2, "WR": 3, "TE": 4, "DST": 5})
        chosen = chosen.sort_values(["order", "proj"], ascending=[True, False]).reset_index(drop=True)

        counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0}
        limits = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
        slots = []
        for _, r in chosen.iterrows():
            p = r["pos"]
            counts[p] += 1
            slots.append("FLEX" if counts[p] > limits[p] else p)
        chosen["slot"] = slots
        lineups.append(chosen)

        prev_vec = sel_mask.astype(float)
        base_constraints.append(LinearConstraint(prev_vec, 0, 9 - min_unique))

    return lineups


# =====================================================================
#                        NBA DATA & ENGINE LOADER
# =====================================================================
def _ensure_nba_dk_pts(df):
    """Computes DraftKings NBA fantasy points on the fly if missing from cached DataFrames."""
    if "DK_PTS" in df.columns:
        return df
    df = df.copy()
    pts = df["PTS"] if "PTS" in df.columns else 0
    reb = df["REB"] if "REB" in df.columns else 0
    ast = df["AST"] if "AST" in df.columns else 0
    fg3m = df["FG3M"] if "FG3M" in df.columns else 0
    stl = df["STL"] if "STL" in df.columns else 0
    blk = df["BLK"] if "BLK" in df.columns else 0
    tov = df["TOV"] if "TOV" in df.columns else 0

    dd_cats = (
        (pts >= 10).astype(int)
        + (reb >= 10).astype(int)
        + (ast >= 10).astype(int)
        + (stl >= 10).astype(int)
        + (blk >= 10).astype(int)
    )
    df["DK_PTS"] = (
        pts * 1.0
        + fg3m * 0.5
        + reb * 1.25
        + ast * 1.5
        + stl * 2.0
        + blk * 2.0
        - tov * 0.5
        + np.where(dd_cats >= 2, 1.5, 0.0)
        + np.where(dd_cats >= 3, 3.0, 0.0)
    )
    return df


def build_nba_context(target_date_str=None):
    if not target_date_str:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\nLoading NBA 2-year data, Pace/Volatility & {target_date_str} slate...")
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    curr_szn = f"{start_year}-{str(start_year + 1)[-2:]}"
    prev_szn = f"{start_year - 1}-{str(start_year)[-2:]}"

    dfs = []
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
                    "TOV": pd.to_numeric(raw["turnovers"], errors="coerce").fillna(0),
                    "FGA": pd.to_numeric(raw["field_goals_attempted"], errors="coerce").fillna(0),
                    "FTA": pd.to_numeric(raw["free_throws_attempted"], errors="coerce").fillna(0),
                    "OREB": pd.to_numeric(raw["offensive_rebounds"], errors="coerce").fillna(0),
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

    abbr_fix = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS", "UTAH": "UTA"}
    nba_df["TEAM_ABBREVIATION"] = nba_df["TEAM_ABBREVIATION"].replace(abbr_fix)
    nba_df["OPP_ABBR"] = nba_df["OPP_ABBR"].replace(abbr_fix)

    nba_df["PRA"] = nba_df["PTS"] + nba_df["REB"] + nba_df["AST"]
    nba_df["PR"] = nba_df["PTS"] + nba_df["REB"]
    nba_df["PA"] = nba_df["PTS"] + nba_df["AST"]
    nba_df["RA"] = nba_df["REB"] + nba_df["AST"]
    nba_df = _ensure_nba_dk_pts(nba_df)

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
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA", "PR", "PA", "RA", "DK_PTS"]:
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
    if stat_col == "DK_PTS" and "DK_PTS" not in p_logs.columns:
        p_logs = _ensure_nba_dk_pts(p_logs)

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
            flags.append(f"{'SNEAKY ' if sneaky_blowout else ''}BLOWOUT RISK {blowout_prob:.0%} (-8% Min)")

        if auto_bump > 1.0:
            flags.append(f"Teammate Out Bump {auto_bump:.2f}x")

        script_note = f"Proj Total: {comp_total:.1f} | Pace: {pace_mult:.2f}x | Blowout Prob: {blowout_prob:.0%}"
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


def build_nba_dfs_pool(nba_ctx, custom_csv_df=None):
    """Builds projected DraftKings NBA DFS pool (8-man roster under $50,000)."""
    nba_df = _ensure_nba_dk_pts(nba_ctx["nba_df"])
    matchup_map = nba_ctx["matchup_map"]

    csv_salaries = {}
    if custom_csv_df is not None and not custom_csv_df.empty:
        for _, row in custom_csv_df.iterrows():
            nm = str(row.get("Name", "")).strip()
            if nm:
                csv_salaries[_norm_name(nm)] = int(pd.to_numeric(row.get("Salary", 5000), errors="coerce") or 5000)

    cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
    active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()
    pool = []

    for pid in active_ids:
        p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
        t = p_logs["TEAM_ABBREVIATION"].iloc[-1]
        if t not in matchup_map or len(p_logs) < 4:
            continue
        h = p_logs[p_logs["MIN"] >= 12.0].tail(12)
        if len(h) < 4 or h["MIN"].mean() < 16.0:
            continue

        res = simulate_nba_prop(nba_ctx, p_logs, matchup_map[t]["opponent"], "DK_PTS", 25.0)
        nm_norm = _norm_name(res["player"])
        if csv_salaries and nm_norm not in csv_salaries:
            continue

        sal = csv_salaries.get(nm_norm, int(np.clip(round((3000 + res["proj"] * 175) / 100) * 100, 3000, 11800)))
        val = (res["proj"] / sal) * 1000.0
        est_own = float(np.clip((val - 4.2) * 9.0, 2.0, 42.0))
        pool.append({
            "name": res["player"],
            "team": t,
            "opp": matchup_map[t]["opponent"],
            "game": res["game"],
            "salary": sal,
            "proj": round(res["proj"], 2),
            "ceiling": round(res["ceiling"], 2),
            "est_own": round(est_own, 1),
            "leverage": round(res["ceiling"] * (1.0 - (est_own / 100.0) * 0.45), 2),
            "value": round(val, 2),
            "flags": res["flags"],
        })
    return pd.DataFrame(pool)


def optimize_nba_dfs_lineups(df_pool, mode="gpp", num_lineups=1, min_unique=2, max_exposure=0.65, locked_names=None, excluded_names=None):
    """Solves 1 to 20 optimal 8-player NBA DFS lineups under $50,000."""
    if df_pool.empty:
        return []
    locked_names = {_norm_name(x) for x in (locked_names or [])}
    excluded_names = {_norm_name(x) for x in (excluded_names or [])}
    score_col = {"cash": "proj", "gpp": "ceiling", "leverage": "leverage"}[mode]

    n = len(df_pool)
    constraints = [
        LinearConstraint(df_pool["salary"].values, 0, 50000),
        LinearConstraint(np.ones(n), 8, 8),
    ]
    for g in df_pool["game"].unique():
        g_vec = (df_pool["game"] == g).astype(float).values
        constraints.append(LinearConstraint(g_vec, 0, 4))

    lb, ub = np.zeros(n), np.ones(n)
    for i, r in df_pool.iterrows():
        nm = _norm_name(r["name"])
        if nm in excluded_names:
            ub[i] = 0.0
        elif nm in locked_names:
            lb[i] = 1.0

    lineups = []
    counts = np.zeros(n)
    max_app = max(1, int(np.ceil(num_lineups * max_exposure)))

    for lu_i in range(num_lineups):
        cur_ub = ub.copy()
        if lu_i > 0:
            for i in range(n):
                if lb[i] < 0.5 and counts[i] >= max_app:
                    cur_ub[i] = 0.0
        res = milp(c=-df_pool[score_col].values, integrality=np.ones(n), bounds=Bounds(lb, cur_ub), constraints=constraints)
        if not res.success:
            break
        sel = res.x > 0.5
        counts[sel] += 1
        chosen = df_pool[sel].sort_values("proj", ascending=False).reset_index(drop=True)
        chosen["slot"] = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"][:len(chosen)]
        lineups.append(chosen)
        constraints.append(LinearConstraint(sel.astype(float), 0, 8 - min_unique))

    return lineups


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
        print("  [3] Pro NFL DFS Lineup Optimizer (Stacking + Bring-Back + Leverage + Multi-Lineup)")
        print("  [4] NFL 2x–3x Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [5] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4/5]: ").strip()

        if choice == "1":
            print("\n--- NFL SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
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
            stat = input("Stat (passing_yards, rushing_yards, receiving_yards, receptions, dk_points): ").strip().lower()
            if stat not in reg_stats.columns:
                print(f"[!] Invalid stat '{stat}'.")
                continue
            try:
                line = float(input("Prop Line (e.g., 245.5): ").strip())
            except ValueError:
                print("[!] Invalid number for prop line.")
                continue

            star_out = input("Is a key teammate OUT tonight? [y/N]: ").strip().lower() == "y"
            res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat, line, manual_bump=(1.08 if star_out else 1.0))

            print("\n" + "=" * 70)
            print(f" {res['player']} ({res['position']} - {res['team']}) vs. {res['opponent']} | {res['stat'].upper()} {res['line']}")
            print("=" * 70)
            print(f" Healthy Games:     {res['games_used']} | Consistency Score: {res['consistency']:.0%}")
            print(f" Matchup & Script:  {res['script_note']}")
            print(f" Context Flags:     {res['flags']}")
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print("=" * 70)

        elif choice == "2":
            print(f"\n--- WEEK {current_week} NFL AUTOMATED PARLAY BUILDER ---")
            try:
                rate_in = float(input("Minimum Hit Rate % per leg (e.g., 75): ").strip() or "75")
                min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
                parlay_legs = int(input("Legs per parlay (e.g., 3): ").strip() or "3")
                discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = alt floor): ").strip() or "0.85")
                skip_blowouts = input("Skip high Blowout-Risk games? [Y/n]: ").strip().lower() != "n"
            except ValueError:
                min_hit_rate, parlay_legs, discount, skip_blowouts = 0.75, 3, 0.85, True

            recent_pool = reg_stats[
                (reg_stats["season"] == current_season) & (reg_stats["week"] >= max(1, current_week - 3))
            ]
            active_players = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(
                subset=["player_id"]
            )

            qualifying_legs = []
            for _, p in active_players.iterrows():
                if _norm_name(p["player_display_name"]) in out_players:
                    continue
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in pos_props:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.35:
                    continue
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
                    res = simulate_nfl_prop(nfl_ctx, p_logs, matchup_map[team]["opponent"], stat_col, np.floor(raw_avg * discount) + 0.5)
                    if res["over_prob"] >= min_hit_rate:
                        qualifying_legs.append(res)

            qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)
            for leg in qualifying_legs[:12]:
                print(
                    f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                    f"(Proj: {leg['proj']:.1f} | {leg['game']} | {leg['flags']})"
                )

        elif choice == "3":
            print(f"\n--- PRO WEEK {current_week} NFL DFS OPTIMIZER ---")
            site = "FD" if input("Site — [1] DraftKings ($50k) or [2] FanDuel ($60k): ").strip() == "2" else "DK"
            mode_in = input("Contest Mode — [1] Cash (Median), [2] GPP (Ceiling), [3] Contrarian Leverage: ").strip()
            mode = {"1": "cash", "2": "gpp", "3": "leverage"}.get(mode_in, "gpp")
            num_lu = int(input("How many lineups to generate (1–20, default 3): ").strip() or "3")
            stack_in = input("Stacking — [1] QB+1+BringBack, [2] QB+2+BringBack, [3] QB+1 Only: ").strip()
            stack_type = {"1": "qb_1_bringback", "2": "qb_2_bringback", "3": "qb_1"}.get(stack_in, "qb_1_bringback")
            lock_str = input("Lock players (comma-separated, or Enter for none): ").strip()
            excl_str = input("Exclude players (comma-separated, or Enter for none): ").strip()

            locked = [x.strip() for x in lock_str.split(",") if x.strip()]
            excluded = [x.strip() for x in excl_str.split(",") if x.strip()]

            custom_df = pd.read_csv("DKSalaries.csv") if os.path.exists("DKSalaries.csv") else None
            df_pool = build_nfl_dfs_pool(nfl_ctx, site=site, custom_csv_df=custom_df)
            lineups = optimize_nfl_dfs_lineups(
                df_pool, site=site, mode=mode, num_lineups=num_lu, stack_type=stack_type, locked_names=locked, excluded_names=excluded
            )

            cap = 50000 if site == "DK" else 60000
            for idx, lu in enumerate(lineups, start=1):
                print("\n" + "=" * 86)
                print(f" OPTIMAL {site} LINEUP #{idx} | MODE: {mode.upper()} | STACK: {stack_type.upper()}")
                print("=" * 86)
                print(f" {'SLOT':<5} {'PLAYER':<22} {'GAME':<12} {'SALARY':>7} {'MEDIAN':>7} {'CEIL':>7} {'OWN%':>6}")
                print("-" * 86)
                for _, r in lu.iterrows():
                    print(
                        f" {r['slot']:<5} {r['name']:<22} {r['game']:<12} "
                        f"${r['salary']:>6,} {r['proj']:>7.1f} {r['ceiling']:>7.1f} {r['est_own']:>5.1f}%"
                    )
                print("-" * 86)
                print(
                    f" SALARY: ${lu['salary'].sum():,} / ${cap:,}  |  PROJ: {lu['proj'].sum():.1f}  |  "
                    f"CEIL: {lu['ceiling'].sum():.1f}  |  CUMULATIVE OWN: {lu['est_own'].sum():.1f}%"
                )

        elif choice == "4":
            print(f"\n--- WEEK {current_week} NFL 2x–3x VALUE FINDER ---")
            min_prob, max_prob = 0.30, 0.50
            recent_pool = reg_stats[
                (reg_stats["season"] == current_season) & (reg_stats["week"] >= max(1, current_week - 3))
            ]
            active_nfl = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            nfl_props_ext = {
                "QB": [("passing_yards", "attempts", 10)],
                "RB": [("rushing_yards", "carries", 5), ("receiving_yards", "targets", 2)],
                "WR": [("receiving_yards", "targets", 3), ("receptions", "targets", 3)],
                "TE": [("receiving_yards", "targets", 2), ("receptions", "targets", 2)],
            }
            candidates = []
            for _, p in active_nfl.iterrows():
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in nfl_props_ext:
                    continue
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue
                for stat_col, vol_col, min_vol in nfl_props_ext[pos]:
                    healthy = p_logs[p_logs[vol_col] >= min_vol].tail(10)
                    if len(healthy) < 4:
                        continue
                    rec_avg = healthy[stat_col].tail(4).mean()
                    two_yr_avg = healthy[stat_col].mean()
                    if (stat_col != "receptions" and rec_avg < 20) or (stat_col == "receptions" and rec_avg < 3.5):
                        continue
                    for mult in [1.15, 1.28]:
                        alt_line = np.floor(rec_avg * mult) + 0.5
                        res = simulate_nfl_prop(nfl_ctx, p_logs, matchup_map[team]["opponent"], stat_col, alt_line)
                        if min_prob <= res["over_prob"] <= max_prob:
                            res["trend_ratio"] = rec_avg / max(two_yr_avg, 1.0)
                            res["value_score"] = float(res["over_prob"] * res["def_mult"] * res["script_mult"])
                            candidates.append(res)
            candidates.sort(key=lambda x: x["value_score"], reverse=True)
            display_2x_3x_report(candidates[:10], min_prob, max_prob, "NFL", reg_stats, ["season", "week"], "player_display_name")

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
        print("  [4] NBA DFS Lineup Optimizer ($50,000 Salary Cap)")
        print("  [5] Change Slate Date (e.g., Tomorrow's Games)")
        print("  [6] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4/5/6]: ").strip()

        if choice == "1":
            player = input("\nPlayer Name (e.g., Nikola Jokic): ").strip()
            p_logs = nba_df[nba_df["PLAYER_NAME"].str.contains(player, case=False, na=False)]
            if p_logs.empty:
                print(f"[!] No NBA player found matching '{player}'.")
                continue
            team_abbr = p_logs.sort_values("GAME_DATE")["TEAM_ABBREVIATION"].iloc[-1]
            auto_opp = matchup_map[team_abbr]["opponent"] if team_abbr in matchup_map else ""
            opp = input(f"Opponent [Enter for {auto_opp}]: ").strip().upper() or auto_opp
            stat = input("Stat (PTS, REB, AST, PRA, PR, PA, RA, FG3M): ").strip().upper()
            line = float(input("Prop Line: ").strip())
            res = simulate_nba_prop(nba_ctx, p_logs, opp, stat, line)
            print(f"\n{res['player']} OVER {line} {stat} -> Median: {res['proj']:.1f} | Hit Prob: {res['over_prob']:.2%} | Flags: {res['flags']}")

        elif choice == "2":
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()
            legs = []
            for pid in active_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                t = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if t not in matchup_map or len(p_logs) < 5 or matchup_map[t]["blowout_prob"] >= 0.34:
                    continue
                h = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(h) < 5 or h["MIN"].mean() < 20.0:
                    continue
                for st_col in ["PTS", "REB", "AST", "PRA"]:
                    avg = h[st_col].mean()
                    if (st_col in ["REB", "AST"] and avg < 4.5) or (st_col == "PTS" and avg < 10.0):
                        continue
                    res = simulate_nba_prop(nba_ctx, p_logs, matchup_map[t]["opponent"], st_col, np.floor(avg * 0.85) + 0.5)
                    if res["over_prob"] >= 0.75:
                        legs.append(res)
            legs.sort(key=lambda x: x["over_prob"], reverse=True)
            for l in legs[:12]:
                print(f"  [{l['over_prob']:.1%}] {l['player']} ({l['team']}) OVER {l['line']} {l['stat']} ({l['flags']})")

        elif choice == "3":
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()
            cands = []
            for pid in active_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                t = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if t not in matchup_map or len(p_logs) < 5 or matchup_map[t]["blowout_prob"] >= 0.34:
                    continue
                h = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(h) < 5 or h["MIN"].mean() < 22.0:
                    continue
                for st_col in ["PTS", "REB", "AST", "PRA"]:
                    rec_avg = h[st_col].tail(5).mean()
                    if (st_col in ["REB", "AST"] and rec_avg < 4.5) or (st_col == "PTS" and rec_avg < 11.0):
                        continue
                    alt_line = np.floor(rec_avg * 1.15) + 0.5
                    res = simulate_nba_prop(nba_ctx, p_logs, matchup_map[t]["opponent"], st_col, alt_line)
                    if 0.30 <= res["over_prob"] <= 0.50:
                        res["trend_ratio"] = rec_avg / max(h[st_col].mean(), 1.0)
                        res["value_score"] = float(res["over_prob"] * res["def_mult"])
                        cands.append(res)
            cands.sort(key=lambda x: x["value_score"], reverse=True)
            display_2x_3x_report(cands[:10], 0.30, 0.50, "NBA", nba_df, ["GAME_DATE"], "PLAYER_NAME")

        elif choice == "4":
            df_pool = build_nba_dfs_pool(nba_ctx)
            lineups = optimize_nba_dfs_lineups(df_pool, mode="gpp", num_lineups=3)
            for idx, lu in enumerate(lineups, start=1):
                print(f"\n=== OPTIMAL NBA DFS LINEUP #{idx} (Salary: ${lu['salary'].sum():,} | Proj: {lu['proj'].sum():.1f}) ===")
                for _, r in lu.iterrows():
                    print(f"  {r['slot']:<5} {r['name']:<22} ({r['team']}) ${r['salary']:>6,} | Proj: {r['proj']:>5.1f} | Ceil: {r['ceiling']:>5.1f}")

        elif choice == "5":
            nd = input("Enter NBA Slate Date (YYYY-MM-DD): ").strip()
            if nd:
                nba_ctx = build_nba_context(nd)
                nba_df, matchup_map, todays_games = nba_ctx["nba_df"], nba_ctx["matchup_map"], nba_ctx["todays_games"]

        elif choice == "6" or choice.lower() in ["back", "quit", "exit", "q"]:
            break

    return nba_ctx


# =====================================================================
#                        MAIN MULTI-SPORT LAUNCHER
# =====================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("   MULTI-SPORT 1,000,000-SIM PROP, DFS & VALUE ENGINE            ")
    print("=================================================================")
    cached_nfl = None
    cached_nba = None

    while True:
        print("\nSelect a Sport:")
        print("  [1] NFL (Props, Parlays, Pro DFS Optimizer & 2x–3x Value Finder)")
        print("  [2] NBA (Props, Parlays, NBA DFS Optimizer & 2x–3x Value Finder)")
        print("  [3] College Sports (Men's CBB Props/Spreads/Upsets & CFB Spreads)")
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
                print("[!] college_simulator.py not found in this folder.")
        elif sport == "4" or sport.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

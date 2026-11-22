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


def _ensure_nfl_anytime_td(df):
    """Computes anytime_td on the fly if missing from cached NFL DataFrames."""
    if "anytime_td" in df.columns:
        return df
    df = df.copy()
    rush_td_s = df["rushing_tds"].fillna(0) if "rushing_tds" in df.columns else 0
    rec_td_s = df["receiving_tds"].fillna(0) if "receiving_tds" in df.columns else 0
    st_td_s = df["special_teams_tds"].fillna(0) if "special_teams_tds" in df.columns else 0
    df["anytime_td"] = rush_td_s + rec_td_s + st_td_s
    return df


def prob_to_american_odds(prob):
    """Converts win probability (0..1) into fair American Moneyline odds."""
    p = float(np.clip(prob, 0.01, 0.99))
    if p >= 0.50:
        return f"{int(round(-100.0 * p / (1.0 - p)))}"
    return f"+{int(round(100.0 * (1.0 - p) / p))}"


def fetch_live_dk_salaries(sport="NFL"):
    """Fetches live DraftKings main-slate salaries from DK's public lobby & draftables endpoints."""
    headers = {"User-Agent": "Mozilla/5.0"}
    salaries, dk_ids, slate_players = {}, {}, set()
    try:
        lobby_url = f"https://www.draftkings.com/lobby/getcontests?sport={sport}"
        lobby = requests.get(lobby_url, headers=headers, timeout=8).json()
        draft_groups = lobby.get("DraftGroups", [])
        if not draft_groups:
            return salaries, dk_ids, slate_players

        classic_groups = [
            dg for dg in draft_groups
            if str(dg.get("ContestTypeId", "")) in ["21", "70", ""] and dg.get("GameCount", 0) >= 2
        ] or draft_groups
        main_dg = max(classic_groups, key=lambda x: x.get("GameCount", 0))
        dg_id = main_dg.get("DraftGroupId")
        if not dg_id:
            return salaries, dk_ids, slate_players

        draftables_url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{dg_id}/draftables"
        d_data = requests.get(draftables_url, headers=headers, timeout=8).json()
        for item in d_data.get("draftables", []):
            disp_name = str(item.get("displayName", "")).strip()
            sal = item.get("salary")
            dk_id = item.get("draftableId")
            team_abbr = str(item.get("teamAbbreviation", "")).strip()
            pos = str(item.get("position", "")).strip()
            if disp_name and sal:
                norm = _norm_name(disp_name)
                salaries[norm] = int(sal)
                dk_ids[norm] = f"{disp_name} ({dk_id})" if dk_id else disp_name
                slate_players.add(norm)
                if pos == "DST" and team_abbr:
                    t_norm = _norm_name(team_abbr)
                    dst_norm = _norm_name(f"{team_abbr} DST")
                    salaries[t_norm] = int(sal)
                    salaries[dst_norm] = int(sal)
                    dk_ids[dst_norm] = f"{disp_name} ({dk_id})" if dk_id else disp_name
                    slate_players.update({t_norm, dst_norm})
    except Exception:
        pass
    return salaries, dk_ids, slate_players


def _parse_uploaded_dfs_csv(custom_csv_df):
    """Parses both DraftKings ('Name') and FanDuel ('Nickname'/'First Name'+'Last Name') CSVs."""
    csv_salaries, csv_ids = {}, {}
    if custom_csv_df is None or custom_csv_df.empty:
        return csv_salaries, csv_ids

    for _, row in custom_csv_df.iterrows():
        nm = str(row.get("Name", row.get("Nickname", ""))).strip()
        if (not nm or nm.lower() == "nan") and "First Name" in row and "Last Name" in row:
            nm = f"{row['First Name']} {row['Last Name']}".strip()
        if not nm or nm.lower() == "nan":
            continue

        norm_nm = _norm_name(nm)
        sal_val = int(pd.to_numeric(row.get("Salary", 5000), errors="coerce") or 5000)
        csv_salaries[norm_nm] = sal_val

        if "Name + ID" in row and pd.notna(row["Name + ID"]):
            csv_ids[norm_nm] = str(row["Name + ID"])
        elif "Id" in row and pd.notna(row["Id"]):
            csv_ids[norm_nm] = f"{nm} ({row['Id']})"
        else:
            csv_ids[norm_nm] = nm

        pos_val = str(row.get("Position", "")).strip().upper()
        team_val = str(row.get("TeamAbbrev", row.get("Team", ""))).strip()
        if pos_val in ["D", "DST", "DEF"] and team_val:
            csv_salaries[_norm_name(team_val)] = sal_val
            csv_salaries[_norm_name(f"{team_val} DST")] = sal_val
            csv_ids[_norm_name(f"{team_val} DST")] = csv_ids[norm_nm]

    return csv_salaries, csv_ids


# =====================================================================
#                   RECENCY-WEIGHTED 100 / 500 REPORT
# =====================================================================
def generate_100_500_report(df_2yr, sort_cols, target_col, threshold, entity_mask=None):
    if target_col == "anytime_td" and "anytime_td" not in df_2yr.columns:
        df_2yr = _ensure_nfl_anytime_td(df_2yr)
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
    reg_stats = _ensure_nfl_anytime_td(reg_stats)

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

        dk = (
            (pass_yds * 0.04) + (pass_tds * 4.0) - (ints * 1.0)
            + (rush_yds * 0.1) + (rush_tds * 6.0)
            + (rec * 1.0) + (rec_yds * 0.1) + (rec_tds * 6.0)
            - (fumbles * 1.0)
        )
        dk += np.where(pass_yds >= 300, 3.0, 0.0)
        dk += np.where(rush_yds >= 100, 3.0, 0.0)
        dk += np.where(rec_yds >= 100, 3.0, 0.0)

        fd = (
            (pass_yds * 0.04) + (pass_tds * 4.0) - (ints * 1.0)
            + (rush_yds * 0.1) + (rush_tds * 6.0)
            + (rec * 0.5) + (rec_yds * 0.1) + (rec_tds * 6.0)
            - (fumbles * 2.0)
        )
        return dk, fd

    reg_stats["dk_points"], reg_stats["fd_points"] = compute_dfs_points(reg_stats)

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

    team_pos_bump = {}
    team_injury_penalty = {}
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
                    pen = 3.2 if pos_str == "QB" else 1.1
                    team_injury_penalty[t_abbr] = min(5.5, team_injury_penalty.get(t_abbr, 0.0) + pen)
    except Exception:
        pass

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
    team_off_pts, team_def_pts, team_margin_std, team_net_rating = {}, {}, {}, {}
    all_teams = set(completed_sched["home_team"]).union(set(completed_sched["away_team"]))

    for t in all_teams:
        t_games = completed_sched[
            (completed_sched["home_team"] == t) | (completed_sched["away_team"] == t)
        ].tail(16)
        if t_games.empty:
            team_off_pts[t], team_def_pts[t], team_margin_std[t], team_net_rating[t] = 22.0, 22.0, 11.0, 0.0
            continue
        scored = np.where(t_games["home_team"] == t, t_games["home_score"], t_games["away_score"])
        allowed = np.where(t_games["home_team"] == t, t_games["away_score"], t_games["home_score"])
        margins = scored - allowed
        w = np.exp(np.linspace(-1.5, 0.0, len(t_games)))
        w /= w.sum()
        team_off_pts[t] = float(np.sum(scored * w))
        team_def_pts[t] = float(np.sum(allowed * w))
        team_net_rating[t] = team_off_pts[t] - team_def_pts[t]
        team_margin_std[t] = float(max(np.std(margins), 8.5))

    curr_sched = schedules[schedules["season"] == current_season]
    wk_games = curr_sched[curr_sched["week"] == current_week].copy()
    matchup_map = {}

    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et_str = now_et.strftime("%Y-%m-%d")

    for _, game in wk_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        spread = game.get("spread_line", np.nan)
        vegas_total = game.get("total_line", np.nan)
        has_vegas = pd.notna(spread) and pd.notna(vegas_total)

        roof = str(game.get("roof", "outdoors")).lower()
        wind = float(game.get("wind", 0)) if pd.notna(game.get("wind", np.nan)) else 0.0
        temp = float(game.get("temp", 65)) if pd.notna(game.get("temp", np.nan)) else 65.0

        weekday_str = str(game.get("weekday", "")).lower()
        gameday_str = str(game.get("gameday", "")).strip()
        is_thursday = (
            ("thursday" in weekday_str)
            or ("thursday" in gameday_str.lower())
            or (gameday_str == today_et_str and now_et.weekday() == 3)
        )

        # Situational adjustments: Home field (+2.0), TNF road fatigue (-1.2), Wind suppression, Injury drops
        home_adj = 2.0 - team_injury_penalty.get(home, 0.0)
        away_adj = (-1.2 if is_thursday else 0.0) - team_injury_penalty.get(away, 0.0)

        comp_home_pts = ((team_off_pts.get(home, 22.0) + team_def_pts.get(away, 22.0)) / 2.0) + (home_adj * 0.5)
        comp_away_pts = ((team_off_pts.get(away, 22.0) + team_def_pts.get(home, 22.0)) / 2.0) + (away_adj * 0.5)
        if roof in ["outdoors", "open"] and wind >= 16.0:
            comp_home_pts *= 0.93
            comp_away_pts *= 0.93

        comp_total = comp_home_pts + comp_away_pts
        blended_total = (0.65 * comp_total + 0.35 * vegas_total) if has_vegas else comp_total

        model_home_margin = comp_home_pts - comp_away_pts
        vegas_home_margin = float(spread) if pd.notna(spread) else model_home_margin
        blended_home_margin = (0.70 * model_home_margin + 0.30 * vegas_home_margin) if has_vegas else model_home_margin

        home_spread = -blended_home_margin
        away_spread = -home_spread
        comb_vol = np.sqrt((team_margin_std.get(home, 11.0) ** 2 + team_margin_std.get(away, 11.0) ** 2) / 2.0)
        if blended_total >= 47.5:
            comb_vol *= 1.12

        margin_sims = np.random.normal(blended_home_margin, comb_vol, N_SIMS).astype(np.float32)
        home_win_prob = float(np.mean(margin_sims > 0.0))
        away_win_prob = 1.0 - home_win_prob
        blowout_prob = float(np.mean(np.abs(margin_sims) >= 14.0))
        sneaky_blowout = (abs(home_spread) <= 4.0) and (blowout_prob >= 0.32)

        for team, opp, t_spread, t_pts, is_home, w_prob in [
            (home, away, home_spread, comp_home_pts, True, home_win_prob),
            (away, home, away_spread, comp_away_pts, False, away_win_prob),
        ]:
            matchup_map[team] = {
                "opponent": opp,
                "is_home": is_home,
                "home_team": home,
                "away_team": away,
                "team_spread": float(t_spread),
                "total_line": float(blended_total),
                "comp_total": float(comp_total),
                "team_proj_pts": float(t_pts),
                "win_prob": float(w_prob),
                "home_win_prob": float(home_win_prob),
                "away_win_prob": float(away_win_prob),
                "proj_home_pts": float(comp_home_pts),
                "proj_away_pts": float(comp_away_pts),
                "has_vegas": has_vegas,
                "game_label": f"{away} @ {home}",
                "blowout_prob": blowout_prob,
                "sneaky_blowout": sneaky_blowout,
                "roof": roof,
                "wind": wind,
                "temp": temp,
                "is_thursday": is_thursday,
            }

    try:
        tz_today = now_et.strftime("%Y%m%d")
        sb = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={tz_today}",
            timeout=8,
        ).json()
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            comps = comp["competitors"]
            h_abbr = espn_abbr_map.get(comps[0]["team"]["abbreviation"], comps[0]["team"]["abbreviation"])
            a_abbr = espn_abbr_map.get(comps[1]["team"]["abbreviation"], comps[1]["team"]["abbreviation"])
            if now_et.weekday() == 3:
                for t_code in [h_abbr, a_abbr]:
                    if t_code in matchup_map:
                        matchup_map[t_code]["is_thursday"] = True
    except Exception:
        pass

    def_lookup = {}
    all_stats = [
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "passing_tds", "rushing_tds", "receiving_tds", "anytime_td", "dk_points", "fd_points",
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


def simulate_nfl_games(nfl_ctx):
    """Returns straight-up winner probabilities & projected scores for every NFL game this week."""
    matchup_map = nfl_ctx["matchup_map"]
    seen_games = set()
    results = []

    for team, info in matchup_map.items():
        g_lbl = info["game_label"]
        if g_lbl in seen_games:
            continue
        seen_games.add(g_lbl)

        home = info["home_team"]
        away = info["away_team"]
        h_prob = info["home_win_prob"]
        a_prob = info["away_win_prob"]

        if h_prob >= a_prob:
            winner, win_prob = home, h_prob
        else:
            winner, win_prob = away, a_prob

        flags = []
        if info["is_thursday"]:
            flags.append("TNF Primetime")
        if info["roof"] in ["outdoors", "open"] and info["wind"] >= 16.0:
            flags.append(f"Wind {info['wind']:.0f}mph")
        if info["sneaky_blowout"]:
            flags.append(f"Sneaky Blowout Risk {info['blowout_prob']:.0%}")
        elif info["blowout_prob"] >= 0.36:
            flags.append(f"Blowout Risk {info['blowout_prob']:.0%}")

        results.append({
            "Matchup": g_lbl,
            "Projected Winner": winner,
            "Win Prob": win_prob,
            "Fair Moneyline": prob_to_american_odds(win_prob),
            "Fair Payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
            "Proj Score": f"{away} {info['proj_away_pts']:.1f} - {home} {info['proj_home_pts']:.1f}",
            "Home Win %": f"{h_prob:.1%}",
            "Away Win %": f"{a_prob:.1%}",
            "Is TNF": info["is_thursday"],
            "Flags": " | ".join(flags) if flags else "Clean Environment",
        })

    results.sort(key=lambda x: x["Win Prob"], reverse=True)
    return results


def simulate_nfl_prop(ctx, p_logs, opponent_team, stat_col, prop_line=0.0, manual_bump=1.0, override_team=None):
    volume_map = {
        "passing_yards": ("attempts", 8),
        "rushing_yards": ("carries", 4),
        "receiving_yards": ("targets", 2),
        "receptions": ("targets", 2),
    }
    team_col = ctx["team_col"]
    if stat_col == "anytime_td" and "anytime_td" not in p_logs.columns:
        p_logs = _ensure_nfl_anytime_td(p_logs)

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
    if stat_col == "anytime_td":
        consistency = float(np.mean(h10["anytime_td"] >= 1.0)) if not h10.empty else 0.0
    else:
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
        if stat_col == "anytime_td":
            total_factor = 1.0 + ((comp_total - 44.0) * 0.011)
            script_factor = 1.0 + (team_spread * 0.008)
        elif stat_col == "rushing_yards":
            script_factor = 1.0 + (-team_spread * 0.006)
        else:
            script_factor = 1.0 + (team_spread * 0.005)

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
            np.clip(total_factor * script_factor * weather_mult * blowout_penalty * auto_bump, 0.80, 1.30)
        )
        script_note = f"Proj Total: {comp_total:.1f} | Spread: {player_team} {team_spread:+.1f} | Blowout Prob: {blowout_prob:.0%}"
    else:
        comp_total = 44.0
        script_mult = float(np.clip(auto_bump, 0.85, 1.25))
        script_note = "Game Total Est: 44.0 pts"

    if stat_col == "anytime_td":
        ewm_td = healthy["anytime_td"].ewm(span=6).mean().iloc[-1]
        season_td = healthy["anytime_td"].mean()
        base_lambda = max(0.04, 0.65 * ewm_td + 0.35 * season_td)
        adj_lambda = float(np.clip(base_lambda * def_mult * script_mult, 0.03, 1.65))
        sims = np.random.poisson(adj_lambda, N_SIMS).astype(np.float32)
        breakdown = f"Expected TDs: {adj_lambda:.2f}/game (Def: {def_mult:.2f}x | Env: {script_mult:.2f}x)"
        proj_val = adj_lambda
        floor_val = float(np.percentile(sims, 20))
        ceil_val = float(np.percentile(sims, 80))
    elif stat_col in volume_map:
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
        proj_val = float(np.median(sims))
        floor_val = float(np.percentile(sims, 20))
        ceil_val = float(np.percentile(sims, 80))
    else:
        ewm_avg = healthy[stat_col].ewm(span=4).mean().iloc[-1]
        stat_std = max(healthy[stat_col].std(), 0.5)
        adj_avg = ewm_avg * def_mult * script_mult
        sims = np.clip(np.random.normal(adj_avg, stat_std, N_SIMS).astype(np.float32), 0, None)
        breakdown = f"Weighted Avg: {adj_avg:.2f}"
        proj_val = float(np.median(sims))
        floor_val = float(np.percentile(sims, 20))
        ceil_val = float(np.percentile(sims, 80))

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
        "proj": proj_val,
        "floor": floor_val,
        "ceiling": ceil_val,
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
    """Builds a projected NFL DFS player pool with live DK/CSV prices & IR/Injury filtering."""
    reg_stats = nfl_ctx["reg_stats"]
    team_col = nfl_ctx["team_col"]
    matchup_map = nfl_ctx["matchup_map"]
    out_players = nfl_ctx["out_players"]
    starting_qbs = nfl_ctx.get("starting_qbs", set())
    active_roster_team = nfl_ctx.get("active_roster_team", {})
    stat_col = "dk_points" if site == "DK" else "fd_points"

    csv_salaries, csv_ids = _parse_uploaded_dfs_csv(custom_csv_df)
    salary_source = f"Uploaded {site} CSV (Exact Prices)" if csv_salaries else "Estimated Model Prices (Upload CSV for Exact Slate Prices)"

    if not csv_salaries and site == "DK":
        live_sals, live_ids, _ = fetch_live_dk_salaries("NFL")
        if live_sals:
            csv_salaries, csv_ids = live_sals, live_ids
            salary_source = "Live DraftKings Main Slate (Exact Prices)"

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
                "salary_source": salary_source,
                "_is_dc_qb1": 1 if (pos == "QB" and p_norm in starting_qbs) else 0,
            }
        )

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
                "salary_source": salary_source,
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
    team_star_out_penalty = {}
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
                    team_star_out_penalty[t] = min(5.5, team_star_out_penalty.get(t, 0.0) + 2.4)

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

            home_last = team_last_date.get(home)
            away_last = team_last_date.get(away)
            home_rest = int((target_dt - home_last).days) if pd.notna(home_last) else 2
            away_rest = int((target_dt - away_last).days) if pd.notna(away_last) else 2

            # Apply same situational B2B fatigue, road penalty & missing star penalties used in props
            home_b2b_pen = -1.6 if home_rest == 1 else (0.6 if home_rest >= 3 else 0.0)
            away_b2b_pen = -2.8 if away_rest == 1 else (0.5 if away_rest >= 3 else 0.0)

            comp_home = (((team_off_pts.get(home, 114.0) + team_def_pts.get(away, 114.0)) / 2.0) * (
                0.5 + 0.5 * pace_ratio
            )) + home_b2b_pen - team_star_out_penalty.get(home, 0.0)
            comp_away = (((team_off_pts.get(away, 114.0) + team_def_pts.get(home, 114.0)) / 2.0) * (
                0.5 + 0.5 * pace_ratio
            )) + away_b2b_pen - team_star_out_penalty.get(away, 0.0)
            comp_total = comp_home + comp_away

            exp_home_margin = comp_home - comp_away + 2.4
            comb_vol = np.sqrt(
                (team_margin_std.get(home, 12.0) ** 2 + team_margin_std.get(away, 12.0) ** 2) / 2.0
            )
            avg_3pa_rate = (team_3pa_rate.get(home, 0.39) + team_3pa_rate.get(away, 0.39)) / 2.0
            if avg_3pa_rate >= 0.41:
                comb_vol *= 1.12

            margin_sims = np.random.normal(exp_home_margin, comb_vol, N_SIMS).astype(np.float32)
            home_win_prob = float(np.mean(margin_sims > 0.0))
            away_win_prob = 1.0 - home_win_prob
            blowout_prob = float(np.mean(np.abs(margin_sims) >= 15.0))
            sneaky_blowout = (abs(exp_home_margin) <= 4.0) and (blowout_prob >= 0.32)

            for team, opp, is_home, t_pts, o_pts, d_rest, w_prob in [
                (home, away, True, comp_home, comp_away, home_rest, home_win_prob),
                (away, home, False, comp_away, comp_home, away_rest, away_win_prob),
            ]:
                nba_matchups[team] = {
                    "opponent": opp,
                    "is_home": is_home,
                    "home_team": home,
                    "away_team": away,
                    "game_label": label,
                    "comp_total": float(comp_total),
                    "team_proj_pts": float(t_pts),
                    "proj_home_pts": float(comp_home + 1.2),
                    "proj_away_pts": float(comp_away - 1.2),
                    "proj_margin": float(t_pts - o_pts + (2.4 if is_home else -2.4)),
                    "win_prob": float(w_prob),
                    "home_win_prob": float(home_win_prob),
                    "away_win_prob": float(away_win_prob),
                    "pace_ratio": float(pace_ratio),
                    "days_rest": d_rest,
                    "is_b2b": d_rest == 1,
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


def simulate_nba_games(nba_ctx):
    """Returns straight-up winner probabilities using pace, B2B fatigue, road rest & star-out adjustments."""
    matchup_map = nba_ctx["matchup_map"]
    seen_games = set()
    results = []

    for team, info in matchup_map.items():
        g_lbl = info["game_label"]
        if g_lbl in seen_games:
            continue
        seen_games.add(g_lbl)

        home = info["home_team"]
        away = info["away_team"]
        h_prob = info["home_win_prob"]
        a_prob = info["away_win_prob"]

        if h_prob >= a_prob:
            winner, win_prob = home, h_prob
        else:
            winner, win_prob = away, a_prob

        flags = []
        if matchup_map.get(home, {}).get("is_b2b"):
            flags.append(f"{home} on B2B")
        if matchup_map.get(away, {}).get("is_b2b"):
            flags.append(f"{away} on Road B2B")
        if info["pace_ratio"] >= 1.03:
            flags.append(f"Fast Pace ({info['pace_ratio']:.2f}x)")
        if info["sneaky_blowout"]:
            flags.append(f"Sneaky Blowout Risk {info['blowout_prob']:.0%}")
        elif info["blowout_prob"] >= 0.35:
            flags.append(f"Blowout Risk {info['blowout_prob']:.0%}")

        results.append({
            "Matchup": g_lbl,
            "Projected Winner": winner,
            "Win Prob": win_prob,
            "Fair Moneyline": prob_to_american_odds(win_prob),
            "Fair Payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
            "Proj Score": f"{away} {info['proj_away_pts']:.1f} - {home} {info['proj_home_pts']:.1f}",
            "Home Win %": f"{h_prob:.1%}",
            "Away Win %": f"{a_prob:.1%}",
            "Flags": " | ".join(flags) if flags else "Rested / Clean Environment",
        })

    results.sort(key=lambda x: x["Win Prob"], reverse=True)
    return results


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
    """Builds projected DraftKings NBA DFS pool with live DK/CSV salaries."""
    nba_df = _ensure_nba_dk_pts(nba_ctx["nba_df"])
    matchup_map = nba_ctx["matchup_map"]

    csv_salaries, _ = _parse_uploaded_dfs_csv(custom_csv_df)
    salary_source = "Uploaded CSV (Exact Prices)" if csv_salaries else "Estimated Model Prices (Upload CSV for Exact Slate Prices)"
    if not csv_salaries:
        live_sals, _, _ = fetch_live_dk_salaries("NBA")
        if live_sals:
            csv_salaries = live_sals
            salary_source = "Live DraftKings Main Slate (Exact Prices)"

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
            "salary_source": salary_source,
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
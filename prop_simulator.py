from datetime import datetime
import itertools
import os
import time
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamelog
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

N_SIMS = 1_000_000  # 1 Million Monte Carlo simulations per projection


# =====================================================================
#                   RECENCY-WEIGHTED 100 / 500 REPORT
# =====================================================================
def generate_100_500_report(df_2yr, sort_cols, target_col, threshold, entity_mask=None):
    """
    Runs a report over the last 100 and last 500 qualifying games (strictly within
    the last 2 years), applying exponential recency weights so recent games
    count significantly more than older games.
    """
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
    print("\nLoading NFL 2-year data & current week matchups...")
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

    completed_sched = schedules.dropna(subset=["home_score", "away_score"]).sort_values(["season", "week"])
    team_off_pts, team_def_pts = {}, {}
    all_teams = set(completed_sched["home_team"]).union(set(completed_sched["away_team"]))

    for t in all_teams:
        t_games = completed_sched[
            (completed_sched["home_team"] == t) | (completed_sched["away_team"] == t)
        ].tail(16)
        if t_games.empty:
            team_off_pts[t], team_def_pts[t] = 22.0, 22.0
            continue
        scored = np.where(t_games["home_team"] == t, t_games["home_score"], t_games["away_score"])
        allowed = np.where(t_games["home_team"] == t, t_games["away_score"], t_games["home_score"])
        w = np.exp(np.linspace(-1.5, 0.0, len(t_games)))
        w /= w.sum()
        team_off_pts[t] = float(np.sum(scored * w))
        team_def_pts[t] = float(np.sum(allowed * w))

    curr_sched = schedules[schedules["season"] == current_season]
    wk_games = curr_sched[curr_sched["week"] == current_week].copy()
    matchup_map = {}

    for _, game in wk_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        spread = game.get("spread_line", np.nan)
        vegas_total = game.get("total_line", np.nan)
        has_vegas = pd.notna(spread) and pd.notna(vegas_total)

        comp_home_pts = (team_off_pts.get(home, 22.0) + team_def_pts.get(away, 22.0)) / 2.0
        comp_away_pts = (team_off_pts.get(away, 22.0) + team_def_pts.get(home, 22.0)) / 2.0
        comp_total = comp_home_pts + comp_away_pts
        blended_total = (0.65 * comp_total + 0.35 * vegas_total) if has_vegas else comp_total

        matchup_map[home] = {
            "opponent": away,
            "team_spread": -spread if pd.notna(spread) else (comp_away_pts - comp_home_pts),
            "total_line": float(blended_total),
            "comp_total": float(comp_total),
            "team_proj_pts": float(comp_home_pts),
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }
        matchup_map[away] = {
            "opponent": home,
            "team_spread": spread if pd.notna(spread) else (comp_home_pts - comp_away_pts),
            "total_line": float(blended_total),
            "comp_total": float(comp_total),
            "team_proj_pts": float(comp_away_pts),
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }

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
    }


def simulate_nfl_prop(ctx, p_logs, opponent_team, stat_col, prop_line=0.0):
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

    def_mult = ctx["def_lookup"].get((opponent_team, position, stat_col), 1.0)
    game_info = ctx["matchup_map"].get(player_team)

    if game_info and game_info["opponent"] == opponent_team:
        comp_total = game_info["comp_total"]
        team_spread = game_info["team_spread"]
        total_factor = 1.0 + ((comp_total - 44.0) * 0.0055)
        script_factor = (
            1.0 + (-team_spread * 0.006)
            if stat_col == "rushing_yards"
            else 1.0 + (team_spread * 0.005)
        )
        script_mult = float(np.clip(total_factor * script_factor, 0.85, 1.20))
        script_note = f"Comp Game Total: {comp_total:.1f} pts | Spread: {player_team} {team_spread:+.1f}"
    else:
        comp_total = 44.0
        script_mult = 1.0
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
def build_nba_context():
    print("\nLoading NBA 2-year data & today's slate...")
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    curr_szn = f"{start_year}-{str(start_year + 1)[-2:]}"
    prev_szn = f"{start_year - 1}-{str(start_year)[-2:]}"

    dfs = []
    for szn in [prev_szn, curr_szn]:
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=szn, player_or_team_abbreviation="P", season_type_all_star="Regular Season"
            ).get_data_frames()[0]
            logs["SEASON_STR"] = szn
            dfs.append(logs)
        except Exception as e:
            print(f"[!] Warning loading NBA {szn}: {e}")

    nba_df = pd.concat(dfs, ignore_index=True)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"])
    nba_df["MIN"] = pd.to_numeric(nba_df["MIN"], errors="coerce").fillna(0)
    nba_df["OPP_ABBR"] = nba_df["MATCHUP"].apply(lambda x: str(x).split()[-1].strip())

    nba_df["PRA"] = nba_df["PTS"] + nba_df["REB"] + nba_df["AST"]
    nba_df["PR"] = nba_df["PTS"] + nba_df["REB"]
    nba_df["PA"] = nba_df["PTS"] + nba_df["AST"]
    nba_df["RA"] = nba_df["REB"] + nba_df["AST"]

    team_games = (
        nba_df.groupby(["TEAM_ABBREVIATION", "OPP_ABBR", "GAME_ID", "GAME_DATE"])["PTS"]
        .sum()
        .reset_index()
        .sort_values("GAME_DATE")
    )
    team_off_pts, team_def_pts = {}, {}
    for t, grp in team_games.groupby("TEAM_ABBREVIATION"):
        recent_off = grp.tail(20)
        w = np.exp(np.linspace(-1.6, 0.0, len(recent_off)))
        w /= w.sum()
        team_off_pts[t] = float(np.sum(recent_off["PTS"].values * w))

    for opp, grp in team_games.groupby("OPP_ABBR"):
        recent_def = grp.tail(20)
        w = np.exp(np.linspace(-1.6, 0.0, len(recent_def)))
        w /= w.sum()
        team_def_pts[opp] = float(np.sum(recent_def["PTS"].values * w))

    nba_matchups = {}
    todays_games = []
    try:
        board = scoreboard.ScoreBoard().get_dict()
        for g in board["scoreboard"]["games"]:
            home = g["homeTeam"]["teamTricode"]
            away = g["awayTeam"]["teamTricode"]
            label = f"{away} @ {home}"
            todays_games.append(label)

            comp_home = (team_off_pts.get(home, 114.0) + team_def_pts.get(away, 114.0)) / 2.0
            comp_away = (team_off_pts.get(away, 114.0) + team_def_pts.get(home, 114.0)) / 2.0
            comp_total = comp_home + comp_away

            nba_matchups[home] = {
                "opponent": away,
                "game_label": label,
                "comp_total": float(comp_total),
                "team_proj_pts": float(comp_home),
            }
            nba_matchups[away] = {
                "opponent": home,
                "game_label": label,
                "comp_total": float(comp_total),
                "team_proj_pts": float(comp_away),
            }
    except Exception:
        pass

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
        "nba_df": nba_df,
        "todays_games": todays_games,
        "matchup_map": nba_matchups,
        "def_lookup": def_lookup,
    }


def simulate_nba_prop(ctx, p_logs, opponent_abbr, stat_col, prop_line=0.0):
    rec_logs = p_logs.sort_values("GAME_DATE").tail(20).copy()
    full_name = rec_logs["PLAYER_NAME"].iloc[-1]
    team_abbr = rec_logs["TEAM_ABBREVIATION"].iloc[-1]

    healthy = rec_logs[rec_logs["MIN"] >= 12.0].tail(15).copy()
    if len(healthy) < 4:
        healthy = rec_logs.tail(15).copy()

    def_mult = ctx["def_lookup"].get((opponent_abbr, stat_col), 1.0)
    game_info = ctx["matchup_map"].get(team_abbr)

    if game_info and game_info["opponent"] == opponent_abbr:
        comp_total = game_info["comp_total"]
        pace_mult = float(np.clip(1.0 + ((comp_total - 228.0) * 0.0025), 0.90, 1.12))
        script_note = f"Comp Game Total: {comp_total:.1f} pts (Pace {pace_mult:.2f}x)"
    else:
        comp_total = 228.0
        pace_mult = 1.0
        script_note = "Comp Game Total Est: 228.0 pts"

    ewm_min = healthy["MIN"].ewm(span=5).mean().iloc[-1]
    min_std = max(healthy["MIN"].std(), 1.5)

    rate_series = (healthy[stat_col] / healthy["MIN"].replace(0, np.nan)).dropna()
    ewm_rate = rate_series.ewm(span=5).mean().iloc[-1]
    rate_std = max(rate_series.std(), 0.02)

    adj_rate = ewm_rate * def_mult * pace_mult
    sim_min = np.clip(np.random.normal(ewm_min, min_std, N_SIMS).astype(np.float32), 0, 48.0)
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
        "stat": stat_col,
        "line": prop_line,
        "proj": float(np.median(sims)),
        "floor": float(np.percentile(sims, 20)),
        "ceiling": float(np.percentile(sims, 80)),
        "over_prob": over_prob,
        "under_prob": float(np.mean(sims < prop_line)),
        "fair_mult": fair_mult,
        "def_mult": def_mult,
        "script_mult": pace_mult,
        "script_note": script_note,
        "breakdown": f"{ewm_min:.1f} mins x {adj_rate:.2f} {stat_col}/min",
        "games_used": len(healthy),
    }


# =====================================================================
#        SHARED DISPLAY HELPER FOR 2x–3x VALUE + 100/500 REPORT
# =====================================================================
def display_2x_3x_report(top_plays, min_prob, max_prob, sport_label, df_source, sort_cols, player_col):
    if not top_plays:
        print(f"[!] No {sport_label} props fell between {min_prob:.0%} and {max_prob:.0%}. Try widening your % range.")
        return

    print("\n" + "=" * 84)
    print(f" TOP {sport_label} 2x–3x VALUE PLAYS ({min_prob:.0%}–{max_prob:.0%} HIT PROBABILITY)")
    print("=" * 84)

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
            f"Median Proj: {play['proj']:.1f} (80th Ceil: {play['ceiling']:.1f})"
        )
        print(
            f"    Matchup: {play['game']}  |  {play['script_note']}  |  "
            f"Def Trend: {play['def_mult']:.2f}x | Player Form: {play['trend_ratio']:.2f}x"
        )
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
        print("\n" + "=" * 84)
        print(f" SUGGESTED {sport_label} 2-LEG & 3-LEG HIGH-MULTIPLIER PARLAYS")
        print("=" * 84)
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

            t0 = time.time()
            res = simulate_nfl_prop(nfl_ctx, p_logs, opp, stat, line)
            elapsed = time.time() - t0

            print("\n" + "=" * 66)
            print(f" {res['player']} ({res['position']} - {res['team']}) vs. {res['opponent']} | {res['stat'].upper()} {res['line']}")
            print("=" * 66)
            print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
            print(f" Healthy Games:     {res['games_used']} | {res['script_note']}")
            print(f" Matchup Factor:    Defense {res['def_mult']:.2f}x | Game Script {res['script_mult']:.2f}x")
            print(f" Engine Breakdown:  {res['breakdown']}")
            print("-" * 66)
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
            print("=" * 66)

        elif choice == "2":
            print(f"\n--- WEEK {current_week} NFL AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
            try:
                rate_in = float(input("Minimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
                min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
                parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
                discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
            except ValueError:
                min_hit_rate, parlay_legs, discount = 0.75, 3, 0.85

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
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in pos_props:
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
            qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)
            print(f"\nCompleted {sims_count:,} sims in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
            for leg in qualifying_legs[:12]:
                print(
                    f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                    f"(Proj: {leg['proj']:.1f} | {leg['game']})"
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
                min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
            except ValueError:
                min_prob, max_prob = 0.30, 0.50

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
                team, pos = p[team_col], p["position"]
                if team not in matchup_map or pos not in nfl_props_ext:
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
        print(f"\nNBA Simulator Menu (Season {nba_ctx['curr_szn']} | Games Today: {len(todays_games)}):")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan Today's NBA Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] NBA 2x–3x Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [4] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4]: ").strip()

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
                input(f"Opponent Team Abbr [Press Enter for Today's Opp: {auto_opp}]: ").strip().upper()
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

            t0 = time.time()
            res = simulate_nba_prop(nba_ctx, p_logs, opp, stat, line)
            elapsed = time.time() - t0

            print("\n" + "=" * 66)
            print(f" {res['player']} ({res['team']}) vs. {res['opponent']} | {res['stat']} {res['line']}")
            print("=" * 66)
            print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
            print(f" Healthy Games:     {res['games_used']} | {res['script_note']}")
            print(f" Matchup Factor:    Defense {res['def_mult']:.2f}x | Pace Factor {res['script_mult']:.2f}x")
            print(f" Engine Breakdown:  {res['breakdown']}")
            print("-" * 66)
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
            print("=" * 66)

        elif choice == "2":
            print("\n--- TODAY'S NBA AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
            if not matchup_map:
                print("[!] No NBA games scheduled today.")
                continue
            try:
                rate_in = float(input("Minimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
                min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
                parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
                discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
            except ValueError:
                min_hit_rate, parlay_legs, discount = 0.75, 3, 0.85

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
            qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)
            print(f"\nCompleted {sims_count:,} sims in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
            for leg in qualifying_legs[:12]:
                print(
                    f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                    f"(Proj: {leg['proj']:.1f} | {leg['game']})"
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
            print("\n--- TODAY'S NBA 2x–3x VALUE FINDER (30%–50% HIT RATE + 100/500 REPORT) ---")
            if not matchup_map:
                print("[!] No NBA games scheduled on today's scoreboard.")
                continue

            try:
                min_pct = float(input("Minimum Hit Rate % (default 30 for 30%): ").strip() or "30")
                max_pct = float(input("Maximum Hit Rate % (default 50 for 50%): ").strip() or "50")
                min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
            except ValueError:
                min_prob, max_prob = 0.30, 0.50

            t0 = time.time()
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_nba_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()

            candidates = []
            sims_count = 0
            print(f"\nScanning Today's {len(todays_games)} NBA games for 2x–3x alt-line upside plays...")

            for pid in active_nba_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                team = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if team not in matchup_map or len(p_logs) < 5:
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

        elif choice == "4" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

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
        print("  [3] Quit")
        sport = input("Select [1/2/3]: ").strip()

        if sport == "1":
            cached_nfl = run_nfl_menu(cached_nfl)
        elif sport == "2":
            cached_nba = run_nba_menu(cached_nba)
        elif sport == "3" or sport.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")
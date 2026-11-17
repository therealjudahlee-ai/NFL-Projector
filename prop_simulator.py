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
#                              NFL ENGINE
# =====================================================================
def run_nfl_app():
    print("\nFetching live NFL schedules & player stats...")
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    seasons = [current_season - 1, current_season]

    player_stats = nfl.load_player_stats(seasons).to_pandas()
    schedules = nfl.load_schedules([current_season]).to_pandas()

    reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()
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
            (pass_yds * 0.04)
            + (pass_tds * 4.0)
            - (ints * 1.0)
            + (rush_yds * 0.1)
            + (rush_tds * 6.0)
            + (rec * 1.0)
            + (rec_yds * 0.1)
            + (rec_tds * 6.0)
            - (fumbles * 1.0)
        )
        pts += np.where(pass_yds >= 300, 3.0, 0.0)
        pts += np.where(rush_yds >= 100, 3.0, 0.0)
        pts += np.where(rec_yds >= 100, 3.0, 0.0)
        return pts

    reg_stats["dk_points"] = compute_dk_points(reg_stats)

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

    # Build weekly matchups
    wk_games = schedules[schedules["week"] == current_week].copy()
    matchup_map = {}
    for _, game in wk_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        spread = game.get("spread_line", np.nan)
        total = game.get("total_line", np.nan)
        has_vegas = pd.notna(spread) and pd.notna(total)

        matchup_map[home] = {
            "opponent": away,
            "team_spread": -spread if pd.notna(spread) else 0.0,
            "total_line": total if pd.notna(total) else 44.5,
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }
        matchup_map[away] = {
            "opponent": home,
            "team_spread": spread if pd.notna(spread) else 0.0,
            "total_line": total if pd.notna(total) else 44.5,
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }

    # Build defensive matchup lookup
    recent_def = reg_stats[reg_stats["season"] == current_season]
    if recent_def["week"].nunique() < 4:
        recent_def = reg_stats

    def_lookup = {}
    all_stats = [
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "passing_tds", "rushing_tds", "receiving_tds", "dk_points",
    ]
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = recent_def[recent_def["position"] == pos]
        for stat in all_stats:
            if stat not in pos_df.columns:
                continue
            game_totals = pos_df.groupby(["opponent_team", "season", "week"])[stat].sum().reset_index()
            league_avg = game_totals[stat].mean()
            if pd.isna(league_avg) or league_avg == 0:
                continue
            for opp, val in game_totals.groupby("opponent_team")[stat].mean().items():
                def_lookup[(opp, pos, stat)] = val / league_avg

    def run_1m_sim(p_logs, opponent_team, stat_col, prop_line=0.0):
        recent = p_logs.sort_values(["season", "week"]).tail(15).copy()
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

        def_mult = def_lookup.get((opponent_team, position, stat_col), 1.0)
        game_info = matchup_map.get(player_team)

        if game_info and game_info["opponent"] == opponent_team and game_info["has_vegas"]:
            total_line = game_info["total_line"]
            team_spread = game_info["team_spread"]
            total_factor = 1.0 + ((total_line - 44.5) * 0.005)
            script_factor = (
                1.0 + (-team_spread * 0.006)
                if stat_col == "rushing_yards"
                else 1.0 + (team_spread * 0.005)
            )
            vegas_mult = float(np.clip(total_factor * script_factor, 0.85, 1.18))
            vegas_note = f"Spread: {player_team} {team_spread:+.1f} | Total: {total_line}"
        else:
            vegas_mult = 1.0
            vegas_note = "Vegas line N/A (1.00x)"

        if stat_col in volume_map:
            vol_col, _ = volume_map[stat_col]
            ewm_vol = healthy[vol_col].ewm(span=5).mean().iloc[-1]
            vol_std = max(healthy[vol_col].std(), 1.0)

            eff_series = (healthy[stat_col] / healthy[vol_col].replace(0, np.nan)).dropna()
            ewm_eff = eff_series.ewm(span=5).mean().iloc[-1]
            eff_std = max(eff_series.std(), 0.05)

            adj_vol = ewm_vol * vegas_mult
            adj_eff = ewm_eff * def_mult

            sim_vol = np.clip(np.random.normal(adj_vol, vol_std, N_SIMS).astype(np.float32), 0, None)
            sim_eff = np.clip(np.random.normal(adj_eff, eff_std, N_SIMS).astype(np.float32), 0, None)
            sims = sim_vol * sim_eff
            breakdown = f"{adj_vol:.1f} {vol_col} x {adj_eff:.2f} {stat_col}/{vol_col[:-1]}"
        else:
            ewm_avg = healthy[stat_col].ewm(span=5).mean().iloc[-1]
            stat_std = max(healthy[stat_col].std(), 0.5)
            adj_avg = ewm_avg * def_mult * vegas_mult
            sims = np.clip(np.random.normal(adj_avg, stat_std, N_SIMS).astype(np.float32), 0, None)
            breakdown = f"Weighted Avg: {adj_avg:.2f}"

        return {
            "player": full_name,
            "position": position,
            "team": player_team,
            "opponent": opponent_team,
            "game": game_info["game_label"] if game_info else f"{player_team} vs {opponent_team}",
            "stat": stat_col,
            "line": prop_line,
            "proj": float(np.median(sims)),
            "floor": float(np.percentile(sims, 20)),
            "ceiling": float(np.percentile(sims, 80)),
            "over_prob": float(np.mean(sims > prop_line)),
            "under_prob": float(np.mean(sims < prop_line)),
            "def_mult": def_mult,
            "vegas_mult": vegas_mult,
            "vegas_note": vegas_note,
            "breakdown": breakdown,
            "games_used": len(healthy),
        }

    def nfl_single_player():
        print("\n--- NFL SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
        print("Stats: passing_yards, rushing_yards, receiving_yards, receptions, dk_points, passing_tds, rushing_tds, receiving_tds")
        player = input("\nPlayer Name (e.g., Josh Allen): ").strip()
        p_logs = reg_stats[reg_stats["player_display_name"].str.contains(player, case=False, na=False)]
        if p_logs.empty:
            print(f"[!] No NFL player found matching '{player}'.")
            return

        player_team = p_logs.sort_values(["season", "week"])[team_col].iloc[-1]
        auto_opp = matchup_map[player_team]["opponent"] if player_team in matchup_map else ""
        opp_prompt = (
            f"Opponent Team Abbr [Press Enter for Week {current_week} opp: {auto_opp}]: "
            if auto_opp
            else "Opponent Team Abbr (e.g., KC, BAL, DAL): "
        )
        opp = input(opp_prompt).strip().upper() or auto_opp

        stat = input("Stat (e.g., passing_yards, rushing_yards, receiving_yards, dk_points): ").strip().lower()
        if stat not in reg_stats.columns:
            print(f"[!] Invalid stat '{stat}'.")
            return

        try:
            line = float(input("Prop Line (e.g., 245.5): ").strip())
        except ValueError:
            print("[!] Invalid number for prop line.")
            return

        t0 = time.time()
        res = run_1m_sim(p_logs, opp, stat, line)
        elapsed = time.time() - t0

        print("\n" + "=" * 66)
        print(f" {res['player']} ({res['position']} - {res['team']}) vs. {res['opponent']} | {res['stat'].upper()} {res['line']}")
        print("=" * 66)
        print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
        print(f" Healthy Games:     {res['games_used']} | {res['vegas_note']}")
        print(f" Matchup Factor:    Defense {res['def_mult']:.2f}x | Vegas Script {res['vegas_mult']:.2f}x")
        print(f" Engine Breakdown:  {res['breakdown']}")
        print("-" * 66)
        print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
        print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}")
        print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
        print("=" * 66)

    def nfl_parlay_builder():
        print(f"\n--- WEEK {current_week} NFL AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
        try:
            rate_in = float(input("Minimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
            min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
            parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
            discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
        except ValueError:
            print("Using defaults: 75% hit rate, 3 legs, 0.85 alt-line floor.")
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
        print("\nRunning 1,000,000 simulations per active starter across the slate...")

        for _, p in active_players.iterrows():
            team, pos = p[team_col], p["position"]
            if team not in matchup_map or pos not in pos_props:
                continue
            opp = matchup_map[team]["opponent"]
            p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values(["season", "week"])
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
                res = run_1m_sim(p_logs, opp, stat_col, target_line)
                sims_count += N_SIMS
                if res["over_prob"] >= min_hit_rate:
                    qualifying_legs.append(res)

        elapsed = time.time() - t0
        qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)

        print(f"\nCompleted {sims_count:,} simulations in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
        for leg in qualifying_legs[:12]:
            print(
                f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                f"(Proj: {leg['proj']:.1f} | {leg['game']})"
            )

        if len(qualifying_legs) < parlay_legs:
            print(f"\n[!] Fewer than {parlay_legs} legs hit >= {min_hit_rate:.0%}. Lower your % or aggressiveness.")
            return

        print(f"\n================ TOP {parlay_legs}-LEG NFL PARLAYS (1 LEG PER GAME) ================")
        valid_parlays = []
        for combo in itertools.combinations(qualifying_legs[:14], parlay_legs):
            if len({leg["game"] for leg in combo}) < parlay_legs:
                continue
            combined_prob = float(np.prod([leg["over_prob"] for leg in combo]))
            valid_parlays.append((combined_prob, combo))

        valid_parlays.sort(key=lambda x: x[0], reverse=True)
        for idx, (prob, combo) in enumerate(valid_parlays[:5], start=1):
            fair_odds = int(-100 * prob / (1 - prob)) if prob > 0.5 else int(100 * (1 - prob) / prob)
            print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.2%} (Fair Odds: {fair_odds:+d})")
            for leg in combo:
                print(
                    f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                    f"[{leg['over_prob']:.2%} | Proj: {leg['proj']:.1f} | {leg['game']}]"
                )

    def nfl_dfs_optimizer():
        print(f"\n--- WEEK {current_week} NFL DFS LINEUP PROJECTOR & OPTIMIZER ($50,000 CAP) ---")
        print("Contest Mode:")
        print("  [1] Cash Game (Maximizes Median Projection)")
        print("  [2] GPP Tournament (Maximizes 80th-Percentile Ceiling + QB/Receiver Stack)")
        is_gpp = input("Select [1/2]: ").strip() == "2"

        dk_salaries = {}
        if os.path.exists("DKSalaries.csv"):
            print("Detected DKSalaries.csv! Loading official DraftKings salaries...")
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
        print("Simulating 1,000,000 slates for all active NFL DFS skill players...")
        for _, p in active_players.iterrows():
            team, pos = p[team_col], p["position"]
            if team not in matchup_map or pos not in ["QB", "RB", "WR", "TE"]:
                continue
            opp = matchup_map[team]["opponent"]
            p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values(["season", "week"])
            if len(p_logs) < 3 or p_logs["dk_points"].tail(6).mean() < 4.5:
                continue

            res = run_1m_sim(p_logs, opp, "dk_points", 15.0)
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
            return

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

    print(f"Loaded NFL Season {current_season} | Current Slate: Week {current_week}")
    while True:
        print("\nNFL Simulator Menu:")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan This Week's Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] NFL DFS Lineup Projector & Optimizer ($50k Salary Cap)")
        print("  [4] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4]: ").strip()

        if choice == "1":
            nfl_single_player()
        elif choice == "2":
            nfl_parlay_builder()
        elif choice == "3":
            nfl_dfs_optimizer()
        elif choice == "4" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


# =====================================================================
#                              NBA ENGINE
# =====================================================================
def run_nba_app():
    print("\nFetching live NBA player game logs & today's slate...")
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
            print(f"[!] Warning loading {szn}: {e}")

    nba_df = pd.concat(dfs, ignore_index=True)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"])
    nba_df["MIN"] = pd.to_numeric(nba_df["MIN"], errors="coerce").fillna(0)
    nba_df["OPP_ABBR"] = nba_df["MATCHUP"].apply(lambda x: str(x).split()[-1].strip())

    # Combo Stats
    nba_df["PRA"] = nba_df["PTS"] + nba_df["REB"] + nba_df["AST"]
    nba_df["PR"] = nba_df["PTS"] + nba_df["REB"]
    nba_df["PA"] = nba_df["PTS"] + nba_df["AST"]
    nba_df["RA"] = nba_df["REB"] + nba_df["AST"]

    # Fetch Today's Live NBA Matchups
    nba_matchups = {}
    todays_games = []
    try:
        board = scoreboard.ScoreBoard().get_dict()
        for g in board["scoreboard"]["games"]:
            home = g["homeTeam"]["teamTricode"]
            away = g["awayTeam"]["teamTricode"]
            label = f"{away} @ {home}"
            todays_games.append(label)
            nba_matchups[home] = {"opponent": away, "game_label": label}
            nba_matchups[away] = {"opponent": home, "game_label": label}
    except Exception:
        pass

    # Build NBA Defense Lookup
    recent = nba_df[nba_df["SEASON_STR"] == curr_szn]
    if recent["GAME_ID"].nunique() < 30:
        recent = nba_df

    def_lookup = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA", "PR", "PA", "RA"]:
        game_totals = recent.groupby(["OPP_ABBR", "GAME_ID"])[stat].sum().reset_index()
        league_avg = game_totals[stat].mean()
        if pd.isna(league_avg) or league_avg == 0:
            continue
        for opp, val in game_totals.groupby("OPP_ABBR")[stat].mean().items():
            def_lookup[(opp, stat)] = val / league_avg

    def run_nba_1m_sim(p_logs, opponent_abbr, stat_col, prop_line=0.0):
        rec_logs = p_logs.sort_values("GAME_DATE").tail(20).copy()
        full_name = rec_logs["PLAYER_NAME"].iloc[-1]
        team_abbr = rec_logs["TEAM_ABBREVIATION"].iloc[-1]

        healthy = rec_logs[rec_logs["MIN"] >= 12.0].tail(15).copy()
        if len(healthy) < 4:
            healthy = rec_logs.tail(15).copy()

        def_mult = def_lookup.get((opponent_abbr, stat_col), 1.0)
        ewm_min = healthy["MIN"].ewm(span=6).mean().iloc[-1]
        min_std = max(healthy["MIN"].std(), 1.5)

        rate_series = (healthy[stat_col] / healthy["MIN"].replace(0, np.nan)).dropna()
        ewm_rate = rate_series.ewm(span=6).mean().iloc[-1]
        rate_std = max(rate_series.std(), 0.02)

        adj_rate = ewm_rate * def_mult
        sim_min = np.clip(np.random.normal(ewm_min, min_std, N_SIMS).astype(np.float32), 0, 48.0)
        sim_rate = np.clip(np.random.normal(adj_rate, rate_std, N_SIMS).astype(np.float32), 0, None)
        sims = sim_min * sim_rate

        return {
            "player": full_name,
            "team": team_abbr,
            "opponent": opponent_abbr,
            "game": nba_matchups[team_abbr]["game_label"] if team_abbr in nba_matchups else f"{team_abbr} vs {opponent_abbr}",
            "stat": stat_col,
            "line": prop_line,
            "proj": float(np.median(sims)),
            "floor": float(np.percentile(sims, 20)),
            "ceiling": float(np.percentile(sims, 80)),
            "over_prob": float(np.mean(sims > prop_line)),
            "under_prob": float(np.mean(sims < prop_line)),
            "def_mult": def_mult,
            "breakdown": f"{ewm_min:.1f} mins x {adj_rate:.2f} {stat_col}/min",
            "games_used": len(healthy),
        }

    def nba_single_player():
        print("\n--- NBA SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
        print("Stats: PTS, REB, AST, FG3M, STL, BLK, PRA, PR, PA, RA")
        player = input("\nPlayer Name (e.g., Nikola Jokic, Jayson Tatum): ").strip()
        p_logs = nba_df[nba_df["PLAYER_NAME"].str.contains(player, case=False, na=False)]
        if p_logs.empty:
            print(f"[!] No NBA player found matching '{player}'.")
            return

        team_abbr = p_logs.sort_values("GAME_DATE")["TEAM_ABBREVIATION"].iloc[-1]
        auto_opp = nba_matchups[team_abbr]["opponent"] if team_abbr in nba_matchups else ""
        opp_prompt = (
            f"Opponent Team Abbr [Press Enter for Today's Opp: {auto_opp}]: "
            if auto_opp
            else "Opponent Team Abbr (e.g., BOS, LAL, DEN): "
        )
        opp = input(opp_prompt).strip().upper() or auto_opp

        stat = input("Stat (PTS, REB, AST, FG3M, PRA, PR, PA, RA): ").strip().upper()
        if stat not in nba_df.columns:
            print(f"[!] Invalid stat '{stat}'.")
            return

        try:
            line = float(input("Prop Line (e.g., 26.5): ").strip())
        except ValueError:
            print("[!] Invalid number for prop line.")
            return

        t0 = time.time()
        res = run_nba_1m_sim(p_logs, opp, stat, line)
        elapsed = time.time() - t0

        print("\n" + "=" * 66)
        print(f" {res['player']} ({res['team']}) vs. {res['opponent']} | {res['stat']} {res['line']}")
        print("=" * 66)
        print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
        print(f" Healthy Games:     {res['games_used']} | Defense Factor: {res['def_mult']:.2f}x")
        print(f" Engine Breakdown:  {res['breakdown']}")
        print("-" * 66)
        print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
        print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}")
        print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
        print("=" * 66)

    def nba_parlay_builder():
        print("\n--- TODAY'S NBA AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
        if not nba_matchups:
            print("[!] No NBA games scheduled on today's scoreboard.")
            return

        print(f"Detected {len(todays_games)} games today:")
        for g in todays_games:
            print(f"  • {g}")

        try:
            rate_in = float(input("\nMinimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
            min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
            parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
            discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip())
        except ValueError:
            print("Using defaults: 75% hit rate, 3 legs, 0.85 alt-line floor.")
            min_hit_rate, parlay_legs, discount = 0.75, 3, 0.85

        t0 = time.time()
        recent_cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
        active_ids = nba_df[nba_df["GAME_DATE"] >= recent_cutoff]["PLAYER_ID"].unique()

        qualifying_legs = []
        sims_count = 0
        print("\nRunning 1,000,000 simulations per active rotation player...")

        for pid in active_ids:
            p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
            team = p_logs["TEAM_ABBREVIATION"].iloc[-1]
            if team not in nba_matchups or len(p_logs) < 5:
                continue

            healthy = p_logs[p_logs["MIN"] >= 15.0].tail(12)
            if len(healthy) < 5 or healthy["MIN"].mean() < 20.0:
                continue

            opp = nba_matchups[team]["opponent"]
            for stat in ["PTS", "REB", "AST", "PRA"]:
                raw_avg = healthy[stat].mean()
                if (stat in ["REB", "AST"] and raw_avg < 4.5) or (stat == "PTS" and raw_avg < 10.0):
                    continue

                target_line = np.floor(raw_avg * discount) + 0.5
                res = run_nba_1m_sim(p_logs, opp, stat, target_line)
                sims_count += N_SIMS
                if res["over_prob"] >= min_hit_rate:
                    qualifying_legs.append(res)

        elapsed = time.time() - t0
        qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)

        print(f"\nCompleted {sims_count:,} simulations in {elapsed:.1f}s | Found {len(qualifying_legs)} legs >= {min_hit_rate:.0%}:")
        for leg in qualifying_legs[:12]:
            print(
                f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                f"(Proj: {leg['proj']:.1f} | {leg['game']})"
            )

        if len(qualifying_legs) < parlay_legs:
            print(f"\n[!] Fewer than {parlay_legs} legs hit >= {min_hit_rate:.0%}. Lower your % or aggressiveness.")
            return

        print(f"\n================ TOP {parlay_legs}-LEG NBA PARLAYS (1 LEG PER GAME) ================")
        valid_parlays = []
        for combo in itertools.combinations(qualifying_legs[:14], parlay_legs):
            if len({leg["game"] for leg in combo}) < parlay_legs:
                continue
            combined_prob = float(np.prod([leg["over_prob"] for leg in combo]))
            valid_parlays.append((combined_prob, combo))

        valid_parlays.sort(key=lambda x: x[0], reverse=True)
        for idx, (prob, combo) in enumerate(valid_parlays[:5], start=1):
            fair_odds = int(-100 * prob / (1 - prob)) if prob > 0.5 else int(100 * (1 - prob) / prob)
            print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.2%} (Fair Odds: {fair_odds:+d})")
            for leg in combo:
                print(
                    f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                    f"[{leg['over_prob']:.2%} | Proj: {leg['proj']:.1f} | {leg['game']}]"
                )

    print(f"Loaded NBA Season {curr_szn} | Games Today: {len(todays_games)}")
    while True:
        print("\nNBA Simulator Menu:")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan Today's NBA Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3]: ").strip()

        if choice == "1":
            nba_single_player()
        elif choice == "2":
            nba_parlay_builder()
        elif choice == "3" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


# =====================================================================
#                        MAIN MULTI-SPORT LAUNCHER
# =====================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("   MULTI-SPORT 1,000,000-SIM PROP, PARLAY & DFS PROJECTOR        ")
    print("=================================================================")
    while True:
        print("\nSelect a Sport:")
        print("  [1] NFL (Props, Automated Weekly Parlays & DFS Optimizer)")
        print("  [2] NBA (Props & Automated Daily Parlays)")
        print("  [3] Quit")
        sport = input("Select [1/2/3]: ").strip()

        if sport == "1":
            run_nfl_app()
        elif sport == "2":
            run_nba_app()
        elif sport == "3" or sport.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")
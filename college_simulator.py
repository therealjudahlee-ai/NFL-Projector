from datetime import datetime
import itertools
import time
import numpy as np
import pandas as pd
import requests

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
#             MEN'S COLLEGE BASKETBALL (NCAAB) ENGINE
# =====================================================================
def build_ncaab_context(target_date_str=None):
    if not target_date_str:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\nLoading Men's College Basketball 2-year data & {target_date_str} D1 slate...")
    now = datetime.now()
    curr_szn = now.year + 1 if now.month >= 10 else now.year
    seasons = [curr_szn - 1, curr_szn]

    p_dfs = []
    for szn in seasons:
        url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_player_boxscores/player_box_{szn}.parquet"
        try:
            df_s = pd.read_parquet(url)
            df_s["SEASON_YR"] = szn
            p_dfs.append(df_s)
        except Exception as e:
            print(f"[!] Warning loading NCAAB {szn} box scores: {e}")

    if not p_dfs:
        raise RuntimeError("Could not download NCAAB box scores. Check internet connection.")

    mbb_df = pd.concat(p_dfs, ignore_index=True)

    # Normalize column names safely across releases
    col_map = {
        "athlete_display_name": "PLAYER_NAME",
        "team_short_display_name": "TEAM_NAME",
        "team_abbreviation": "TEAM_ABBR",
        "opponent_team_abbreviation": "OPP_ABBR",
        "opponent_team_short_display_name": "OPP_NAME",
        "game_date": "GAME_DATE",
        "minutes": "MIN",
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "three_point_field_goals_made": "FG3M",
        "three_point_field_goals_attempted": "FG3A",
        "field_goals_attempted": "FGA",
        "free_throws_attempted": "FTA",
        "offensive_rebounds": "OREB",
        "turnovers": "TOV",
        "home_away": "HOME_AWAY",
    }
    for old_c, new_c in col_map.items():
        if old_c in mbb_df.columns and new_c not in mbb_df.columns:
            mbb_df[new_c] = mbb_df[old_c]

    mbb_df["GAME_DATE"] = pd.to_datetime(mbb_df["GAME_DATE"], errors="coerce")
    for c in ["MIN", "PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "FTA", "OREB", "TOV"]:
        if c not in mbb_df.columns:
            mbb_df[c] = 0.0
        mbb_df[c] = pd.to_numeric(mbb_df[c], errors="coerce").fillna(0.0)

    mbb_df["PRA"] = mbb_df["PTS"] + mbb_df["REB"] + mbb_df["AST"]
    mbb_df["PR"] = mbb_df["PTS"] + mbb_df["REB"]
    mbb_df["PA"] = mbb_df["PTS"] + mbb_df["AST"]
    mbb_df["RA"] = mbb_df["REB"] + mbb_df["AST"]
    mbb_df["IS_HOME"] = mbb_df.get("HOME_AWAY", pd.Series("home", index=mbb_df.index)).astype(str).str.lower() == "home"

    # Team-level Pace, Scoring & Margin Volatility (for Sneaky Blowout detection)
    team_agg = (
        mbb_df.groupby(["TEAM_ABBR", "OPP_ABBR", "game_id", "GAME_DATE"])
        .agg({"PTS": "sum", "FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum", "FG3A": "sum"})
        .reset_index()
        .sort_values("GAME_DATE")
    )
    team_agg["POSS"] = team_agg["FGA"] + 0.44 * team_agg["FTA"] - team_agg["OREB"] + team_agg["TOV"]
    team_agg["FG3_RATE"] = team_agg["FG3A"] / team_agg["FGA"].replace(0, 1)

    opp_pts_map = team_agg.set_index(["game_id", "TEAM_ABBR"])["PTS"].to_dict()
    team_agg["OPP_PTS"] = team_agg.apply(lambda r: opp_pts_map.get((r["game_id"], r["OPP_ABBR"]), r["PTS"]), axis=1)
    team_agg["MARGIN"] = team_agg["PTS"] - team_agg["OPP_PTS"]

    team_off, team_def, team_pace, team_std, team_3pa = {}, {}, {}, {}, {}
    for t, grp in team_agg.groupby("TEAM_ABBR"):
        rec = grp.tail(22)
        w = np.exp(np.linspace(-1.6, 0.0, len(rec)))
        w /= w.sum()
        team_off[t] = float(np.sum(rec["PTS"].values * w))
        team_def[t] = float(np.sum(rec["OPP_PTS"].values * w))
        team_pace[t] = float(np.sum(rec["POSS"].values * w))
        team_3pa[t] = float(np.sum(rec["FG3_RATE"].values * w))
        team_std[t] = float(max(rec["MARGIN"].std(), 11.0))

    league_pace = np.mean(list(team_pace.values())) if team_pace else 68.5

    # Fetch Today's D1 Scoreboard & Vegas Odds from ESPN
    date_compact = target_date_str.replace("-", "")
    espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=500&dates={date_compact}"
    matchup_map = {}
    todays_games = []

    try:
        resp = requests.get(espn_url, timeout=10).json()
        for ev in resp.get("events", []):
            comp = ev["competitions"][0]
            competitors = comp["competitors"]
            home_c = next(c for c in competitors if c["homeAway"] == "home")
            away_c = next(c for c in competitors if c["homeAway"] == "away")

            home = home_c["team"]["abbreviation"]
            away = away_c["team"]["abbreviation"]
            home_name = home_c["team"]["shortDisplayName"]
            away_name = away_c["team"]["shortDisplayName"]
            label = f"{away_name} ({away}) @ {home_name} ({home})"
            todays_games.append(label)

            # Extract Vegas spread & total if posted
            vegas_spread, vegas_total = np.nan, np.nan
            odds_list = comp.get("odds", [])
            if odds_list:
                vegas_total = float(odds_list[0].get("overUnder", np.nan))
                details = str(odds_list[0].get("details", ""))
                parts = details.split()
                if len(parts) == 2:
                    try:
                        fav_abbr, sp_val = parts[0], float(parts[1])
                        vegas_spread = sp_val if fav_abbr == home else -sp_val
                    except ValueError:
                        pass

            game_pace = (team_pace.get(home, league_pace) + team_pace.get(away, league_pace)) / 2.0
            pace_ratio = game_pace / league_pace

            comp_home = ((team_off.get(home, 74.0) + team_def.get(away, 74.0)) / 2.0) * (0.5 + 0.5 * pace_ratio)
            comp_away = ((team_off.get(away, 74.0) + team_def.get(home, 74.0)) / 2.0) * (0.5 + 0.5 * pace_ratio)
            comp_total = comp_home + comp_away
            blended_total = (0.65 * comp_total + 0.35 * vegas_total) if pd.notna(vegas_total) else comp_total

            exp_home_margin = -vegas_spread if pd.notna(vegas_spread) else (comp_home - comp_away + 3.2)
            comb_vol = np.sqrt((team_std.get(home, 12.0) ** 2 + team_std.get(away, 12.0) ** 2) / 2.0)
            if (team_3pa.get(home, 0.37) + team_3pa.get(away, 0.37)) / 2.0 >= 0.40:
                comb_vol *= 1.12

            margin_sims = np.random.normal(exp_home_margin, comb_vol, N_SIMS).astype(np.float32)
            blowout_prob = float(np.mean(np.abs(margin_sims) >= 16.0))
            sneaky_blowout = (abs(exp_home_margin) <= 4.5) and (blowout_prob >= 0.33)

            for t_abbr, o_abbr, is_h, t_pts, o_pts in [
                (home, away, True, comp_home, comp_away),
                (away, home, False, comp_away, comp_home),
            ]:
                matchup_map[t_abbr] = {
                    "opponent": o_abbr,
                    "is_home": is_h,
                    "game_label": label,
                    "comp_total": float(blended_total),
                    "team_proj_pts": float(t_pts),
                    "pace_ratio": float(pace_ratio),
                    "blowout_prob": blowout_prob,
                    "sneaky_blowout": sneaky_blowout,
                }
    except Exception as e:
        print(f"[!] Warning fetching ESPN NCAAB scoreboard: {e}")

    # Recency-weighted defensive lookup by stat
    def_lookup = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "PRA", "PR", "PA", "RA"]:
        g_tot = mbb_df.groupby(["OPP_ABBR", "game_id", "GAME_DATE"])[stat].sum().reset_index().sort_values("GAME_DATE")
        lg_avg = g_tot.tail(1500)[stat].mean()
        if pd.isna(lg_avg) or lg_avg == 0:
            continue
        for opp, grp in g_tot.groupby("OPP_ABBR"):
            rec_g = grp.tail(20)
            w = np.exp(np.linspace(-1.5, 0.0, len(rec_g)))
            w /= w.sum()
            def_lookup[(opp, stat)] = float(np.sum(rec_g[stat].values * w)) / lg_avg

    return {
        "curr_szn": curr_szn,
        "target_date": target_date_str,
        "mbb_df": mbb_df,
        "todays_games": todays_games,
        "matchup_map": matchup_map,
        "def_lookup": def_lookup,
    }


def simulate_ncaab_prop(ctx, p_logs, opponent_abbr, stat_col, prop_line=0.0):
    rec = p_logs.sort_values("GAME_DATE").tail(22).copy()
    full_name = rec["PLAYER_NAME"].iloc[-1]
    team_abbr = rec["TEAM_ABBR"].iloc[-1]

    healthy = rec[rec["MIN"] >= 12.0].tail(15).copy()
    if len(healthy) < 4:
        healthy = rec.tail(15).copy()

    h10 = healthy.tail(10)
    consistency = float(np.mean(h10[stat_col] >= (h10[stat_col].mean() * 0.85))) if not h10.empty else 0.5
    def_mult = ctx["def_lookup"].get((opponent_abbr, stat_col), 1.0)
    game_info = ctx["matchup_map"].get(team_abbr)

    ewm_min = healthy["MIN"].ewm(span=5).mean().iloc[-1]
    min_std = max(healthy["MIN"].std(), 1.8)
    rate_series = (healthy[stat_col] / healthy["MIN"].replace(0, np.nan)).dropna()
    ewm_rate = rate_series.ewm(span=5).mean().iloc[-1]
    rate_std = max(rate_series.std(), 0.02)

    flags = []
    blowout_prob, sneaky_blowout, pace_mult, min_adj = 0.25, False, 1.0, 1.0

    if game_info and game_info["opponent"] == opponent_abbr:
        comp_total = game_info["comp_total"]
        blowout_prob = game_info["blowout_prob"]
        sneaky_blowout = game_info["sneaky_blowout"]
        pace_mult = float(np.clip(game_info["pace_ratio"], 0.90, 1.12))

        split_logs = healthy[healthy["IS_HOME"] == game_info["is_home"]]
        if len(split_logs) >= 4:
            split_rate = (split_logs[stat_col] / split_logs["MIN"].replace(0, np.nan)).dropna().mean()
            ewm_rate = 0.75 * ewm_rate + 0.25 * split_rate

        if blowout_prob >= 0.35 and ewm_min >= 25.0:
            min_adj = 0.91
            flags.append(f"{'SNEAKY ' if sneaky_blowout else ''}BLOWOUT RISK {blowout_prob:.0%} (-9% Min)")

        script_note = f"Proj Total: {comp_total:.1f} | Pace: {pace_mult:.2f}x | Blowout Prob: {blowout_prob:.0%}"
    else:
        comp_total = 148.0
        script_note = "Proj Game Total Est: 148.0 pts"

    adj_min = ewm_min * min_adj
    adj_rate = ewm_rate * def_mult * pace_mult

    sim_min = np.clip(np.random.normal(adj_min, min_std, N_SIMS).astype(np.float32), 0, 40.0)
    sim_rate = np.clip(np.random.normal(adj_rate, rate_std, N_SIMS).astype(np.float32), 0, None)
    sims = sim_min * sim_rate

    over_prob = float(np.mean(sims > prop_line))
    fair_mult = round(1.0 / over_prob, 2) if over_prob > 0.01 else 99.0

    return {
        "sport": "NCAAB",
        "player": full_name,
        "team": team_abbr,
        "opponent": opponent_abbr,
        "game": game_info["game_label"] if game_info else f"{team_abbr} vs {opponent_abbr}",
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
        "script_mult": pace_mult,
        "script_note": script_note,
        "breakdown": f"{adj_min:.1f} mins x {adj_rate:.2f} {stat_col}/min",
        "games_used": len(healthy),
    }


def run_ncaab_menu(ctx=None):
    if ctx is None:
        ctx = build_ncaab_context()

    mbb_df = ctx["mbb_df"]
    matchup_map = ctx["matchup_map"]
    todays_games = ctx["todays_games"]

    while True:
        print(f"\nMen's College Basketball Menu (Season {ctx['curr_szn']} | Date: {ctx['target_date']} | Games: {len(todays_games)}):")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan Today's D1 Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] NCAAB 2x–3x Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [4] Change Slate Date (YYYY-MM-DD)")
        print("  [5] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4/5]: ").strip()

        if choice == "1":
            print("\n--- NCAAB SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
            print("Stats: PTS, REB, AST, FG3M, PRA, PR, PA, RA")
            player = input("\nPlayer Name (e.g., Cooper Flagg, Johni Broome): ").strip()
            p_logs = mbb_df[mbb_df["PLAYER_NAME"].str.contains(player, case=False, na=False)]
            if p_logs.empty:
                print(f"[!] No NCAAB player found matching '{player}'.")
                continue

            team_abbr = p_logs.sort_values("GAME_DATE")["TEAM_ABBR"].iloc[-1]
            auto_opp = matchup_map[team_abbr]["opponent"] if team_abbr in matchup_map else ""
            opp = input(f"Opponent Team Abbr [Press Enter for {ctx['target_date']} opp: {auto_opp}]: ").strip().upper() or auto_opp
            stat = input("Stat (PTS, REB, AST, FG3M, PRA, PR, PA, RA): ").strip().upper()
            if stat not in mbb_df.columns:
                print(f"[!] Invalid stat '{stat}'.")
                continue
            try:
                line = float(input("Prop Line (e.g., 16.5): ").strip())
            except ValueError:
                print("[!] Invalid number for prop line.")
                continue

            t0 = time.time()
            res = simulate_ncaab_prop(ctx, p_logs, opp, stat, line)
            elapsed = time.time() - t0

            print("\n" + "=" * 72)
            print(f" {res['player']} ({res['team']}) vs. {res['opponent']} | {res['stat']} {res['line']}")
            print("=" * 72)
            print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
            print(f" Healthy Games:     {res['games_used']} | Consistency Score: {res['consistency']:.0%}")
            print(f" Matchup & Script:  {res['script_note']}")
            print(f" Context Flags:     {res['flags']}")
            print(f" Engine Breakdown:  {res['breakdown']} (Def: {res['def_mult']:.2f}x)")
            print("-" * 72)
            print(f" Median Projection: {res['proj']:.1f}  (Floor: {res['floor']:.1f} | Ceiling: {res['ceiling']:.1f})")
            print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}  (Fair Payout: {res['fair_mult']:.2f}x)")
            print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
            print("=" * 72)

        elif choice in ["2", "3"]:
            is_val_mode = choice == "3"
            if not matchup_map:
                print("[!] No D1 games found on ESPN scoreboard for this date.")
                continue

            try:
                if is_val_mode:
                    min_pct = float(input("\nMinimum Hit Rate % (default 30): ").strip() or "30")
                    max_pct = float(input("Maximum Hit Rate % (default 50): ").strip() or "50")
                    min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                    max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
                    parlay_legs, discount = 2, 1.15
                else:
                    rate_in = float(input("\nMinimum Hit Rate % per leg (e.g., 75): ").strip() or "75")
                    min_prob = rate_in / 100.0 if rate_in > 1 else rate_in
                    max_prob = 1.0
                    parlay_legs = int(input("Legs per parlay (e.g., 3): ").strip() or "3")
                    discount = float(input("Line Aggressiveness (1.0 = full avg, 0.85 = alt floor): ").strip() or "0.85")
                skip_blowouts = input("Skip high Blowout-Risk games (including sneaky close-spread traps)? [Y/n]: ").strip().lower() != "n"
            except ValueError:
                min_prob, max_prob, parlay_legs, discount, skip_blowouts = (0.30, 0.50, 2, 1.15, True) if is_val_mode else (0.75, 1.0, 3, 0.85, True)

            t0 = time.time()
            cutoff = mbb_df["GAME_DATE"].max() - pd.Timedelta(days=18)
            active_df = mbb_df[(mbb_df["GAME_DATE"] >= cutoff) & (mbb_df["TEAM_ABBR"].isin(matchup_map.keys()))]
            active_players = active_df["PLAYER_NAME"].unique()

            qualifying = []
            sims_count = 0
            print(f"\nScanning {len(active_players)} active D1 players on today's slate...")

            for pname in active_players:
                p_logs = mbb_df[mbb_df["PLAYER_NAME"] == pname].sort_values("GAME_DATE")
                team = p_logs["TEAM_ABBR"].iloc[-1]
                if team not in matchup_map or len(p_logs) < 5:
                    continue
                if skip_blowouts and matchup_map[team]["blowout_prob"] >= 0.35:
                    continue

                healthy = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(healthy) < 5 or healthy["MIN"].mean() < 22.0:
                    continue

                opp = matchup_map[team]["opponent"]
                for stat in ["PTS", "REB", "AST", "PRA"]:
                    recent_avg = healthy[stat].tail(5).mean()
                    two_yr_avg = healthy[stat].mean()
                    if (stat in ["REB", "AST"] and recent_avg < 4.5) or (stat == "PTS" and recent_avg < 11.0):
                        continue

                    mults = [1.12, 1.25] if is_val_mode else [discount]
                    for m in mults:
                        line_val = np.floor(recent_avg * m) + 0.5
                        res = simulate_ncaab_prop(ctx, p_logs, opp, stat, line_val)
                        sims_count += N_SIMS

                        if min_prob <= res["over_prob"] <= max_prob:
                            trend_boost = recent_avg / max(two_yr_avg, 1.0)
                            res["trend_ratio"] = trend_boost
                            res["value_score"] = float(
                                res["over_prob"] * res["def_mult"] * res["script_mult"] * np.clip(trend_boost, 0.85, 1.30)
                            )
                            qualifying.append(res)

            elapsed = time.time() - t0
            qualifying.sort(key=lambda x: x["value_score"] if is_val_mode else x["over_prob"], reverse=True)
            seen, top_plays = set(), []
            for q in qualifying:
                if (q["player"], q["stat"]) not in seen:
                    seen.add((q["player"], q["stat"]))
                    top_plays.append(q)

            print(f"\nCompleted {sims_count:,} NCAAB simulations in {elapsed:.1f}s | Found {len(top_plays)} qualifying plays:")
            for idx, leg in enumerate(top_plays[:10], start=1):
                print(
                    f"  #{idx} [{leg['over_prob']:.1%} | {leg['fair_mult']:.2f}x] {leg['player']} ({leg['team']}) "
                    f"OVER {leg['line']} {leg['stat']} (Proj: {leg['proj']:.1f} | {leg['game']} | {leg['flags']})"
                )
                if is_val_mode:
                    rep = generate_100_500_report(
                        mbb_df, ["GAME_DATE"], leg["stat"], leg["line"], entity_mask=(mbb_df["PLAYER_NAME"] == leg["player"])
                    )
                    r100, r500 = rep[100], rep[500]
                    print(
                        f"      -> 2-Yr History: Last {r100['n']}g Weighted {r100['weighted_rate']:.1%} (Raw {r100['raw_rate']:.1%}) | "
                        f"Last {r500['n']}g Weighted {r500['weighted_rate']:.1%}"
                    )

            if len(top_plays) >= parlay_legs:
                print(f"\n================ TOP {parlay_legs}-LEG NCAAB PARLAYS ================")
                combos = []
                for c in itertools.combinations(top_plays[:12], parlay_legs):
                    if len({x["game"] for x in c}) < parlay_legs:
                        continue
                    combos.append((float(np.prod([x["over_prob"] for x in c])), c))
                combos.sort(key=lambda x: x[0], reverse=True)
                for idx, (prob, c) in enumerate(combos[:5], start=1):
                    payout = 1.0 / prob if prob > 0 else 0
                    print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.1%} (Est. Payout: {payout:.2f}x)")
                    for leg in c:
                        print(f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} [{leg['over_prob']:.1%} | {leg['game']}]")

        elif choice == "4":
            nd = input("Enter NCAAB Date (YYYY-MM-DD): ").strip()
            if nd:
                ctx = build_ncaab_context(nd)
                mbb_df, matchup_map, todays_games = ctx["mbb_df"], ctx["matchup_map"], ctx["todays_games"]

        elif choice == "5" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice.")

    return ctx


# =====================================================================
#           COLLEGE FOOTBALL (NCAAF) SPREAD & TOTAL ENGINE
# =====================================================================
def build_ncaaf_context():
    print("\nLoading College Football 2-year schedules, team efficiency & live Vegas spreads...")
    now = datetime.now()
    curr_szn = now.year if now.month >= 8 else now.year - 1
    seasons = [curr_szn - 1, curr_szn]

    sched_dfs, box_dfs = [], []
    for szn in seasons:
        try:
            s_url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedules_{szn}.parquet"
            sched_dfs.append(pd.read_parquet(s_url))
        except Exception as e:
            print(f"[!] Warning loading CFB {szn} schedule: {e}")
        try:
            b_url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_team_boxscores/team_box_{szn}.parquet"
            box_dfs.append(pd.read_parquet(b_url))
        except Exception:
            pass

    sched_df = pd.concat(sched_dfs, ignore_index=True)
    sched_df["game_date"] = pd.to_datetime(sched_df["start_date"], errors="coerce")
    completed = sched_df.dropna(subset=["home_points", "away_points"]).sort_values("game_date")

    # Compute recency-weighted Team Offense, Defense, Yards/Play & Turnover Margin
    team_stats = {}
    all_teams = set(completed["home_team"].dropna()).union(set(completed["away_team"].dropna()))

    # Optional yards per play & turnovers from team_box
    ypp_map, to_map = {}, {}
    if box_dfs:
        box_df = pd.concat(box_dfs, ignore_index=True)
        if "yards_per_play" in box_df.columns and "team_short_display_name" in box_df.columns:
            for t, grp in box_df.groupby("team_short_display_name"):
                ypp_map[t] = float(pd.to_numeric(grp["yards_per_play"], errors="coerce").tail(12).mean())
                if "turnovers" in grp.columns:
                    to_map[t] = float(pd.to_numeric(grp["turnovers"], errors="coerce").tail(12).mean())

    for t in all_teams:
        g = completed[(completed["home_team"] == t) | (completed["away_team"] == t)].tail(16)
        if g.empty:
            continue
        scored = np.where(g["home_team"] == t, g["home_points"], g["away_points"]).astype(float)
        allowed = np.where(g["home_team"] == t, g["away_points"], g["home_points"]).astype(float)
        margins = scored - allowed
        w = np.exp(np.linspace(-1.6, 0.0, len(g)))
        w /= w.sum()

        team_stats[t] = {
            "off_pts": float(np.sum(scored * w)),
            "def_pts": float(np.sum(allowed * w)),
            "margin_std": float(max(np.std(margins), 11.5)),
            "ypp": ypp_map.get(t, 5.6),
            "tov": to_map.get(t, 1.3),
            "games_played": len(g),
        }

    # Fetch Upcoming FBS Games & Live Spreads from ESPN Scoreboard
    espn_cfb_url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=200"
    upcoming_games = []
    try:
        resp = requests.get(espn_cfb_url, timeout=10).json()
        for ev in resp.get("events", []):
            comp = ev["competitions"][0]
            if comp["status"]["type"]["completed"]:
                continue
            competitors = comp["competitors"]
            home_c = next(c for c in competitors if c["homeAway"] == "home")
            away_c = next(c for c in competitors if c["homeAway"] == "away")

            home_name = home_c["team"]["shortDisplayName"]
            away_name = away_c["team"]["shortDisplayName"]
            home_abbr = home_c["team"]["abbreviation"]
            away_abbr = away_c["team"]["abbreviation"]

            vegas_spread = np.nan  # Negative = Home Favored
            vegas_total = np.nan
            odds_list = comp.get("odds", [])
            if odds_list:
                vegas_total = float(odds_list[0].get("overUnder", np.nan))
                details = str(odds_list[0].get("details", ""))
                parts = details.split()
                if len(parts) == 2:
                    try:
                        fav_abbr, sp_val = parts[0], float(parts[1])
                        vegas_spread = sp_val if fav_abbr == home_abbr else -sp_val
                    except ValueError:
                        pass

            upcoming_games.append(
                {
                    "home": home_name,
                    "away": away_name,
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                    "label": f"{away_name} ({away_abbr}) @ {home_name} ({home_abbr})",
                    "vegas_spread": vegas_spread,
                    "vegas_total": vegas_total,
                    "date": ev.get("date", "")[:10],
                }
            )
    except Exception as e:
        print(f"[!] Warning fetching ESPN CFB scoreboard: {e}")

    return {
        "curr_szn": curr_szn,
        "completed": completed,
        "team_stats": team_stats,
        "upcoming_games": upcoming_games,
    }


def simulate_cfb_game(ctx, game, custom_spread=None, custom_total=None):
    t_stats = ctx["team_stats"]
    home, away = game["home"], game["away"]
    h_st = t_stats.get(home, {"off_pts": 27.0, "def_pts": 27.0, "margin_std": 13.5, "ypp": 5.6, "tov": 1.3})
    a_st = t_stats.get(away, {"off_pts": 27.0, "def_pts": 27.0, "margin_std": 13.5, "ypp": 5.6, "tov": 1.3})

    # Base projected team points + Yards/Play efficiency edge + Turnover edge + +2.8 Home Field
    ypp_edge = (h_st["ypp"] - a_st["ypp"]) * 2.5
    tov_edge = (a_st["tov"] - h_st["tov"]) * 2.2

    proj_home = ((h_st["off_pts"] + a_st["def_pts"]) / 2.0) + 1.4 + (ypp_edge / 2.0) + (tov_edge / 2.0)
    proj_away = ((a_st["off_pts"] + h_st["def_pts"]) / 2.0) - 1.4 - (ypp_edge / 2.0) - (tov_edge / 2.0)

    comp_spread = -(proj_home - proj_away)  # Negative = Home favored
    comp_total = proj_home + proj_away

    spread_line = custom_spread if custom_spread is not None else (
        game["vegas_spread"] if pd.notna(game["vegas_spread"]) else round(comp_spread * 2) / 2
    )
    total_line = custom_total if custom_total is not None else (
        game["vegas_total"] if pd.notna(game["vegas_total"]) else round(comp_total * 2) / 2
    )

    comb_std = np.sqrt((h_st["margin_std"] ** 2 + a_st["margin_std"] ** 2) / 2.0)
    if comp_total >= 58.0:
        comb_std *= 1.12  # Shootouts amplify late-game margin swings

    # Simulate 1,000,000 College Football games
    sim_margins = np.random.normal(proj_home - proj_away, comb_std, N_SIMS).astype(np.float32)
    sim_totals = np.random.normal(comp_total, comb_std * 0.92, N_SIMS).astype(np.float32)

    home_cover_prob = float(np.mean((sim_margins + spread_line) > 0))
    away_cover_prob = float(np.mean((sim_margins + spread_line) < 0))
    over_prob = float(np.mean(sim_totals > total_line))
    under_prob = float(np.mean(sim_totals < total_line))

    blowout_prob = float(np.mean(np.abs(sim_margins) >= 17.0))
    sneaky_blowout = (abs(spread_line) <= 4.0) and (blowout_prob >= 0.34)

    best_side = f"{home} {spread_line:+.1f}" if home_cover_prob >= away_cover_prob else f"{away} {-spread_line:+.1f}"
    best_side_prob = max(home_cover_prob, away_cover_prob)
    spread_edge = abs(comp_spread - spread_line)

    return {
        "game": game["label"],
        "home": home,
        "away": away,
        "spread_line": spread_line,
        "comp_spread": float(comp_spread),
        "spread_edge": float(spread_edge),
        "total_line": total_line,
        "comp_total": float(comp_total),
        "proj_score": f"{away} {proj_away:.1f} - {home} {proj_home:.1f}",
        "home_cover_prob": home_cover_prob,
        "away_cover_prob": away_cover_prob,
        "best_side": best_side,
        "best_side_prob": best_side_prob,
        "fair_mult": round(1.0 / best_side_prob, 2) if best_side_prob > 0.01 else 99.0,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "blowout_prob": blowout_prob,
        "sneaky_blowout": sneaky_blowout,
    }


def run_ncaaf_menu(ctx=None):
    if ctx is None:
        ctx = build_ncaaf_context()

    upcoming = ctx["upcoming_games"]
    completed = ctx["completed"]

    while True:
        print(f"\nCollege Football (NCAAF) Spread & Total Menu (Season {ctx['curr_szn']} | Upcoming Games: {len(upcoming)}):")
        print("  [1] Single Matchup Spread & Total Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan This Week's CFB Slate & Build ATS Parlays (1,000,000 Sims/Game)")
        print("  [3] CFB 2x–3x Alt-Spread Value Finder (30%–50% Hit Rate + 100/500 Report)")
        print("  [4] Switch Sport / Back to Main Menu")
        choice = input("Select [1/2/3/4]: ").strip()

        if choice == "1":
            print("\n--- NCAAF SINGLE MATCHUP SPREAD SIMULATOR (1,000,000 SIMS) ---")
            for idx, g in enumerate(upcoming[:25], start=1):
                sp_str = f"{g['home_abbr']} {g['vegas_spread']:+.1f}" if pd.notna(g["vegas_spread"]) else "Line TBD"
                print(f"  [{idx}] {g['label']} ({sp_str})")

            sel = input("\nEnter game # from list (or type team name): ").strip()
            game = None
            if sel.isdigit() and 1 <= int(sel) <= len(upcoming):
                game = upcoming[int(sel) - 1]
            else:
                matches = [g for g in upcoming if sel.lower() in g["label"].lower()]
                if matches:
                    game = matches[0]

            if not game:
                print("[!] Game not found on this week's FBS schedule.")
                continue

            sp_in = input(f"Home Spread ({game['home']} spread, Enter for Vegas {game['vegas_spread']}): ").strip()
            tot_in = input(f"Game Total (Enter for Vegas {game['vegas_total']}): ").strip()
            c_sp = float(sp_in) if sp_in else None
            c_tot = float(tot_in) if tot_in else None

            t0 = time.time()
            res = simulate_cfb_game(ctx, game, custom_spread=c_sp, custom_total=c_tot)
            elapsed = time.time() - t0

            print("\n" + "=" * 74)
            print(f" {res['game']} | Line: {res['home']} {res['spread_line']:+.1f} | Total: {res['total_line']}")
            print("=" * 74)
            print(f" Sims Run:          {N_SIMS:,} ({elapsed:.2f}s) | Comp Score: {res['proj_score']}")
            print(f" Model Fair Spread: {res['home']} {res['comp_spread']:+.1f} (Edge: {res['spread_edge']:.1f} pts)")
            print(f" 17+ Blowout Prob:  {res['blowout_prob']:.1%} {'[!] SNEAKY CLOSE-SPREAD BLOWOUT RISK' if res['sneaky_blowout'] else ''}")
            print("-" * 74)
            print(f" {res['home']} {res['spread_line']:+.1f} Cover Prob:  {res['home_cover_prob']:.2%}")
            print(f" {res['away']} {-res['spread_line']:+.1f} Cover Prob:  {res['away_cover_prob']:.2%}")
            print(f" OVER {res['total_line']:<5} Prob:      {res['over_prob']:.2%}  |  UNDER Prob: {res['under_prob']:.2%}")
            print("=" * 74)

        elif choice == "2":
            print("\n--- THIS WEEK'S NCAAF AUTOMATED ATS PARLAY BUILDER ---")
            try:
                rate_in = float(input("Minimum Cover Prob % per leg (e.g., 58 for 58%): ").strip() or "58")
                min_prob = rate_in / 100.0 if rate_in > 1 else rate_in
                legs = int(input("Legs per parlay (e.g., 3): ").strip() or "3")
                skip_blowouts = input("Skip high Blowout-Risk games (including sneaky -2.5 traps)? [Y/n]: ").strip().lower() != "n"
            except ValueError:
                min_prob, legs, skip_blowouts = 0.58, 3, True

            t0 = time.time()
            qualifying = []
            for g in upcoming:
                res = simulate_cfb_game(ctx, g)
                if skip_blowouts and res["blowout_prob"] >= 0.36:
                    continue
                if res["best_side_prob"] >= min_prob:
                    qualifying.append(res)

            elapsed = time.time() - t0
            qualifying.sort(key=lambda x: x["best_side_prob"], reverse=True)
            print(f"\nSimulated {len(upcoming) * N_SIMS:,} CFB games in {elapsed:.1f}s | Found {len(qualifying)} ATS edges >= {min_prob:.0%}:")
            for q in qualifying[:12]:
                print(
                    f"  [{q['best_side_prob']:.2%}] {q['best_side']} (Model: {q['home']} {q['comp_spread']:+.1f} | "
                    f"Edge: {q['spread_edge']:.1f} pts | Blowout: {q['blowout_prob']:.0%})"
                )

            if len(qualifying) >= legs:
                print(f"\n================ TOP {legs}-LEG CFB ATS PARLAYS ================")
                combos = list(itertools.combinations(qualifying[:10], legs))
                scored = [(float(np.prod([x["best_side_prob"] for x in c])), c) for c in combos]
                scored.sort(key=lambda x: x[0], reverse=True)
                for idx, (prob, c) in enumerate(scored[:5], start=1):
                    odds = int(-100 * prob / (1 - prob)) if prob > 0.5 else int(100 * (1 - prob) / prob)
                    print(f"\nParlay #{idx} | Combined Cover Prob: {prob:.2%} (Fair Odds: {odds:+d})")
                    for leg in c:
                        print(f"   -> {leg['best_side']} [{leg['best_side_prob']:.2%} | Edge: {leg['spread_edge']:.1f} pts | {leg['game']}]")

        elif choice == "3":
            print("\n--- NCAAF 2x–3x ALT-SPREAD VALUE FINDER (30%–50% HIT RATE + 100/500 REPORT) ---")
            try:
                min_pct = float(input("Minimum Cover Prob % (default 30): ").strip() or "30")
                max_pct = float(input("Maximum Cover Prob % (default 50): ").strip() or "50")
                min_prob = min_pct / 100.0 if min_pct > 1 else min_pct
                max_prob = max_pct / 100.0 if max_pct > 1 else max_pct
            except ValueError:
                min_prob, max_prob = 0.30, 0.50

            t0 = time.time()
            val_plays = []
            for g in upcoming:
                base_res = simulate_cfb_game(ctx, g)
                # Test aggressive alt-spreads (-7.5 and -13.5 beyond projected spread) for 2x-3x payout
                for push in [7.5, 13.5]:
                    alt_sp = base_res["spread_line"] - push if base_res["comp_spread"] < base_res["spread_line"] else base_res["spread_line"] + push
                    res = simulate_cfb_game(ctx, g, custom_spread=alt_sp)
                    if min_prob <= res["best_side_prob"] <= max_prob and res["spread_edge"] >= 2.0:
                        val_plays.append(res)

            elapsed = time.time() - t0
            val_plays.sort(key=lambda x: x["best_side_prob"] * (1.0 + 0.05 * x["spread_edge"]), reverse=True)
            print(f"\nCompleted {len(upcoming) * 2 * N_SIMS:,} CFB alt-spread sims in {elapsed:.1f}s!")

            # 2-Year Historical FBS Margin Cover Report (Last 100 & Last 500 FBS games)
            completed["abs_margin"] = (completed["home_points"] - completed["away_points"]).abs()
            rep = generate_100_500_report(completed, ["game_date"], "abs_margin", 14.0)
            print(
                f"FBS 2-Yr Blowout Trend (14+ pt margin): Last 100g Weighted {rep[100]['weighted_rate']:.1%} | "
                f"Last 500g Weighted {rep[500]['weighted_rate']:.1%}"
            )

            for idx, p in enumerate(val_plays[:10], start=1):
                print(
                    f"\n#{idx} [NCAAF ALT-SPREAD] {p['best_side']}  |  Cover Prob: {p['best_side_prob']:.1%} "
                    f"(Fair Payout: {p['fair_mult']:.2f}x)"
                )
                print(
                    f"    Matchup: {p['game']} | Comp Score: {p['proj_score']} | Model Fair Line: {p['home']} {p['comp_spread']:+.1f}"
                )

        elif choice == "4" or choice.lower() in ["back", "quit", "exit", "q"]:
            break
        else:
            print("Invalid choice.")

    return ctx


# =====================================================================
#                        COLLEGE LAUNCHER MENU
# =====================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("   COLLEGE 1,000,000-SIMULATOR | MEN'S CBB & COLLEGE FOOTBALL    ")
    print("=================================================================")
    cached_ncaab = None
    cached_ncaaf = None

    while True:
        print("\nSelect a College Sport:")
        print("  [1] Men's College Basketball (Props, Parlays & 2x–3x Value Finder)")
        print("  [2] College Football (Spreads, Totals, ATS Parlays & 2x–3x Alt-Spreads)")
        print("  [3] Quit")
        choice = input("Select [1/2/3]: ").strip()

        if choice == "1":
            cached_ncaab = run_ncaab_menu(cached_ncaab)
        elif choice == "2":
            cached_ncaaf = run_ncaaf_menu(cached_ncaaf)
        elif choice == "3" or choice.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")
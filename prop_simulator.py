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

    # Normalize ESPN team abbreviations (GS, NY, NO, SA, WSH, UTA) to standard 3-letter codes
    abbr_fix = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS", "UTAH": "UTA"}
    nba_df["TEAM_ABBREVIATION"] = nba_df["TEAM_ABBREVIATION"].replace(abbr_fix)
    nba_df["OPP_ABBR"] = nba_df["OPP_ABBR"].replace(abbr_fix)

    nba_df["PRA"] = nba_df["PTS"] + nba_df["REB"] + nba_df["AST"]
    nba_df["PR"] = nba_df["PTS"] + nba_df["REB"]
    nba_df["PA"] = nba_df["PTS"] + nba_df["AST"]
    nba_df["RA"] = nba_df["REB"] + nba_df["AST"]

    team_agg = (
        nba_df.groupby(["TEAM_ABBREVIATION", "OPP_ABBR", "GAME_ID", "GAME_DATE"])
        .agg({"PTS": "sum", "FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum", "FG3A": "sum"} if "FG3A" in nba_df.columns else {"PTS": "sum", "FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum"})
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
Step 2: Push to GitHub & Verify on Streamlit Cloud
Press Ctrl + S in Notepad to save prop_simulator.py.




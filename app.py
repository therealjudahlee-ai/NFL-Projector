from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

import college_simulator as col_sim
import prop_simulator as pro_sim

st.set_page_config(
    page_title="1M-Sim Multi-Sport Projector & DFS Optimizer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 1,000,000-Sim Multi-Sport Prop, Spread & Pro DFS Engine")
st.caption("NFL • NBA • Men's College Basketball • College Football | Recency-Weighted + Sneaky Blowout Protection")


def ensure_nba_dk_pts(df):
    """Guarantees DK_PTS exists on cached NBA DataFrames."""
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

    dd = (
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
        + np.where(dd >= 2, 1.5, 0.0)
        + np.where(dd >= 3, 3.0, 0.0)
    )
    return df


@st.cache_resource(show_spinner="Loading NFL 2-year stats, weather & injuries...")
def get_nfl():
    return pro_sim.build_nfl_context()


@st.cache_resource(show_spinner="Loading NBA 2-year stats & live slate...")
def get_nba(date_str):
    ctx = pro_sim.build_nba_context(date_str)
    ctx["nba_df"] = ensure_nba_dk_pts(ctx["nba_df"])
    return ctx


@st.cache_resource(show_spinner="Loading Men's CBB 2-year stats & D1 slate...")
def get_ncaab(date_str):
    return col_sim.build_ncaab_context(date_str)


@st.cache_resource(show_spinner="Loading College Football 2-year stats & FBS spreads...")
def get_ncaaf():
    return col_sim.build_ncaaf_context()


sport_tab = st.sidebar.radio(
    "Select Sport",
    ["🏈 NFL", "🏀 NBA", "🎓 Men's College Basketball", "🏟️ College Football"],
)

# =====================================================================
#                             NFL WEB TAB
# =====================================================================
if sport_tab == "🏈 NFL":
    ctx = get_nfl()
    reg_stats = ctx["reg_stats"]
    team_col = ctx["team_col"]
    matchup_map = ctx["matchup_map"]

    st.subheader(f"NFL Season {ctx['season']} — Week {ctx['week']}")
    mode = st.tabs([
        "Single Player Prop",
        "Auto Parlay Builder",
        "2x–3x Value Finder (100/500 Report)",
        "🏆 Pro NFL DFS Optimizer",
    ])

    with mode[0]:
        col1, col2, col3, col4 = st.columns(4)
        player_in = col1.text_input("Player Name", value="Josh Allen")
        stat_in = col2.selectbox(
            "Stat",
            ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "dk_points", "passing_tds", "rushing_tds", "receiving_tds"],
        )
        line_in = col3.number_input("Prop Line", value=245.5, step=0.5)
        star_out = col4.checkbox("Key Teammate OUT (+8% Usage)")

        if st.button("Run 1,000,000 Simulations", key="nfl_single"):
            p_logs = reg_stats[reg_stats["player_display_name"].str.contains(player_in, case=False, na=False)]
            if p_logs.empty:
                st.error(f"No NFL player found matching '{player_in}'.")
            else:
                p_team = p_logs.sort_values("sort_key")[team_col].iloc[-1]
                opp = matchup_map[p_team]["opponent"] if p_team in matchup_map else ""
                res = pro_sim.simulate_nfl_prop(ctx, p_logs, opp, stat_in, line_in, manual_bump=(1.08 if star_out else 1.0))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Median Projection", f"{res['proj']:.1f}", f"Floor {res['floor']:.1f} / Ceil {res['ceiling']:.1f}")
                c2.metric(f"OVER {line_in} Prob", f"{res['over_prob']:.1%}", f"Fair Payout: {res['fair_mult']:.2f}x")
                c3.metric("Consistency Score", f"{res['consistency']:.0%}", f"{res['games_used']} healthy games")
                c4.metric("Blowout Risk", f"{res['blowout_prob']:.0%}", "Sneaky Blowout!" if res["sneaky_blowout"] else "Normal")
                st.info(f"**Matchup:** {res['game']} | **Script:** {res['script_note']} | **Flags:** {res['flags']}")

    with mode[1]:
        c1, c2, c3, c4 = st.columns(4)
        min_hr = c1.slider("Min Hit Rate %", 55, 95, 75) / 100.0
        n_legs = c2.selectbox("Parlay Legs", [2, 3, 4], index=1)
        disc = c3.slider("Line Aggressiveness", 0.70, 1.05, 0.85, 0.05)
        skip_bo = c4.checkbox("Skip Blowout-Risk Games", value=True, key="nfl_bo1")

        if st.button("Scan NFL Slate & Build Parlays", key="nfl_parlay"):
            recent_pool = reg_stats[
                (reg_stats["season"] == ctx["season"]) & (reg_stats["week"] >= max(1, ctx["week"] - 3))
            ]
            active = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            vol_map = {"passing_yards": ("attempts", 8), "rushing_yards": ("carries", 4), "receiving_yards": ("targets", 2), "receptions": ("targets", 2)}
            pos_props = {"QB": ["passing_yards"], "RB": ["rushing_yards", "receiving_yards"], "WR": ["receiving_yards", "receptions"], "TE": ["receiving_yards", "receptions"]}

            legs = []
            for _, p in active.iterrows():
                if str(p["player_display_name"]).strip().lower() in ctx["out_players"]:
                    continue
                t, pos = p[team_col], p["position"]
                if t not in matchup_map or pos not in pos_props:
                    continue
                if skip_bo and matchup_map[t]["blowout_prob"] >= 0.35:
                    continue
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue
                for st_col in pos_props[pos]:
                    v_col, m_vol = vol_map[st_col]
                    h = p_logs[p_logs[v_col] >= m_vol].tail(10)
                    if len(h) < 4:
                        continue
                    avg = h[st_col].mean()
                    if (st_col != "receptions" and avg < 18) or (st_col == "receptions" and avg < 3.5):
                        continue
                    res = pro_sim.simulate_nfl_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, np.floor(avg * disc) + 0.5)
                    if res["over_prob"] >= min_hr:
                        legs.append(res)

            legs.sort(key=lambda x: x["over_prob"], reverse=True)
            if legs:
                df_out = pd.DataFrame(legs)[["player", "team", "game", "stat", "line", "proj", "over_prob", "consistency", "flags"]]
                df_out["over_prob"] = df_out["over_prob"].map(lambda x: f"{x:.1%}")
                df_out["consistency"] = df_out["consistency"].map(lambda x: f"{x:.0%}")
                st.dataframe(df_out, use_container_width=True)
            else:
                st.warning("No legs met your threshold. Lower the Hit Rate % or Line Aggressiveness.")

    with mode[2]:
        c1, c2, c3 = st.columns(3)
        p_min = c1.slider("Min Hit Rate %", 20, 45, 30, key="nfl_v1") / 100.0
        p_max = c2.slider("Max Hit Rate %", 35, 60, 50, key="nfl_v2") / 100.0
        skip_bo_v = c3.checkbox("Skip Blowout-Risk Games", value=True, key="nfl_bo2")

        if st.button("Find 2x–3x Value Plays + 100/500 Report", key="nfl_val"):
            recent_pool = reg_stats[(reg_stats["season"] == ctx["season"]) & (reg_stats["week"] >= max(1, ctx["week"] - 3))]
            active = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            vol_map = {"passing_yards": ("attempts", 10), "rushing_yards": ("carries", 5), "receiving_yards": ("targets", 3), "receptions": ("targets", 3)}
            pos_props = {"QB": ["passing_yards"], "RB": ["rushing_yards", "receiving_yards"], "WR": ["receiving_yards", "receptions"], "TE": ["receiving_yards", "receptions"]}

            rows = []
            for _, p in active.iterrows():
                t, pos = p[team_col], p["position"]
                if t not in matchup_map or pos not in pos_props:
                    continue
                if skip_bo_v and matchup_map[t]["blowout_prob"] >= 0.35:
                    continue
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue
                for st_col in pos_props[pos]:
                    v_col, m_vol = vol_map[st_col]
                    h = p_logs[p_logs[v_col] >= m_vol].tail(10)
                    if len(h) < 4:
                        continue
                    rec_avg = h[st_col].tail(4).mean()
                    if (st_col != "receptions" and rec_avg < 20) or (st_col == "receptions" and rec_avg < 3.5):
                        continue
                    for mult in [1.15, 1.28]:
                        alt_line = np.floor(rec_avg * mult) + 0.5
                        res = pro_sim.simulate_nfl_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, alt_line)
                        if p_min <= res["over_prob"] <= p_max:
                            rep = pro_sim.generate_100_500_report(reg_stats, ["season", "week"], st_col, alt_line, reg_stats["player_display_name"] == res["player"])
                            rows.append({
                                "Player": res["player"],
                                "Team": res["team"],
                                "Stat": st_col,
                                "Alt Line": alt_line,
                                "Sim Prob": f"{res['over_prob']:.1%}",
                                "Fair Payout": f"{res['fair_mult']:.2f}x",
                                "Last 100g (Weighted)": f"{rep[100]['weighted_rate']:.1%}",
                                "Last 500g (Weighted)": f"{rep[500]['weighted_rate']:.1%}",
                                "Flags": res["flags"],
                            })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("No NFL props matched that 30%–50% window.")

    with mode[3]:
        st.markdown("### 🏆 Pro NFL DFS Lineup Builder (Stacking, Bring-Backs, Leverage & Multi-Lineup)")
        uploaded_csv = st.file_uploader("Optional: Upload DraftKings / FanDuel Salaries CSV", type=["csv"], key="nfl_csv")
        custom_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else None

        c1, c2, c3, c4 = st.columns(4)
        dfs_site = c1.selectbox("DFS Site", ["DraftKings ($50k)", "FanDuel ($60k)"])
        site_code = "DK" if "DraftKings" in dfs_site else "FD"
        dfs_mode = c2.selectbox("Optimization Objective", ["GPP (80th Ceiling)", "Contrarian Leverage", "Cash (Median)"])
        mode_code = {"GPP (80th Ceiling)": "gpp", "Contrarian Leverage": "leverage", "Cash (Median)": "cash"}[dfs_mode]
        stack_choice = c3.selectbox(
            "Correlation Stack Rule",
            ["QB + 1 Pass Catcher + Opp Bring-Back", "QB + 2 Pass Catchers + Opp Bring-Back", "QB + 1 Pass Catcher Only", "No Stack"],
        )
        stack_code = {
            "QB + 1 Pass Catcher + Opp Bring-Back": "qb_1_bringback",
            "QB + 2 Pass Catchers + Opp Bring-Back": "qb_2_bringback",
            "QB + 1 Pass Catcher Only": "qb_1",
            "No Stack": "none",
        }[stack_choice]
        num_lu = c4.slider("Lineups to Generate", 1, 20, 3)

        c5, c6, c7, c8 = st.columns(4)
        min_uniq = c5.slider("Min Unique Players / Lineup", 1, 5, 2)
        max_exp = c6.slider("Max Player Exposure %", 20, 100, 65) / 100.0
        no_opp_dst = c7.checkbox("Block Opposing Players vs DST", value=True)
        one_rb = c8.checkbox("Max 1 RB per Team", value=True)

        df_pool = pro_sim.build_nfl_dfs_pool(ctx, site=site_code, custom_csv_df=custom_df)
        if not df_pool.empty:
            all_names = sorted(df_pool["name"].tolist())
            lc1, lc2 = st.columns(2)
            locked = lc1.multiselect("🔒 Lock Core Players", all_names)
            excluded = lc2.multiselect("🚫 Exclude Players", [n for n in all_names if n not in locked])

            if st.button("Generate Optimal NFL DFS Lineups", key="run_nfl_dfs"):
                lineups = pro_sim.optimize_nfl_dfs_lineups(
                    df_pool,
                    site=site_code,
                    mode=mode_code,
                    num_lineups=num_lu,
                    min_unique=min_uniq,
                    max_exposure=max_exp,
                    stack_type=stack_code,
                    no_opp_vs_dst=no_opp_dst,
                    max_1_rb_per_team=one_rb,
                    locked_names=locked,
                    excluded_names=excluded,
                )
                if not lineups:
                    st.error("No valid lineups found with current Lock/Stack constraints. Try relaxing a lock or stack rule.")
                else:
                    cap = 50000 if site_code == "DK" else 60000
                    export_rows = []
                    for idx, lu in enumerate(lineups, start=1):
                        st.markdown(
                            f"**Lineup #{idx}** — Salary: `${lu['salary'].sum():,} / ${cap:,}` | "
                            f"Median: `{lu['proj'].sum():.1f} pts` | Ceiling: `{lu['ceiling'].sum():.1f} pts` | "
                            f"Cumulative Own: `{lu['est_own'].sum():.1f}%`"
                        )
                        disp = lu[["slot", "name", "pos", "team", "game", "salary", "proj", "ceiling", "est_own", "value", "flags"]]
                        st.dataframe(disp, use_container_width=True, hide_index=True)

                        qbs = lu[lu["pos"] == "QB"]["dk_id"].tolist()
                        rbs = lu[lu["slot"] == "RB"]["dk_id"].tolist()
                        wrs = lu[lu["slot"] == "WR"]["dk_id"].tolist()
                        tes = lu[lu["slot"] == "TE"]["dk_id"].tolist()
                        flx = lu[lu["slot"] == "FLEX"]["dk_id"].tolist()
                        dsts = lu[lu["pos"] == "DST"]["dk_id"].tolist()
                        if len(qbs) == 1 and len(rbs) == 2 and len(wrs) == 3 and len(tes) == 1 and len(flx) == 1 and len(dsts) == 1:
                            export_rows.append({
                                "QB": qbs[0], "RB": rbs[0], "RB.1": rbs[1],
                                "WR": wrs[0], "WR.1": wrs[1], "WR.2": wrs[2],
                                "TE": tes[0], "FLEX": flx[0], "DST": dsts[0],
                            })

                    if export_rows:
                        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download DraftKings Upload CSV",
                            data=csv_bytes,
                            file_name=f"DK_NFL_Week{ctx['week']}_Lineups.csv",
                            mime="text/csv",
                        )

            with st.expander("📋 View Full Projected Player Pool, Ownership & Leverage"):
                st.dataframe(df_pool.sort_values("leverage", ascending=False), use_container_width=True, hide_index=True)

# =====================================================================
#                             NBA WEB TAB
# =====================================================================
elif sport_tab == "🏀 NBA":
    slate_date = st.sidebar.date_input("NBA Slate Date", value=datetime.now()).strftime("%Y-%m-%d")
    ctx = get_nba(slate_date)
    ctx["nba_df"] = ensure_nba_dk_pts(ctx["nba_df"])
    nba_df = ctx["nba_df"]
    matchup_map = ctx["matchup_map"]

    st.subheader(f"NBA Season {ctx['curr_szn']} — {slate_date} ({len(ctx['todays_games'])} Games)")
    mode = st.tabs([
        "Single Player Prop",
        "Auto Parlay Builder",
        "2x–3x Value Finder (100/500 Report)",
        "🏆 NBA DFS Optimizer ($50k)",
    ])

    with mode[0]:
        c1, c2, c3, c4 = st.columns(4)
        player_in = c1.text_input("Player Name", value="Nikola Jokic")
        stat_in = c2.selectbox("Stat", ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "FG3M", "STL", "BLK", "DK_PTS"])
        line_in = c3.number_input("Prop Line", value=26.5, step=0.5)
        star_out = c4.checkbox("Star Teammate OUT (+7% Usage)")

        if st.button("Run 1,000,000 Simulations", key="nba_single"):
            p_logs = nba_df[nba_df["PLAYER_NAME"].str.contains(player_in, case=False, na=False)]
            if p_logs.empty:
                st.error(f"No NBA player found matching '{player_in}'.")
            else:
                t_abbr = p_logs.sort_values("GAME_DATE")["TEAM_ABBREVIATION"].iloc[-1]
                opp = matchup_map[t_abbr]["opponent"] if t_abbr in matchup_map else ""
                res = pro_sim.simulate_nba_prop(ctx, p_logs, opp, stat_in, line_in, manual_bump=(1.07 if star_out else 1.0))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Median Projection", f"{res['proj']:.1f}", f"Floor {res['floor']:.1f} / Ceil {res['ceiling']:.1f}")
                c2.metric(f"OVER {line_in} Prob", f"{res['over_prob']:.1%}", f"Fair Payout: {res['fair_mult']:.2f}x")
                c3.metric("Consistency Score", f"{res['consistency']:.0%}", f"{res['games_used']} healthy games")
                c4.metric("Blowout Risk", f"{res['blowout_prob']:.0%}", "Sneaky Blowout!" if res["sneaky_blowout"] else "Normal")
                st.info(f"**Matchup:** {res['game']} | **Script:** {res['script_note']} | **Flags:** {res['flags']}")

    with mode[1]:
        c1, c2, c3 = st.columns(3)
        min_hr = c1.slider("Min Hit Rate %", 55, 95, 75, key="nba_hr") / 100.0
        disc = c2.slider("Line Aggressiveness", 0.70, 1.05, 0.85, 0.05, key="nba_disc")
        skip_bo = c3.checkbox("Skip Blowout-Risk Games", value=True, key="nba_bo1")

        if st.button("Scan Today's NBA Slate", key="nba_parlay"):
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()
            legs = []
            for pid in active_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                t = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if t not in matchup_map or len(p_logs) < 5:
                    continue
                if skip_bo and matchup_map[t]["blowout_prob"] >= 0.34:
                    continue
                h = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(h) < 5 or h["MIN"].mean() < 20.0:
                    continue
                for st_col in ["PTS", "REB", "AST", "PRA"]:
                    avg = h[st_col].mean()
                    if (st_col in ["REB", "AST"] and avg < 4.5) or (st_col == "PTS" and avg < 10.0):
                        continue
                    res = pro_sim.simulate_nba_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, np.floor(avg * disc) + 0.5)
                    if res["over_prob"] >= min_hr:
                        legs.append(res)

            legs.sort(key=lambda x: x["over_prob"], reverse=True)
            if legs:
                df_out = pd.DataFrame(legs)[["player", "team", "game", "stat", "line", "proj", "over_prob", "consistency", "flags"]]
                df_out["over_prob"] = df_out["over_prob"].map(lambda x: f"{x:.1%}")
                df_out["consistency"] = df_out["consistency"].map(lambda x: f"{x:.0%}")
                st.dataframe(df_out, use_container_width=True)
            else:
                st.warning("No NBA props met your filter today.")

    with mode[2]:
        c1, c2, c3 = st.columns(3)
        p_min = c1.slider("Min Hit Rate %", 20, 45, 30, key="nba_v1") / 100.0
        p_max = c2.slider("Max Hit Rate %", 35, 60, 50, key="nba_v2") / 100.0
        skip_bo_v = c3.checkbox("Skip Blowout-Risk Games", value=True, key="nba_bo2")

        if st.button("Find 2x–3x NBA Value Plays", key="nba_val"):
            cutoff = nba_df["GAME_DATE"].max() - pd.Timedelta(days=14)
            active_ids = nba_df[nba_df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique()
            rows = []
            for pid in active_ids:
                p_logs = nba_df[nba_df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
                t = p_logs["TEAM_ABBREVIATION"].iloc[-1]
                if t not in matchup_map or len(p_logs) < 5:
                    continue
                if skip_bo_v and matchup_map[t]["blowout_prob"] >= 0.34:
                    continue
                h = p_logs[p_logs["MIN"] >= 15.0].tail(12)
                if len(h) < 5 or h["MIN"].mean() < 22.0:
                    continue
                for st_col in ["PTS", "REB", "AST", "PRA"]:
                    rec_avg = h[st_col].tail(5).mean()
                    if (st_col in ["REB", "AST"] and rec_avg < 4.5) or (st_col == "PTS" and rec_avg < 11.0):
                        continue
                    for mult in [1.12, 1.24]:
                        alt_line = np.floor(rec_avg * mult) + 0.5
                        res = pro_sim.simulate_nba_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, alt_line)
                        if p_min <= res["over_prob"] <= p_max:
                            rep = pro_sim.generate_100_500_report(nba_df, ["GAME_DATE"], st_col, alt_line, nba_df["PLAYER_NAME"] == res["player"])
                            rows.append({
                                "Player": res["player"],
                                "Team": res["team"],
                                "Stat": st_col,
                                "Alt Line": alt_line,
                                "Sim Prob": f"{res['over_prob']:.1%}",
                                "Fair Payout": f"{res['fair_mult']:.2f}x",
                                "Last 100g (Weighted)": f"{rep[100]['weighted_rate']:.1%}",
                                "Last 500g (Weighted)": f"{rep[500]['weighted_rate']:.1%}",
                                "Flags": res["flags"],
                            })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("No NBA props matched that 30%–50% window.")

    with mode[3]:
        st.markdown("### 🏆 NBA DFS Lineup Builder (8-Player Roster under $50,000)")
        nba_csv = st.file_uploader("Optional: Upload DraftKings NBA Salaries CSV", type=["csv"], key="nba_csv")
        nba_custom_df = pd.read_csv(nba_csv) if nba_csv is not None else None

        c1, c2, c3, c4 = st.columns(4)
        nba_dfs_mode = c1.selectbox("Objective", ["GPP (80th Ceiling)", "Contrarian Leverage", "Cash (Median)"], key="nba_dfs_m")
        m_code = {"GPP (80th Ceiling)": "gpp", "Contrarian Leverage": "leverage", "Cash (Median)": "cash"}[nba_dfs_mode]
        nba_num_lu = c2.slider("Lineups to Generate", 1, 15, 3, key="nba_dfs_n")
        nba_min_u = c3.slider("Min Unique Players", 1, 4, 2, key="nba_dfs_u")
        nba_max_e = c4.slider("Max Exposure %", 25, 100, 65, key="nba_dfs_e") / 100.0

        nba_pool = pro_sim.build_nba_dfs_pool(ctx, custom_csv_df=nba_custom_df)
        if not nba_pool.empty:
            nba_names = sorted(nba_pool["name"].tolist())
            lc1, lc2 = st.columns(2)
            nba_locked = lc1.multiselect("🔒 Lock NBA Core", nba_names)
            nba_excl = lc2.multiselect("🚫 Exclude NBA Players", [n for n in nba_names if n not in nba_locked])

            if st.button("Generate Optimal NBA DFS Lineups", key="run_nba_dfs"):
                lus = pro_sim.optimize_nba_dfs_lineups(
                    nba_pool,
                    mode=m_code,
                    num_lineups=nba_num_lu,
                    min_unique=nba_min_u,
                    max_exposure=nba_max_e,
                    locked_names=nba_locked,
                    excluded_names=nba_excl,
                )
                for idx, lu in enumerate(lus, start=1):
                    st.markdown(
                        f"**NBA Lineup #{idx}** — Salary: `${lu['salary'].sum():,} / $50,000` | "
                        f"Median DK Pts: `{lu['proj'].sum():.1f}` | Ceiling: `{lu['ceiling'].sum():.1f}`"
                    )
                    st.dataframe(lu[["slot", "name", "team", "game", "salary", "proj", "ceiling", "est_own", "value", "flags"]], use_container_width=True, hide_index=True)

# =====================================================================
#                     MEN'S COLLEGE BASKETBALL TAB
# =====================================================================
elif sport_tab == "🎓 Men's College Basketball":
    slate_date = st.sidebar.date_input("NCAAB Slate Date", value=datetime.now()).strftime("%Y-%m-%d")
    ctx = get_ncaab(slate_date)
    st.subheader(f"Men's College Basketball — {slate_date} ({len(ctx['todays_games'])} D1 Games)")

    if st.button("Simulate All Today's CBB Spreads, Totals & Underdog Upsets (1M Sims/Game)"):
        rows = []
        tight_dog_rows = []
        for g in ctx["game_list"]:
            res = col_sim.simulate_ncaab_game(g)
            rows.append({
                "Matchup": res["game"],
                "Proj Score": res["proj_score"],
                "Vegas Line": f"{res['home']} {res['spread_line']:+.1f}",
                "Model Fair Line": f"{res['home']} {res['comp_spread']:+.1f}",
                "Best Side": res["best_side"],
                "Cover Prob": f"{res['best_side_prob']:.1%}",
                "Edge (Pts)": round(res["spread_edge"], 1),
                "Blowout Risk": f"{res['blowout_prob']:.0%}{' ⚠️' if res['sneaky_blowout'] else ''}",
            })
            if res["is_tight_dog"]:
                tight_dog_rows.append({
                    "Underdog": res["dog_label"],
                    "Matchup": res["game"],
                    "Proj Score": res["proj_score"],
                    "Outright Win %": res["dog_win_prob"],
                    "Fair Payout": f"{res['dog_fair_mult']:.2f}x",
                    "Spread Cover %": f"{res['best_side_prob']:.1%}",
                })

        if tight_dog_rows:
            st.markdown("### 🐕 Tight-Spread Underdog Outright Win Probabilities (Spread ≤ 3.5)")
            df_dogs = pd.DataFrame(tight_dog_rows).sort_values("Outright Win %", ascending=False)
            df_dogs["Outright Win %"] = df_dogs["Outright Win %"].map(lambda x: f"{x:.1%}")
            st.dataframe(df_dogs, use_container_width=True)

        if rows:
            st.markdown("### 📊 Full D1 Slate Spread & Blowout Projections")
            df_cbb = pd.DataFrame(rows).sort_values("Edge (Pts)", ascending=False)
            st.dataframe(df_cbb, use_container_width=True)
        else:
            st.warning("No D1 games found on ESPN scoreboard for this date.")

# =====================================================================
#                        COLLEGE FOOTBALL TAB
# =====================================================================
elif sport_tab == "🏟️ College Football":
    ctx = get_ncaaf()
    st.subheader(f"College Football FBS — Season {ctx['curr_szn']} ({len(ctx['upcoming_games'])} Upcoming Games)")

    if st.button("Simulate All Upcoming CFB Spreads & Totals (1M Sims/Game)"):
        rows = []
        for g in ctx["upcoming_games"]:
            res = col_sim.simulate_cfb_game(ctx, g)
            rows.append({
                "Matchup": res["game"],
                "Proj Score": res["proj_score"],
                "Vegas Line": f"{res['home']} {res['spread_line']:+.1f}",
                "Model Fair Line": f"{res['home']} {res['comp_spread']:+.1f}",
                "Best Side": res["best_side"],
                "Cover Prob": f"{res['best_side_prob']:.1%}",
                "Edge (Pts)": round(res["spread_edge"], 1),
                "Blowout Risk": f"{res['blowout_prob']:.0%}{' ⚠️' if res['sneaky_blowout'] else ''}",
            })
        if rows:
            df_cfb = pd.DataFrame(rows).sort_values("Edge (Pts)", ascending=False)
            st.dataframe(df_cfb, use_container_width=True)
        else:
            st.warning("No upcoming FBS games found on ESPN scoreboard.")
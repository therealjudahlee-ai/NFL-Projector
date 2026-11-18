from datetime import datetime
import itertools
import numpy as np
import pandas as pd
import streamlit as st

import college_simulator as col_sim
import prop_simulator as pro_sim

st.set_page_config(
    page_title="1M-Sim Multi-Sport Projector",
    page_icon="📊",
    layout="wide",
)

st.title("📊 1,000,000-Sim Multi-Sport Prop, Spread & Parlay Engine")
st.caption("NFL • NBA • Men's College Basketball • College Football | Recency-Weighted + Sneaky Blowout Protection")


# Cache data loaders so switching tabs is instant
@st.cache_resource(show_spinner="Loading NFL 2-year stats, weather & injuries...")
def get_nfl():
    return pro_sim.build_nfl_context()


@st.cache_resource(show_spinner="Loading NBA 2-year stats & live slate...")
def get_nba(date_str):
    return pro_sim.build_nba_context(date_str)


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
    mode = st.tabs(["Single Player Prop", "Auto Parlay Builder", "2x–3x Value Finder (100/500 Report)"])

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

# =====================================================================
#                             NBA WEB TAB
# =====================================================================
elif sport_tab == "🏀 NBA":
    slate_date = st.sidebar.date_input("NBA Slate Date", value=datetime.now()).strftime("%Y-%m-%d")
    ctx = get_nba(slate_date)
    nba_df = ctx["nba_df"]
    matchup_map = ctx["matchup_map"]

    st.subheader(f"NBA Season {ctx['curr_szn']} — {slate_date} ({len(ctx['todays_games'])} Games)")
    mode = st.tabs(["Single Player Prop", "Auto Parlay Builder", "2x–3x Value Finder (100/500 Report)"])

    with mode[0]:
        c1, c2, c3, c4 = st.columns(4)
        player_in = col1_val = c1.text_input("Player Name", value="Nikola Jokic")
        stat_in = c2.selectbox("Stat", ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "FG3M", "STL", "BLK"])
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

# =====================================================================
#                     MEN'S COLLEGE BASKETBALL TAB
# =====================================================================
elif sport_tab == "🎓 Men's College Basketball":
    slate_date = st.sidebar.date_input("NCAAB Slate Date", value=datetime.now()).strftime("%Y-%m-%d")
    ctx = get_ncaab(slate_date)
    st.subheader(f"Men's College Basketball — {slate_date} ({len(ctx['todays_games'])} D1 Games)")

    if st.button("Simulate All Today's CBB Spreads & Totals (1M Sims/Game)"):
        rows = []
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
        if rows:
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
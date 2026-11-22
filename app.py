from datetime import datetime
import itertools
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

st.title("📊 1,000,000-Sim Multi-Sport Prop, Game Winner & Pro DFS Engine")
st.caption("NFL • NBA • Men's College Basketball • College Football | Recency-Weighted + Fatigue & Blowout Protection")


def ensure_nfl_anytime_td(df):
    """Guarantees anytime_td exists on cached NFL DataFrames."""
    if "anytime_td" in df.columns:
        return df
    df = df.copy()
    rush_td_s = df["rushing_tds"].fillna(0) if "rushing_tds" in df.columns else 0
    rec_td_s = df["receiving_tds"].fillna(0) if "receiving_tds" in df.columns else 0
    st_td_s = df["special_teams_tds"].fillna(0) if "special_teams_tds" in df.columns else 0
    df["anytime_td"] = rush_td_s + rec_td_s + st_td_s
    return df


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


def render_tiered_game_winners(games_list, max_tiers=6, sport_label="NFL"):
    """Renders Tier 1 / 2 / 3+ Straight-Up Game Winner (Moneyline) cards."""
    if not games_list:
        st.warning(f"No {sport_label} matchups matched your filter.")
        return

    top_games = games_list[:max_tiers]
    tier_headers = [
        "🥇 TIER 1 — HIGHEST-CONFIDENCE STRAIGHT-UP WINNER",
        "🥈 TIER 2 — PRIME SITUATIONAL / EFFICIENCY FAVORITE",
        "🥉 TIER 3 — SOLID MONEYLINE LEAN",
    ]

    st.markdown(f"### 🏆 Tiered {sport_label} Game Winner Projections (Top {len(top_games)} Ranked)")

    for idx, g in enumerate(top_games, start=1):
        tier_title = tier_headers[idx - 1] if idx <= 3 else f"🎯 TIER {idx} — MATCHUP LEAN"
        with st.container(border=True):
            st.caption(f"**{tier_title}** • {g['Matchup']}")
            c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.3])
            c1.markdown(f"#### #{idx} {g['Projected Winner']} TO WIN")
            c2.metric("Win Probability", f"{g['Win Prob']:.1%}", f"Fair ML: {g['Fair Moneyline']}")
            c3.metric("Fair Payout", g["Fair Payout"], f"Home: {g['Home Win %']}")
            c4.metric("Projected Score", g["Proj Score"], f"Away: {g['Away Win %']}")
            st.caption(f"**Context & Situational Factors:** {g['Flags']}")

    with st.expander(f"📋 View Full {sport_label} Slate Moneyline Board"):
        df_g = pd.DataFrame(games_list).copy()
        df_g["Win Prob"] = df_g["Win Prob"].map(lambda x: f"{x:.1%}")
        cols = ["Matchup", "Projected Winner", "Win Prob", "Fair Moneyline", "Fair Payout", "Proj Score", "Flags"]
        st.dataframe(df_g[cols], use_container_width=True, hide_index=True)


def render_parlay_tickets(legs, n_legs, max_parlays, is_single_game=False, sport_label="NFL"):
    """Builds and renders tiered, cleanly separated Parlay Ticket Cards."""
    best_per_player = []
    seen_p = set()
    for lg in sorted(legs, key=lambda x: x["over_prob"], reverse=True):
        if lg["player"] not in seen_p:
            seen_p.add(lg["player"])
            best_per_player.append(lg)

    if len(best_per_player) < n_legs:
        st.warning(
            f"Found {len(best_per_player)} distinct player leg(s), which is fewer than the {n_legs} legs needed per parlay. "
            "Try lowering Min Hit Rate % or Line Aggressiveness."
        )
        return

    candidate_pool = best_per_player[:20]
    scored_combos = []

    for combo in itertools.combinations(candidate_pool, n_legs):
        if is_single_game and len({x["game"] for x in combo}) > 1:
            continue

        raw_prob = float(np.prod([x["over_prob"] for x in combo]))
        corr_boost = 1.0
        is_correlated = False

        if sport_label == "NFL":
            for qb_leg in [x for x in combo if x.get("position") == "QB" and x["stat"] == "passing_yards"]:
                has_stack = any(
                    x["team"] == qb_leg["team"]
                    and x.get("position") in ["WR", "TE"]
                    and x["stat"] in ["receiving_yards", "receptions"]
                    for x in combo
                )
                if has_stack:
                    corr_boost = 1.06
                    is_correlated = True
                    break

        adj_prob = min(0.95, raw_prob * corr_boost)
        avg_consistency = float(np.mean([x["consistency"] for x in combo]))
        rank_score = adj_prob * (0.85 + 0.15 * avg_consistency)
        scored_combos.append((rank_score, adj_prob, is_correlated, combo))

    scored_combos.sort(key=lambda item: item[0], reverse=True)

    selected_tickets = []
    player_usage = {}
    max_reuse = 2 if is_single_game else 1

    for rank_score, adj_prob, is_corr, combo in scored_combos:
        if any(player_usage.get(x["player"], 0) >= max_reuse for x in combo):
            continue
        for x in combo:
            player_usage[x["player"]] = player_usage.get(x["player"], 0) + 1
        selected_tickets.append((adj_prob, is_corr, combo))
        if len(selected_tickets) >= max_parlays:
            break

    if len(selected_tickets) < max_parlays:
        used_sigs = {tuple(sorted(x["player"] for x in t[2])) for t in selected_tickets}
        for rank_score, adj_prob, is_corr, combo in scored_combos:
            sig = tuple(sorted(x["player"] for x in combo))
            if sig in used_sigs:
                continue
            used_sigs.add(sig)
            selected_tickets.append((adj_prob, is_corr, combo))
            if len(selected_tickets) >= max_parlays:
                break

    if not selected_tickets:
        st.warning("Could not form valid parlays from the qualifying legs. Try reducing Legs per Parlay.")
        return

    tier_labels = [
        "🥇 TIER 1 — TOP ANCHOR PARLAY (Highest Hit Probability)",
        "🥈 TIER 2 — PRIME CORRELATED / VALUE PARLAY",
        "🥉 TIER 3 — DIVERSIFIED SECONDARY PARLAY",
    ]

    st.markdown(f"### 🎟️ Tiered Recommended {n_legs}-Leg Parlays ({len(selected_tickets)} Tickets)")

    for idx, (comb_prob, is_corr, combo) in enumerate(selected_tickets, start=1):
        fair_payout = round(1.0 / max(comb_prob, 0.01), 2)
        matchup_title = combo[0]["game"] if is_single_game else f"{len({x['game'] for x in combo})}-Game Slate"
        tier_title = tier_labels[idx - 1] if idx <= 3 else f"🎯 TIER {idx} — ALTERNATE COMBINATION"
        stack_badge = " • 🔗 QB+Receiver Correlated" if is_corr else ""

        with st.container(border=True):
            st.caption(f"**{tier_title}**{stack_badge}")
            h1, h2, h3 = st.columns([2.2, 1, 1])
            h1.markdown(f"#### 🎫 Ticket #{idx} — {matchup_title}")
            h2.metric("Combined Hit Prob", f"{comb_prob:.1%}")
            h3.metric("Fair Payout", f"{fair_payout:.2f}x")

            ticket_rows = []
            for leg_i, lg in enumerate(combo, start=1):
                ticket_rows.append({
                    "Leg": f"Leg {leg_i}",
                    "Player": lg["player"],
                    "Pos": lg.get("position", "-"),
                    "Team": lg["team"],
                    "Matchup": lg["game"],
                    "Pick": f"OVER {lg['line']} {lg['stat'].upper()}",
                    "Median Proj": round(lg["proj"], 1),
                    "Hit Prob": f"{lg['over_prob']:.1%}",
                    "Consistency": f"{lg['consistency']:.0%}",
                    "Flags": lg["flags"],
                })
            st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True, hide_index=True)


def render_tiered_value_plays(candidates, max_tiers=6, sport_label="NFL"):
    """Deduplicates to 1 best alt-line play per player and renders Tier 1 / 2 / 3+ cards."""
    if not candidates:
        st.warning(f"No {sport_label} props matched that 2x–3x probability window. Try widening the Min/Max Hit Rate sliders.")
        return

    sorted_cands = sorted(candidates, key=lambda x: x["value_score"], reverse=True)
    unique_plays = []
    seen_players = set()
    for item in sorted_cands:
        if item["Player"] not in seen_players:
            seen_players.add(item["Player"])
            unique_plays.append(item)

    top_plays = unique_plays[:max_tiers]
    tier_headers = [
        "🥇 TIER 1 — TOP 2x–3x ANCHOR VALUE PLAY",
        "🥈 TIER 2 — PRIME MATCHUP & FORM EDGE",
        "🥉 TIER 3 — HIGH-UPSIDE VALUE PLAY",
    ]

    st.markdown(f"### 💎 Tiered {sport_label} 2x–3x Multiplier Plays (Top {len(top_plays)} Ranked)")

    for idx, play in enumerate(top_plays, start=1):
        tier_title = tier_headers[idx - 1] if idx <= 3 else f"🎯 TIER {idx} — SECONDARY VALUE PLAY"
        with st.container(border=True):
            st.caption(f"**{tier_title}** • Matchup: {play['Matchup']}")
            c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
            c1.markdown(f"#### #{idx} {play['Player']} ({play['Team']}) — OVER {play['Alt Line']} {play['Stat'].upper()}")
            c2.metric("Sim Hit Prob", play["Sim Prob"], f"Proj: {play['Median Proj']}")
            c3.metric("Fair Payout", play["Fair Payout"], f"100g: {play['Last 100g (Weighted)']}")
            c4.metric("500g Weighted", play["Last 500g (Weighted)"], f"Form: {play['Form Trend']}")
            st.caption(f"**Context Flags:** {play['Flags']}")

    with st.expander("📋 View All Qualifying 2x–3x Value Candidates (Full Table)"):
        disp_cols = [
            "Player", "Team", "Matchup", "Stat", "Alt Line", "Median Proj",
            "Sim Prob", "Fair Payout", "Last 100g (Weighted)", "Last 500g (Weighted)", "Flags",
        ]
        st.dataframe(pd.DataFrame(unique_plays)[disp_cols], use_container_width=True, hide_index=True)


def render_tiered_attd_plays(candidates, max_tiers=6):
    """Renders Tier 1 / 2 / 3+ Anytime Touchdown Scorer cards with Sim Prob & Fair Payout."""
    if not candidates:
        st.warning("No Anytime TD scorers matched your minimum probability threshold. Try lowering Min Anytime TD %.")
        return

    sorted_cands = sorted(candidates, key=lambda x: x["rank_score"], reverse=True)
    top_plays = sorted_cands[:max_tiers]
    tier_headers = [
        "🥇 TIER 1 — TOP ANCHOR ANYTIME TD SCORER (Highest Scoring Probability)",
        "🥈 TIER 2 — PRIME RED-ZONE & MATCHUP TD PLAY",
        "🥉 TIER 3 — HIGH-UPSIDE / PLUS-MONEY TD VALUE",
    ]

    st.markdown(f"### 🏈 Tiered Anytime Touchdown Scorers (Top {len(top_plays)} Ranked)")

    for idx, play in enumerate(top_plays, start=1):
        tier_title = tier_headers[idx - 1] if idx <= 3 else f"🎯 TIER {idx} — SECONDARY ANYTIME TD PLAY"
        with st.container(border=True):
            st.caption(f"**{tier_title}** • Matchup: {play['Matchup']}")
            c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
            c1.markdown(f"#### #{idx} {play['Player']} ({play['Pos']} — {play['Team']}) — ANYTIME TD")
            c2.metric("Sim ATTD Prob", play["Sim Prob"], f"Exp TDs: {play['Exp TDs']}")
            c3.metric("Fair Payout", play["Fair Payout"], f"100g: {play['Last 100g (Weighted)']}")
            c4.metric("500g Weighted", play["Last 500g (Weighted)"], f"Def Trend: {play['Def Trend']}")
            st.caption(f"**Context Flags:** {play['Flags']}")

    with st.expander("📋 View All Qualifying Anytime TD Scorers (Full Table)"):
        disp_cols = [
            "Player", "Pos", "Team", "Matchup", "Exp TDs",
            "Sim Prob", "Fair Payout", "Last 100g (Weighted)", "Last 500g (Weighted)", "Flags",
        ]
        st.dataframe(pd.DataFrame(sorted_cands)[disp_cols], use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner="Loading NFL 2-year stats, live rosters, depth charts & IR/injuries...")
def get_nfl():
    ctx = pro_sim.build_nfl_context()
    ctx["reg_stats"] = ensure_nfl_anytime_td(ctx["reg_stats"])
    return ctx


@st.cache_resource(show_spinner="Loading NBA 2-year stats & live slate...")
def get_nba(date_str):
    ctx = pro_sim.build_nba_context(date_str)
    ctx["nba_df"] = ensure_nba_dk_pts(ctx["nba_df"])
    return ctx


@st.cache_resource(show_spinner="Loading Men's CBB 2-year SOS-adjusted stats & D1 slate...")
def get_ncaab(date_str):
    return col_sim.build_ncaab_context(date_str)


@st.cache_resource(show_spinner="Loading College Football 2-year stats & FBS matchups...")
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
    ctx["reg_stats"] = ensure_nfl_anytime_td(ctx["reg_stats"])
    reg_stats = ctx["reg_stats"]
    team_col = ctx["team_col"]
    matchup_map = ctx["matchup_map"]

    st.subheader(f"NFL Season {ctx['season']} — Week {ctx['week']}")
    mode = st.tabs([
        "🏆 Game Winners (Moneyline)",
        "Single Player Prop",
        "Auto Multi-Leg Parlay Builder",
        "2x–3x Value Finder (100/500 Report)",
        "🏈 Anytime Touchdown Finder",
        "Pro NFL DFS Optimizer",
    ])

    with mode[0]:
        st.markdown("### 🏆 NFL Straight-Up Game Winner Projections (1,000,000 Sims / Game)")
        c1, c2, c3 = st.columns(3)
        nfl_min_wp = c1.slider("Min Win Probability %", 50, 85, 52, key="nfl_gw_min") / 100.0
        nfl_gw_tiers = c2.slider("Tiered Games to Show", 2, 16, 6, key="nfl_gw_tiers")
        nfl_gw_filter = c3.selectbox("Slate Filter", ["Full Week Slate", "🌙 Thursday Night Football Only"], key="nfl_gw_filt")

        if st.button("🚀 Simulate Week's NFL Game Winners", key="run_nfl_gw", type="primary"):
            g_res = pro_sim.simulate_nfl_games(ctx)
            if "Thursday" in nfl_gw_filter:
                g_res = [g for g in g_res if g.get("Is TNF")]
            g_res = [g for g in g_res if g["Win Prob"] >= nfl_min_wp]
            render_tiered_game_winners(g_res, max_tiers=nfl_gw_tiers, sport_label="NFL")

    with mode[1]:
        col1, col2, col3, col4 = st.columns(4)
        player_in = col1.text_input("Player Name", value="Josh Allen")
        stat_in = col2.selectbox(
            "Stat",
            ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "anytime_td", "dk_points", "passing_tds", "rushing_tds", "receiving_tds"],
        )
        line_in = col3.number_input("Prop Line", value=0.5 if stat_in == "anytime_td" else 245.5, step=0.5)
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
                c1.metric("Median / Exp Proj", f"{res['proj']:.2f}", f"Floor {res['floor']:.1f} / Ceil {res['ceiling']:.1f}")
                c2.metric(f"OVER {line_in} Prob", f"{res['over_prob']:.1%}", f"Fair Payout: {res['fair_mult']:.2f}x")
                c3.metric("Consistency Score", f"{res['consistency']:.0%}", f"{res['games_used']} healthy games")
                c4.metric("Blowout Risk", f"{res['blowout_prob']:.0%}", "Sneaky Blowout!" if res["sneaky_blowout"] else "Normal")
                st.info(f"**Matchup:** {res['game']} | **Script:** {res['script_note']} | **Flags:** {res['flags']}")

    with mode[2]:
        st.markdown("### 🎰 Tiered NFL Multi-Leg Parlay & TNF SGP Builder")
        c1, c2, c3, c4, c5 = st.columns(5)
        min_hr = c1.slider("Min Hit Rate % / Leg", 55, 95, 72, key="nfl_hr") / 100.0
        n_legs = c2.selectbox("Legs per Parlay", [2, 3, 4, 5], index=1, key="nfl_nlegs")
        max_tickets = c3.slider("Tiered Tickets to Show", 1, 8, 4, key="nfl_max_tix")
        disc = c4.slider("Line Aggressiveness", 0.70, 1.05, 0.85, 0.05, key="nfl_disc")
        skip_bo = c5.checkbox("Skip Blowout Games", value=False, key="nfl_bo1")

        slate_scope = st.radio(
            "Game Filter",
            ["🌙 Thursday Night Football / Single-Game SGP", "🏈 Full Week Slate"],
            horizontal=True,
            key="nfl_parlay_scope",
        )

        tnf_teams = {t for t, info in matchup_map.items() if info.get("is_thursday")}
        all_game_labels = sorted({info["game_label"] for info in matchup_map.values()})
        is_tnf_mode = "Thursday" in slate_scope
        selected_single_game = None

        if is_tnf_mode:
            if tnf_teams:
                selected_single_game = next(matchup_map[t]["game_label"] for t in tnf_teams)
                st.info(f"**Tonight's TNF Matchup:** {selected_single_game} (Short-week road fatigue & live IR filters active)")
            elif all_game_labels:
                selected_single_game = st.selectbox(
                    "Select Standalone / Primetime Game for Single-Game Parlay (SGP):",
                    all_game_labels,
                    key="nfl_sgp_game_pick",
                )
            else:
                st.warning("No games found in this week's schedule.")

        if st.button("🚀 Build Tiered NFL Parlays", key="nfl_parlay", type="primary"):
            recent_pool = reg_stats[
                (reg_stats["season"] == ctx["season"]) & (reg_stats["week"] >= max(1, ctx["week"] - 4))
            ]
            active = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            vol_map = {
                "passing_yards": ("attempts", 8),
                "rushing_yards": ("carries", 4),
                "receiving_yards": ("targets", 2),
                "receptions": ("targets", 2),
            }
            pos_props = {
                "QB": ["passing_yards", "rushing_yards"],
                "RB": ["rushing_yards", "receiving_yards"],
                "WR": ["receiving_yards", "receptions"],
                "TE": ["receiving_yards", "receptions"],
            }

            legs = []
            for _, p in active.iterrows():
                p_norm = pro_sim._norm_name(p["player_display_name"])
                if p_norm in ctx["out_players"]:
                    continue

                if ctx.get("active_roster_team"):
                    if p_norm not in ctx["active_roster_team"]:
                        continue
                    t = ctx["active_roster_team"][p_norm]
                else:
                    t = p[team_col]

                pos = p["position"]
                if t not in matchup_map or pos not in pos_props:
                    continue
                if is_tnf_mode and selected_single_game and matchup_map[t]["game_label"] != selected_single_game:
                    continue
                if skip_bo and matchup_map[t]["blowout_prob"] >= 0.35:
                    continue

                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 3:
                    continue

                if pos == "QB":
                    recent_att = p_logs["attempts"].tail(4).mean() if "attempts" in p_logs.columns else 0
                    if recent_att < 20.0:
                        continue

                for st_col in pos_props[pos]:
                    v_col, m_vol = vol_map[st_col]
                    h = p_logs[p_logs[v_col] >= m_vol].tail(10)
                    if len(h) < 4:
                        continue
                    avg = h[st_col].mean()
                    if (st_col != "receptions" and avg < 16) or (st_col == "receptions" and avg < 3.0):
                        continue
                    res = pro_sim.simulate_nfl_prop(
                        ctx,
                        p_logs,
                        matchup_map[t]["opponent"],
                        st_col,
                        np.floor(avg * disc) + 0.5,
                        override_team=t,
                    )
                    if res["over_prob"] >= min_hr:
                        legs.append(res)

            legs.sort(key=lambda x: x["over_prob"], reverse=True)
            render_parlay_tickets(legs, n_legs=n_legs, max_parlays=max_tickets, is_single_game=is_tnf_mode, sport_label="NFL")

            if legs:
                with st.expander("📋 View All Qualifying Individual Player Legs"):
                    df_out = pd.DataFrame(legs)[["player", "position", "team", "game", "stat", "line", "proj", "over_prob", "consistency", "flags"]]
                    df_out["over_prob"] = df_out["over_prob"].map(lambda x: f"{x:.1%}")
                    df_out["consistency"] = df_out["consistency"].map(lambda x: f"{x:.0%}")
                    st.dataframe(df_out, use_container_width=True, hide_index=True)

    with mode[3]:
        st.markdown("### 💎 Tiered NFL 2x–3x Multiplier Value Finder")
        c1, c2, c3, c4, c5 = st.columns(5)
        p_min = c1.slider("Min Hit Rate %", 20, 45, 30, key="nfl_v1") / 100.0
        p_max = c2.slider("Max Hit Rate %", 35, 60, 50, key="nfl_v2") / 100.0
        nfl_val_tiers = c3.slider("Tiered Plays to Show", 2, 10, 6, key="nfl_v_tiers")
        val_scope = c4.selectbox("Game Filter", ["Full Week Slate", "🌙 Thursday Night Football / Single Game"], key="nfl_val_scope")
        skip_bo_v = c5.checkbox("Skip Blowout Games", value=False, key="nfl_bo2")

        tnf_teams_v = {t for t, info in matchup_map.items() if info.get("is_thursday")}
        all_game_labels_v = sorted({info["game_label"] for info in matchup_map.values()})
        val_single_game = None
        if "Thursday" in val_scope:
            if tnf_teams_v:
                val_single_game = next(matchup_map[t]["game_label"] for t in tnf_teams_v)
            elif all_game_labels_v:
                val_single_game = st.selectbox("Select Game for Value Finder:", all_game_labels_v, key="nfl_val_game_pick")

        if st.button("🚀 Find Tiered NFL 2x–3x Value Plays", key="nfl_val", type="primary"):
            recent_pool = reg_stats[(reg_stats["season"] == ctx["season"]) & (reg_stats["week"] >= max(1, ctx["week"] - 4))]
            active = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            vol_map = {"passing_yards": ("attempts", 10), "rushing_yards": ("carries", 5), "receiving_yards": ("targets", 3), "receptions": ("targets", 3)}
            pos_props = {"QB": ["passing_yards"], "RB": ["rushing_yards", "receiving_yards"], "WR": ["receiving_yards", "receptions"], "TE": ["receiving_yards", "receptions"]}

            rows = []
            for _, p in active.iterrows():
                p_norm = pro_sim._norm_name(p["player_display_name"])
                if p_norm in ctx["out_players"]:
                    continue
                t = ctx["active_roster_team"].get(p_norm, p[team_col]) if ctx.get("active_roster_team") else p[team_col]
                pos = p["position"]
                if t not in matchup_map or pos not in pos_props:
                    continue
                if "Thursday" in val_scope and val_single_game and matchup_map[t]["game_label"] != val_single_game:
                    continue
                if skip_bo_v and matchup_map[t]["blowout_prob"] >= 0.35:
                    continue
                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 4:
                    continue
                if pos == "QB":
                    recent_att = p_logs["attempts"].tail(4).mean() if "attempts" in p_logs.columns else 0
                    if recent_att < 20.0:
                        continue
                for st_col in pos_props[pos]:
                    v_col, m_vol = vol_map[st_col]
                    h = p_logs[p_logs[v_col] >= m_vol].tail(10)
                    if len(h) < 4:
                        continue
                    rec_avg = h[st_col].tail(4).mean()
                    two_yr_avg = h[st_col].mean()
                    if (st_col != "receptions" and rec_avg < 20) or (st_col == "receptions" and rec_avg < 3.5):
                        continue
                    for mult in [1.15, 1.28]:
                        alt_line = np.floor(rec_avg * mult) + 0.5
                        res = pro_sim.simulate_nfl_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, alt_line, override_team=t)
                        if p_min <= res["over_prob"] <= p_max:
                            rep = pro_sim.generate_100_500_report(reg_stats, ["season", "week"], st_col, alt_line, reg_stats["player_display_name"] == res["player"])
                            form_ratio = rec_avg / max(two_yr_avg, 1.0)
                            edge_score = float(res["over_prob"] * res["def_mult"] * res["script_mult"] * (0.9 + 0.1 * form_ratio))
                            rows.append({
                                "value_score": edge_score,
                                "Player": res["player"],
                                "Team": res["team"],
                                "Matchup": res["game"],
                                "Stat": st_col,
                                "Alt Line": alt_line,
                                "Median Proj": round(res["proj"], 1),
                                "Sim Prob": f"{res['over_prob']:.1%}",
                                "Fair Payout": f"{res['fair_mult']:.2f}x",
                                "Form Trend": f"{form_ratio:.2f}x",
                                "Last 100g (Weighted)": f"{rep[100]['weighted_rate']:.1%}",
                                "Last 500g (Weighted)": f"{rep[500]['weighted_rate']:.1%}",
                                "Flags": res["flags"],
                            })
            render_tiered_value_plays(rows, max_tiers=nfl_val_tiers, sport_label="NFL")

    with mode[4]:
        st.markdown("### 🏈 Tiered NFL Anytime Touchdown Scorer Engine (1,000,000 Sims)")
        c1, c2, c3, c4 = st.columns(4)
        min_td_prob = c1.slider("Min Anytime TD %", 15, 70, 32, key="attd_min") / 100.0
        attd_tiers = c2.slider("Tiered TD Scorers to Show", 2, 12, 6, key="attd_tiers")
        attd_scope = c3.selectbox(
            "Game Filter",
            ["🌙 Thursday Night Football / Single Game", "🏈 Full Week Slate"],
            key="attd_scope",
        )
        include_qb_rush = c4.checkbox("Include Rushing QBs", value=True, key="attd_qb")

        tnf_teams_td = {t for t, info in matchup_map.items() if info.get("is_thursday")}
        all_game_labels_td = sorted({info["game_label"] for info in matchup_map.values()})
        attd_single_game = None
        if "Thursday" in attd_scope:
            if tnf_teams_td:
                attd_single_game = next(matchup_map[t]["game_label"] for t in tnf_teams_td)
                st.info(f"**Tonight's TNF Matchup:** {attd_single_game} (Poisson red-zone & live IR filters active)")
            elif all_game_labels_td:
                attd_single_game = st.selectbox("Select Game for Anytime TD Scorers:", all_game_labels_td, key="attd_game_pick")

        if st.button("🚀 Find Tiered Anytime TD Scorers", key="run_attd", type="primary"):
            recent_pool = reg_stats[
                (reg_stats["season"] == ctx["season"]) & (reg_stats["week"] >= max(1, ctx["week"] - 4))
            ]
            active = recent_pool[["player_id", "player_display_name", "position", team_col]].drop_duplicates(subset=["player_id"])
            allowed_pos = ["RB", "WR", "TE"] + (["QB"] if include_qb_rush else [])

            td_rows = []
            for _, p in active.iterrows():
                p_norm = pro_sim._norm_name(p["player_display_name"])
                if p_norm in ctx["out_players"]:
                    continue
                t = ctx["active_roster_team"].get(p_norm, p[team_col]) if ctx.get("active_roster_team") else p[team_col]
                pos = p["position"]
                if t not in matchup_map or pos not in allowed_pos:
                    continue
                if "Thursday" in attd_scope and attd_single_game and matchup_map[t]["game_label"] != attd_single_game:
                    continue

                p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values("sort_key")
                if len(p_logs) < 3:
                    continue

                if pos == "QB":
                    recent_att = p_logs["attempts"].tail(4).mean() if "attempts" in p_logs.columns else 0
                    recent_rush_td = p_logs["rushing_tds"].tail(6).mean() if "rushing_tds" in p_logs.columns else 0
                    if recent_att < 20.0 or recent_rush_td < 0.20:
                        continue
                else:
                    if p_logs["dk_points"].tail(6).mean() < 5.0:
                        continue

                res = pro_sim.simulate_nfl_prop(ctx, p_logs, matchup_map[t]["opponent"], "anytime_td", 0.5, override_team=t)
                if res["over_prob"] >= min_td_prob:
                    rep = pro_sim.generate_100_500_report(
                        reg_stats, ["season", "week"], "anytime_td", 0.5, reg_stats["player_display_name"] == res["player"]
                    )
                    rank_sc = float(res["over_prob"] * (0.85 + 0.15 * res["def_mult"]))
                    td_rows.append({
                        "rank_score": rank_sc,
                        "Player": res["player"],
                        "Pos": pos,
                        "Team": res["team"],
                        "Matchup": res["game"],
                        "Exp TDs": f"{res['proj']:.2f}",
                        "Sim Prob": f"{res['over_prob']:.1%}",
                        "Fair Payout": f"{res['fair_mult']:.2f}x",
                        "Def Trend": f"{res['def_mult']:.2f}x",
                        "Last 100g (Weighted)": f"{rep[100]['weighted_rate']:.1%}",
                        "Last 500g (Weighted)": f"{rep[500]['weighted_rate']:.1%}",
                        "Flags": res["flags"],
                    })
            render_tiered_attd_plays(td_rows, max_tiers=attd_tiers)

    with mode[5]:
        st.markdown("### 🏆 Pro NFL DFS Lineup Builder (Stacking, Bring-Backs, Leverage & Multi-Lineup)")
        uploaded_csv = st.file_uploader(
            "Upload DraftKings ('DKSalaries.csv') or FanDuel ('FanDuel-NFL.csv') for 100% Exact Slate Prices",
            type=["csv"],
            key="nfl_csv",
        )
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
        all_names = sorted(df_pool["name"].tolist()) if not df_pool.empty else []

        if not df_pool.empty and "salary_source" in df_pool.columns:
            src_txt = df_pool["salary_source"].iloc[0]
            if "Exact" in src_txt and "Estimated" not in src_txt:
                st.success(f"**Salary Data Source:** {src_txt}")
            else:
                st.warning(
                    f"**Salary Data Source:** {src_txt} — DraftKings/FanDuel block automated cloud scraping, "
                    "so upload your contest CSV above whenever you want exact down-to-the-dollar slate salaries."
                )

        lc1, lc2 = st.columns(2)
        locked = lc1.multiselect("🔒 Lock Core Players", all_names, key="nfl_lock")
        excluded = lc2.multiselect("🚫 Exclude Players", [n for n in all_names if n not in locked], key="nfl_excl")

        st.caption(f"Player Pool Loaded: **{len(df_pool)}** active players across **{len(matchup_map) // 2}** Week {ctx['week']} games.")

        if st.button("🚀 Generate Optimal NFL DFS Lineups", key="run_nfl_dfs", type="primary"):
            if df_pool.empty:
                st.warning("No NFL players in pool for this week's slate (or uploaded CSV names didn't match).")
            else:
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
                        with st.container(border=True):
                            st.markdown(
                                f"#### 🏆 Optimal {site_code} Lineup #{idx} — Salary: `${lu['salary'].sum():,} / ${cap:,}` | "
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

        if not df_pool.empty:
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
        "🏆 Game Winners (Moneyline)",
        "Single Player Prop",
        "Auto Multi-Leg Parlay Builder",
        "2x–3x Value Finder (100/500 Report)",
        "NBA DFS Optimizer ($50k)",
    ])

    with mode[0]:
        st.markdown("### 🏆 NBA Straight-Up Game Winner Projections (Pace, B2B Fatigue & Star-Out Adjusted)")
        c1, c2 = st.columns(2)
        nba_min_wp = c1.slider("Min Win Probability %", 50, 85, 53, key="nba_gw_min") / 100.0
        nba_gw_tiers = c2.slider("Tiered Games to Show", 2, 15, 6, key="nba_gw_tiers")

        if st.button("🚀 Simulate Today's NBA Game Winners", key="run_nba_gw", type="primary"):
            g_res = pro_sim.simulate_nba_games(ctx)
            g_res = [g for g in g_res if g["Win Prob"] >= nba_min_wp]
            render_tiered_game_winners(g_res, max_tiers=nba_gw_tiers, sport_label="NBA")

    with mode[1]:
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

    with mode[2]:
        st.markdown("### 🎰 Tiered NBA Multi-Leg Parlay Builder")
        c1, c2, c3, c4, c5 = st.columns(5)
        min_hr = c1.slider("Min Hit Rate % / Leg", 55, 95, 75, key="nba_hr") / 100.0
        nba_n_legs = c2.selectbox("Legs per Parlay", [2, 3, 4, 5], index=1, key="nba_nlegs")
        nba_max_tix = c3.slider("Tiered Tickets to Show", 1, 8, 4, key="nba_max_tix")
        disc = c4.slider("Line Aggressiveness", 0.70, 1.05, 0.85, 0.05, key="nba_disc")
        skip_bo = c5.checkbox("Skip Blowout Games", value=True, key="nba_bo1")

        if st.button("🚀 Build Tiered NBA Parlays", key="nba_parlay", type="primary"):
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
            render_parlay_tickets(legs, n_legs=nba_n_legs, max_parlays=nba_max_tix, is_single_game=False, sport_label="NBA")

            if legs:
                with st.expander("📋 View All Qualifying Individual NBA Player Legs"):
                    df_out = pd.DataFrame(legs)[["player", "team", "game", "stat", "line", "proj", "over_prob", "consistency", "flags"]]
                    df_out["over_prob"] = df_out["over_prob"].map(lambda x: f"{x:.1%}")
                    df_out["consistency"] = df_out["consistency"].map(lambda x: f"{x:.0%}")
                    st.dataframe(df_out, use_container_width=True, hide_index=True)

    with mode[3]:
        st.markdown("### 💎 Tiered NBA 2x–3x Multiplier Value Finder")
        c1, c2, c3, c4 = st.columns(4)
        p_min = c1.slider("Min Hit Rate %", 20, 45, 30, key="nba_v1") / 100.0
        p_max = c2.slider("Max Hit Rate %", 35, 60, 50, key="nba_v2") / 100.0
        nba_val_tiers = c3.slider("Tiered Plays to Show", 2, 10, 6, key="nba_v_tiers")
        skip_bo_v = c4.checkbox("Skip Blowout-Risk Games", value=True, key="nba_bo2")

        if st.button("🚀 Find Tiered NBA 2x–3x Value Plays", key="nba_val", type="primary"):
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
                    two_yr_avg = h[st_col].mean()
                    if (st_col in ["REB", "AST"] and rec_avg < 4.5) or (st_col == "PTS" and rec_avg < 11.0):
                        continue
                    for mult in [1.12, 1.24]:
                        alt_line = np.floor(rec_avg * mult) + 0.5
                        res = pro_sim.simulate_nba_prop(ctx, p_logs, matchup_map[t]["opponent"], st_col, alt_line)
                        if p_min <= res["over_prob"] <= p_max:
                            rep = pro_sim.generate_100_500_report(nba_df, ["GAME_DATE"], st_col, alt_line, nba_df["PLAYER_NAME"] == res["player"])
                            form_ratio = rec_avg / max(two_yr_avg, 1.0)
                            edge_score = float(res["over_prob"] * res["def_mult"] * res["script_mult"] * (0.9 + 0.1 * form_ratio))
                            rows.append({
                                "value_score": edge_score,
                                "Player": res["player"],
                                "Team": res["team"],
                                "Matchup": res["game"],
                                "Stat": st_col,
                                "Alt Line": alt_line,
                                "Median Proj": round(res["proj"], 1),
                                "Sim Prob": f"{res['over_prob']:.1%}",
                                "Fair Payout": f"{res['fair_mult']:.2f}x",
                                "Form Trend": f"{form_ratio:.2f}x",
                                "Last 100g (Weighted)": f"{rep[100]['weighted_rate']:.1%}",
                                "Last 500g (Weighted)": f"{rep[500]['weighted_rate']:.1%}",
                                "Flags": res["flags"],
                            })
            render_tiered_value_plays(rows, max_tiers=nba_val_tiers, sport_label="NBA")

    with mode[4]:
        st.markdown("### 🏆 NBA DFS Lineup Builder (8-Player Roster under $50,000)")
        nba_csv = st.file_uploader(
            "Upload DraftKings ('DKSalaries.csv') or FanDuel NBA CSV for 100% Exact Slate Prices",
            type=["csv"],
            key="nba_csv",
        )
        nba_custom_df = pd.read_csv(nba_csv) if nba_csv is not None else None

        c1, c2, c3, c4 = st.columns(4)
        nba_dfs_mode = c1.selectbox("Objective", ["GPP (80th Ceiling)", "Contrarian Leverage", "Cash (Median)"], key="nba_dfs_m")
        m_code = {"GPP (80th Ceiling)": "gpp", "Contrarian Leverage": "leverage", "Cash (Median)": "cash"}[nba_dfs_mode]
        nba_num_lu = c2.slider("Lineups to Generate", 1, 15, 3, key="nba_dfs_n")
        nba_min_u = c3.slider("Min Unique Players", 1, 4, 2, key="nba_dfs_u")
        nba_max_e = c4.slider("Max Exposure %", 25, 100, 65, key="nba_dfs_e") / 100.0

        nba_pool = pro_sim.build_nba_dfs_pool(ctx, custom_csv_df=nba_custom_df)
        nba_names = sorted(nba_pool["name"].tolist()) if not nba_pool.empty else []

        if not nba_pool.empty and "salary_source" in nba_pool.columns:
            src_txt = nba_pool["salary_source"].iloc[0]
            if "Exact" in src_txt and "Estimated" not in src_txt:
                st.success(f"**Salary Data Source:** {src_txt}")
            else:
                st.warning(
                    f"**Salary Data Source:** {src_txt} — Upload your DraftKings/FanDuel NBA CSV above for exact slate salaries."
                )

        lc1, lc2 = st.columns(2)
        nba_locked = lc1.multiselect("🔒 Lock NBA Core", nba_names, key="nba_lock")
        nba_excl = lc2.multiselect("🚫 Exclude NBA Players", [n for n in nba_names if n not in nba_locked], key="nba_excl")

        st.caption(f"NBA Pool Loaded: **{len(nba_pool)}** players across **{len(ctx['todays_games'])}** games on {slate_date}.")

        if st.button("🚀 Generate Optimal NBA DFS Lineups", key="run_nba_dfs", type="primary"):
            if nba_pool.empty:
                st.warning(f"No NBA games/players found for {slate_date}. Try changing the NBA Slate Date in the left sidebar!")
            else:
                lus = pro_sim.optimize_nba_dfs_lineups(
                    nba_pool,
                    mode=m_code,
                    num_lineups=nba_num_lu,
                    min_unique=nba_min_u,
                    max_exposure=nba_max_e,
                    locked_names=nba_locked,
                    excluded_names=nba_excl,
                )
                if not lus:
                    st.error("No valid NBA lineups found with those locks. Try removing a locked player.")
                else:
                    for idx, lu in enumerate(lus, start=1):
                        with st.container(border=True):
                            st.markdown(
                                f"#### 🏆 Optimal NBA Lineup #{idx} — Salary: `${lu['salary'].sum():,} / $50,000` | "
                                f"Median DK Pts: `{lu['proj'].sum():.1f}` | Ceiling: `{lu['ceiling'].sum():.1f}`"
                            )
                            st.dataframe(lu[["slot", "name", "team", "game", "salary", "proj", "ceiling", "est_own", "value", "flags"]], use_container_width=True, hide_index=True)

        if not nba_pool.empty:
            with st.expander("📋 View Full Projected NBA Player Pool, Ownership & Leverage"):
                st.dataframe(nba_pool.sort_values("leverage", ascending=False), use_container_width=True, hide_index=True)

# =====================================================================
#                     MEN'S COLLEGE BASKETBALL TAB
# =====================================================================
elif sport_tab == "🎓 Men's College Basketball":
    slate_date = st.sidebar.date_input("NCAAB Slate Date", value=datetime.now()).strftime("%Y-%m-%d")
    ctx = get_ncaab(slate_date)
    st.subheader(f"Men's College Basketball — {slate_date} ({len(ctx['todays_games'])} D1 Games)")
    st.caption("Upgraded with Opponent-Adjusted Efficiency (SOS), Tempo-Adjusted Possessions & Short-Rest/B2B Fatigue")

    c1, c2 = st.columns(2)
    cbb_min_wp = c1.slider("Min Winner Confidence %", 52, 90, 62, key="cbb_min_wp") / 100.0
    cbb_tiers = c2.slider("Tiered Winners to Show", 2, 12, 6, key="cbb_tiers")

    if st.button("🚀 Simulate Today's CBB Straight-Up Winners (1M Sims/Game)", type="primary"):
        winner_cards = []
        rows = []
        tight_dog_rows = []
        for g in ctx["game_list"]:
            res = col_sim.simulate_ncaab_game(g)
            if res["win_prob"] >= cbb_min_wp:
                winner_cards.append({
                    "Matchup": res["game"],
                    "Projected Winner": res["winner"],
                    "Win Prob": res["win_prob"],
                    "Fair Moneyline": res["fair_ml"],
                    "Fair Payout": res["fair_payout"],
                    "Proj Score": res["proj_score"],
                    "Home Win %": f"{res['win_prob']:.1%}" if res["winner"] == res["home"] else f"{1.0 - res['win_prob']:.1%}",
                    "Away Win %": f"{res['win_prob']:.1%}" if res["winner"] == res["away"] else f"{1.0 - res['win_prob']:.1%}",
                    "Flags": res["flags"],
                })
            rows.append({
                "Matchup": res["game"],
                "Projected Winner": res["winner"],
                "Win Prob": f"{res['win_prob']:.1%}",
                "Fair ML": res["fair_ml"],
                "Proj Score": res["proj_score"],
                "Vegas Line": f"{res['home']} {res['spread_line']:+.1f}",
                "Best Spread": res["best_side"],
                "Flags": res["flags"],
            })
            if res["is_tight_dog"]:
                tight_dog_rows.append({
                    "Underdog": res["dog_label"],
                    "Matchup": res["game"],
                    "Proj Score": res["proj_score"],
                    "Outright Win %": res["dog_win_prob"],
                    "Fair Payout": f"{res['dog_fair_mult']:.2f}x",
                })

        winner_cards.sort(key=lambda x: x["Win Prob"], reverse=True)
        render_tiered_game_winners(winner_cards, max_tiers=cbb_tiers, sport_label="Men's CBB")

        if tight_dog_rows:
            st.markdown("### 🐕 Tight-Spread Underdog Outright Win Probabilities (Spread ≤ 3.5)")
            df_dogs = pd.DataFrame(tight_dog_rows).sort_values("Outright Win %", ascending=False)
            df_dogs["Outright Win %"] = df_dogs["Outright Win %"].map(lambda x: f"{x:.1%}")
            st.dataframe(df_dogs, use_container_width=True, hide_index=True)

        if rows:
            with st.expander("📊 View Full D1 Slate Winner & Spread Board"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =====================================================================
#                        COLLEGE FOOTBALL TAB
# =====================================================================
elif sport_tab == "🏟️ College Football":
    ctx = get_ncaaf()
    st.subheader(f"College Football FBS — Season {ctx['curr_szn']} ({len(ctx['upcoming_games'])} Upcoming Games)")

    c1, c2 = st.columns(2)
    cfb_min_wp = c1.slider("Min Winner Confidence %", 50, 90, 58, key="cfb_min_wp") / 100.0
    cfb_tiers = c2.slider("Tiered Winners to Show", 2, 12, 6, key="cfb_tiers")

    if st.button("🚀 Simulate All Upcoming CFB Winners & Spreads (1M Sims/Game)", type="primary"):
        winner_cards = []
        rows = []
        for g in ctx["upcoming_games"]:
            res = col_sim.simulate_cfb_game(ctx, g)
            if res["win_prob"] >= cfb_min_wp:
                winner_cards.append({
                    "Matchup": res["game"],
                    "Projected Winner": res["winner"],
                    "Win Prob": res["win_prob"],
                    "Fair Moneyline": res["fair_ml"],
                    "Fair Payout": res["fair_payout"],
                    "Proj Score": res["proj_score"],
                    "Home Win %": f"{res['win_prob']:.1%}" if res["winner"] == res["home"] else f"{1.0 - res['win_prob']:.1%}",
                    "Away Win %": f"{res['win_prob']:.1%}" if res["winner"] == res["away"] else f"{1.0 - res['win_prob']:.1%}",
                    "Flags": f"Blowout Risk: {res['blowout_prob']:.0%}",
                })
            rows.append({
                "Matchup": res["game"],
                "Projected Winner": res["winner"],
                "Win Prob": f"{res['win_prob']:.1%}",
                "Fair ML": res["fair_ml"],
                "Proj Score": res["proj_score"],
                "Vegas Line": f"{res['home']} {res['spread_line']:+.1f}",
                "Best Side": res["best_side"],
                "Cover Prob": f"{res['best_side_prob']:.1%}",
            })
        winner_cards.sort(key=lambda x: x["Win Prob"], reverse=True)
        render_tiered_game_winners(winner_cards, max_tiers=cfb_tiers, sport_label="College Football")

        if rows:
            with st.expander("📊 View Full FBS Slate Winner & Spread Board"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
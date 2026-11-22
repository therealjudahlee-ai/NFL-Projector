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


def _prob_to_ml(prob):
    p = float(np.clip(prob, 0.01, 0.99))
    if p >= 0.50:
        return f"{int(round(-100.0 * p / (1.0 - p)))}"
    return f"+{int(round(100.0 * (1.0 - p) / p))}"


def run_nfl_game_winners(ctx):
    """Uses pro_sim.simulate_nfl_games if available, or computes straight-up winners directly from ctx."""
    if hasattr(pro_sim, "simulate_nfl_games"):
        return pro_sim.simulate_nfl_games(ctx)

    matchup_map = ctx.get("matchup_map", {})
    seen_games = set()
    results = []
    for team, info in matchup_map.items():
        g_lbl = info["game_label"]
        if g_lbl in seen_games:
            continue
        seen_games.add(g_lbl)

        parts = g_lbl.split(" @ ")
        away, home = (parts[0], parts[1]) if len(parts) == 2 else (info["opponent"], team)
        h_info = matchup_map.get(home, info)
        a_info = matchup_map.get(away, info)

        h_pts = h_info.get("proj_home_pts", h_info.get("team_proj_pts", 22.0))
        a_pts = h_info.get("proj_away_pts", a_info.get("team_proj_pts", 21.0))

        if "home_win_prob" in h_info:
            h_prob = h_info["home_win_prob"]
            a_prob = h_info["away_win_prob"]
        else:
            margin = -h_info.get("team_spread", a_pts - h_pts)
            sims = np.random.normal(margin, 11.5, 250_000).astype(np.float32)
            h_prob = float(np.mean(sims > 0.0))
            a_prob = 1.0 - h_prob

        winner, win_prob = (home, h_prob) if h_prob >= a_prob else (away, a_prob)
        flags = []
        if h_info.get("is_thursday"):
            flags.append("TNF Primetime")
        if h_info.get("blowout_prob", 0) >= 0.35:
            flags.append(f"Blowout Risk {h_info['blowout_prob']:.0%}")

        results.append({
            "Matchup": g_lbl,
            "Projected Winner": winner,
            "Win Prob": win_prob,
            "Fair Moneyline": _prob_to_ml(win_prob),
            "Fair Payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
            "Proj Score": f"{away} {a_pts:.1f} - {home} {h_pts:.1f}",
            "Home Win %": f"{h_prob:.1%}",
            "Away Win %": f"{a_prob:.1%}",
            "Is TNF": h_info.get("is_thursday", False),
            "Flags": " | ".join(flags) if flags else "Clean Environment",
        })
    results.sort(key=lambda x: x["Win Prob"], reverse=True)
    return results


def run_nba_game_winners(ctx):
    """Uses pro_sim.simulate_nba_games if available, or computes straight-up winners directly from ctx."""
    if hasattr(pro_sim, "simulate_nba_games"):
        return pro_sim.simulate_nba_games(ctx)

    matchup_map = ctx.get("matchup_map", {})
    seen_games = set()
    results = []
    for team, info in matchup_map.items():
        g_lbl = info["game_label"]
        if g_lbl in seen_games:
            continue
        seen_games.add(g_lbl)

        parts = g_lbl.split(" @ ")
        away, home = (parts[0], parts[1]) if len(parts) == 2 else (info["opponent"], team)
        h_info = matchup_map.get(home, info)
        a_info = matchup_map.get(away, info)

        h_pts = h_info.get("proj_home_pts", h_info.get("team_proj_pts", 114.0))
        a_pts = h_info.get("proj_away_pts", a_info.get("team_proj_pts", 112.0))

        if "home_win_prob" in h_info:
            h_prob = h_info["home_win_prob"]
            a_prob = h_info["away_win_prob"]
        else:
            margin = h_info.get("proj_margin", h_pts - a_pts + 2.2)
            sims = np.random.normal(margin, 12.0, 250_000).astype(np.float32)
            h_prob = float(np.mean(sims > 0.0))
            a_prob = 1.0 - h_prob

        winner, win_prob = (home, h_prob) if h_prob >= a_prob else (away, a_prob)
        flags = []
        if h_info.get("is_b2b"):
            flags.append(f"{home} on B2B")
        if a_info.get("is_b2b"):
            flags.append(f"{away} on Road B2B")

        results.append({
            "Matchup": g_lbl,
            "Projected Winner": winner,
            "Win Prob": win_prob,
            "Fair Moneyline": _prob_to_ml(win_prob),
            "Fair Payout": f"{round(1.0 / max(win_prob, 0.01), 2):.2f}x",
            "Proj Score": f"{away} {a_pts:.1f} - {home} {h_pts:.1f}",
            "Home Win %": f"{h_prob:.1%}",
            "Away Win %": f"{a_prob:.1%}",
            "Flags": " | ".join(flags) if flags else "Rested / Clean Environment",
        })
    results.sort(key=lambda x: x["Win Prob"], reverse=True)
    return results


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
            st.caption(f"**
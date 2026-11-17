import itertools
import time
import nflreadpy as nfl
import numpy as np
import pandas as pd

N_SIMS = 1_000_000  # 1 Million Monte Carlo simulations per prop

print("=================================================================")
print("        NFL-PROJECTOR | 1,000,000-SIM MONTE CARLO ENGINE         ")
print("=================================================================")
print("Fetching live NFL schedules & player stats...")

current_season = nfl.get_current_season()
current_week = nfl.get_current_week()
seasons = [current_season - 1, current_season]

player_stats = nfl.load_player_stats(seasons).to_pandas()
schedules = nfl.load_schedules([current_season]).to_pandas()

# Filter to regular season games only
reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()
team_col = "team" if "team" in reg_stats.columns else "recent_team"

# Map stats to their underlying volume metric (Volume x Efficiency) & min healthy usage
VOLUME_MAP = {
    "passing_yards": ("attempts", 8),
    "rushing_yards": ("carries", 4),
    "receiving_yards": ("targets", 2),
    "receptions": ("targets", 2),
}

# Primary props scanned for each position in Automated Parlay Mode
POS_PROPS = {
    "QB": ["passing_yards"],
    "RB": ["rushing_yards", "receiving_yards"],
    "WR": ["receiving_yards", "receptions"],
    "TE": ["receiving_yards", "receptions"],
}


def get_weekly_matchups():
    """Maps every team playing in the current NFL week to its opponent and Vegas lines."""
    wk_games = schedules[schedules["week"] == current_week].copy()
    matchup_map = {}

    for _, game in wk_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        spread = game.get("spread_line", np.nan)  # Positive = home favored in nflverse
        total = game.get("total_line", np.nan)

        home_spread = -spread if pd.notna(spread) else 0.0
        away_spread = spread if pd.notna(spread) else 0.0
        total_val = total if pd.notna(total) else 44.5
        has_vegas = pd.notna(spread) and pd.notna(total)

        matchup_map[home] = {
            "opponent": away,
            "team_spread": home_spread,
            "total_line": total_val,
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }
        matchup_map[away] = {
            "opponent": home,
            "team_spread": away_spread,
            "total_line": total_val,
            "has_vegas": has_vegas,
            "game_label": f"{away} @ {home}",
        }

    return wk_games, matchup_map


def build_defense_lookup():
    """Pre-computes defensive matchup multipliers by (opponent_team, position, stat_col)."""
    recent_def = reg_stats[reg_stats["season"] == current_season]
    if recent_def["week"].nunique() < 4:
        recent_def = reg_stats

    lookup = {}
    all_stats = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
    ]

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = recent_def[recent_def["position"] == pos]
        for stat in all_stats:
            if stat not in pos_df.columns:
                continue
            game_totals = (
                pos_df.groupby(["opponent_team", "season", "week"])[stat]
                .sum()
                .reset_index()
            )
            league_avg = game_totals[stat].mean()
            if pd.isna(league_avg) or league_avg == 0:
                continue

            opp_avgs = game_totals.groupby("opponent_team")[stat].mean()
            for opp, val in opp_avgs.items():
                lookup[(opp, pos, stat)] = val / league_avg

    return lookup


WK_GAMES, MATCHUP_MAP = get_weekly_matchups()
DEFENSE_LOOKUP = build_defense_lookup()


def run_1m_simulation(p_logs, opponent_team, stat_col, prop_line):
    """
    Runs 1,000,000 Volume x Efficiency Monte Carlo simulations for a player prop
    with injury filtering, EWMA recency weighting, defensive matchup, and Vegas script.
    """
    recent = p_logs.sort_values(["season", "week"]).tail(15).copy()
    full_name = recent["player_display_name"].iloc[-1]
    position = recent["position"].iloc[-1]
    player_team = recent[team_col].iloc[-1]

    # 1. Filter out early-exit injury/decoy games
    if stat_col in VOLUME_MAP:
        vol_col, min_vol = VOLUME_MAP[stat_col]
        healthy = recent[recent[vol_col] >= min_vol].tail(12).copy()
        if len(healthy) < 4:
            healthy = recent.tail(12).copy()
    else:
        healthy = recent.tail(12).copy()

    # 2. Defense Matchup Multiplier
    def_mult = DEFENSE_LOOKUP.get((opponent_team, position, stat_col), 1.0)

    # 3. Vegas Game Script Multiplier
    game_info = MATCHUP_MAP.get(player_team)
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

    # 4. Run 1,000,000 Simulations (using float32 for speed & low RAM)
    if stat_col in VOLUME_MAP:
        vol_col, _ = VOLUME_MAP[stat_col]
        ewm_vol = healthy[vol_col].ewm(span=5).mean().iloc[-1]
        vol_std = max(healthy[vol_col].std(), 1.0)

        eff_series = (healthy[stat_col] / healthy[vol_col].replace(0, np.nan)).dropna()
        ewm_eff = eff_series.ewm(span=5).mean().iloc[-1]
        eff_std = max(eff_series.std(), 0.05)

        adj_vol = ewm_vol * vegas_mult
        adj_eff = ewm_eff * def_mult

        sim_vol = np.clip(
            np.random.normal(adj_vol, vol_std, N_SIMS).astype(np.float32), 0, None
        )
        sim_eff = np.clip(
            np.random.normal(adj_eff, eff_std, N_SIMS).astype(np.float32), 0, None
        )
        sims = sim_vol * sim_eff
        breakdown = f"{adj_vol:.1f} {vol_col} x {adj_eff:.2f} {stat_col}/{vol_col[:-1]}"
    else:
        ewm_avg = healthy[stat_col].ewm(span=5).mean().iloc[-1]
        stat_std = max(healthy[stat_col].std(), 0.5)
        adj_avg = ewm_avg * def_mult * vegas_mult
        sims = np.clip(
            np.random.normal(adj_avg, stat_std, N_SIMS).astype(np.float32), 0, None
        )
        breakdown = f"Weighted Avg: {adj_avg:.2f}"

    over_prob = float(np.mean(sims > prop_line))
    under_prob = float(np.mean(sims < prop_line))
    median_proj = float(np.median(sims))

    return {
        "player": full_name,
        "position": position,
        "team": player_team,
        "opponent": opponent_team,
        "game": game_info["game_label"] if game_info else f"{player_team} vs {opponent_team}",
        "stat": stat_col,
        "line": prop_line,
        "proj": median_proj,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "def_mult": def_mult,
        "vegas_mult": vegas_mult,
        "vegas_note": vegas_note,
        "breakdown": breakdown,
        "games_used": len(healthy),
    }


def single_player_mode():
    print("\n--- SINGLE PLAYER PROP SIMULATOR (1,000,000 SIMS) ---")
    print("Stats: passing_yards, rushing_yards, receiving_yards, receptions, passing_tds, rushing_tds, receiving_tds")

    player = input("\nPlayer Name (e.g., Josh Allen): ").strip()
    p_logs = reg_stats[
        reg_stats["player_display_name"].str.contains(player, case=False, na=False)
    ]

    if p_logs.empty:
        print(f"[!] No player found matching '{player}'.")
        return

    player_team = p_logs.sort_values(["season", "week"])[team_col].iloc[-1]
    auto_opp = MATCHUP_MAP[player_team]["opponent"] if player_team in MATCHUP_MAP else ""

    opp_prompt = (
        f"Opponent Team Abbr [Press Enter for Week {current_week} opp: {auto_opp}]: "
        if auto_opp
        else "Opponent Team Abbr (e.g., KC, BAL, DAL): "
    )
    opp = input(opp_prompt).strip().upper() or auto_opp

    stat = input("Stat (e.g., passing_yards, rushing_yards, receiving_yards, receptions): ").strip().lower()
    if stat not in reg_stats.columns:
        print(f"[!] Invalid stat '{stat}'.")
        return

    try:
        line = float(input("Prop Line (e.g., 245.5): ").strip())
    except ValueError:
        print("[!] Invalid number for prop line.")
        return

    t0 = time.time()
    res = run_1m_simulation(p_logs, opp, stat, line)
    elapsed = time.time() - t0

    print("\n" + "=" * 66)
    print(f" {res['player']} ({res['position']} - {res['team']}) vs. {res['opponent']} | {res['stat'].upper()} {res['line']}")
    print("=" * 66)
    print(f" Sims Run:          {N_SIMS:,} (Completed in {elapsed:.2f}s)")
    print(f" Healthy Games:     {res['games_used']} | {res['vegas_note']}")
    print(f" Matchup Factor:    Defense {res['def_mult']:.2f}x | Vegas Script {res['vegas_mult']:.2f}x")
    print(f" Engine Breakdown:  {res['breakdown']}")
    print("-" * 66)
    print(f" Median Projection: {res['proj']:.1f}  (Edge vs Line: {res['proj'] - res['line']:+.1f})")
    print(f" OVER {res['line']:<5} Prob: {res['over_prob']:.2%}")
    print(f" UNDER {res['line']:<4} Prob: {res['under_prob']:.2%}")
    print("=" * 66)


def automated_parlay_mode():
    print(f"\n--- WEEK {current_week} AUTOMATED PARLAY BUILDER (1,000,000 SIMS/PROP) ---")
    print(f"Detected {len(WK_GAMES)} games on the Week {current_week} slate:")
    for _, g in WK_GAMES.iterrows():
        print(f"  • {g['away_team']} @ {g['home_team']}")

    try:
        rate_in = float(input("\nMinimum Hit Rate % per leg (e.g., 75 for 75%): ").strip())
        min_hit_rate = rate_in / 100.0 if rate_in > 1 else rate_in
        parlay_legs = int(input("Legs per parlay (e.g., 3 or 4): ").strip())
        discount = float(
            input("Line Aggressiveness (1.0 = full avg, 0.85 = safer alt-line floor): ").strip()
        )
    except ValueError:
        print("Using defaults: 75% hit rate, 3 legs, 0.85 alt-line floor.")
        min_hit_rate, parlay_legs, discount = 0.75, 3, 0.85

    t0 = time.time()
    recent_pool = reg_stats[
        (reg_stats["season"] == current_season)
        & (reg_stats["week"] >= max(1, current_week - 3))
    ]
    active_players = recent_pool[
        ["player_id", "player_display_name", "position", team_col]
    ].drop_duplicates(subset=["player_id"])

    qualifying_legs = []
    sims_count = 0

    print("\nRunning 1,000,000 simulations per active starter across the slate...")
    for _, p in active_players.iterrows():
        team = p[team_col]
        pos = p["position"]
        if team not in MATCHUP_MAP or pos not in POS_PROPS:
            continue

        opp = MATCHUP_MAP[team]["opponent"]
        p_logs = reg_stats[reg_stats["player_id"] == p["player_id"]].sort_values(
            ["season", "week"]
        )
        if len(p_logs) < 4:
            continue

        for stat_col in POS_PROPS[pos]:
            vol_col, min_vol = VOLUME_MAP[stat_col]
            healthy = p_logs[p_logs[vol_col] >= min_vol].tail(10)
            if len(healthy) < 4:
                continue

            raw_avg = healthy[stat_col].mean()
            if (stat_col != "receptions" and raw_avg < 18) or (
                stat_col == "receptions" and raw_avg < 3.5
            ):
                continue

            target_line = np.floor(raw_avg * discount) + 0.5
            res = run_1m_simulation(p_logs, opp, stat_col, target_line)
            sims_count += N_SIMS

            if res["over_prob"] >= min_hit_rate:
                qualifying_legs.append(res)

    elapsed = time.time() - t0
    qualifying_legs.sort(key=lambda x: x["over_prob"], reverse=True)

    print(f"\nCompleted {sims_count:,} total simulations in {elapsed:.1f} seconds!")
    print(f"Found {len(qualifying_legs)} individual props hitting >= {min_hit_rate:.0%}:")
    for leg in qualifying_legs[:15]:
        print(
            f"  [{leg['over_prob']:.2%}] {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
            f"(Proj: {leg['proj']:.1f} | {leg['game']})"
        )

    if len(qualifying_legs) < parlay_legs:
        print(f"\n[!] Fewer than {parlay_legs} legs hit >= {min_hit_rate:.0%}. Try lowering your target % or line aggressiveness.")
        return

    print(f"\n================ TOP {parlay_legs}-LEG PARLAYS (1 LEG PER GAME) ================")
    top_pool = qualifying_legs[:14]
    valid_parlays = []

    for combo in itertools.combinations(top_pool, parlay_legs):
        if len({leg["game"] for leg in combo}) < parlay_legs:
            continue
        combined_prob = float(np.prod([leg["over_prob"] for leg in combo]))
        valid_parlays.append((combined_prob, combo))

    valid_parlays.sort(key=lambda x: x[0], reverse=True)

    for idx, (prob, combo) in enumerate(valid_parlays[:5], start=1):
        fair_odds = (
            int(-100 * prob / (1 - prob))
            if prob > 0.5
            else int(100 * (1 - prob) / prob)
        )
        print(f"\nParlay #{idx} | Combined Hit Prob: {prob:.2%} (Fair Odds: {fair_odds:+d})")
        for leg in combo:
            print(
                f"   -> {leg['player']} ({leg['team']}) OVER {leg['line']} {leg['stat']} "
                f"[{leg['over_prob']:.2%} | Proj: {leg['proj']:.1f} | {leg['game']}]"
            )


if __name__ == "__main__":
    print(f"Loaded Season {current_season} | Current Slate: Week {current_week}")
    while True:
        print("\nChoose a mode:")
        print("  [1] Single Player Prop Simulator (1,000,000 Sims)")
        print("  [2] Auto-Scan This Week's Matchups & Build Parlays (1,000,000 Sims/Prop)")
        print("  [3] Quit")
        choice = input("Select [1/2/3]: ").strip()

        if choice == "1":
            single_player_mode()
        elif choice == "2":
            automated_parlay_mode()
        elif choice == "3" or choice.lower() in ["quit", "exit", "q"]:
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")
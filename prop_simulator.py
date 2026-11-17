import nflreadpy as nfl
import numpy as np
import pandas as pd

print("Loading NFL player stats & schedules (2025-2026)...")
seasons = [2025, 2026]
player_stats = nfl.load_player_stats(seasons).to_pandas()
schedules = nfl.load_schedules(seasons).to_pandas()

# Filter to regular season games
reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()

# Map each yardage stat to its underlying volume metric (Volume x Efficiency)
VOLUME_MAP = {
    "passing_yards": ("attempts", 5),        # min 5 pass attempts to count as a healthy game
    "rushing_yards": ("carries", 3),         # min 3 carries
    "receiving_yards": ("targets", 2),       # min 2 targets
    "receptions": ("targets", 2),            # min 2 targets
}


def get_defense_multiplier(opponent_team, position, stat_col):
    """Calculates opponent defense strength against a specific position (1.00 = league avg)."""
    latest_season = reg_stats["season"].max()
    recent_def = reg_stats[
        (reg_stats["season"] == latest_season) & (reg_stats["position"] == position)
    ]

    if recent_def["week"].nunique() < 4:
        recent_def = reg_stats[reg_stats["position"] == position]

    game_totals = (
        recent_def.groupby(["opponent_team", "season", "week"])[stat_col]
        .sum()
        .reset_index()
    )

    league_avg = game_totals[stat_col].mean()
    opp_games = game_totals[game_totals["opponent_team"] == opponent_team]

    if opp_games.empty or league_avg == 0:
        return 1.0

    return opp_games[stat_col].mean() / league_avg


def get_vegas_script_multiplier(player_team, opponent_team, stat_col):
    """
    Pulls the upcoming/latest scheduled game between player_team and opponent_team
    to adjust volume based on Vegas spread and game total.
    """
    latest_season = schedules["season"].max()
    season_sched = schedules[schedules["season"] == latest_season]

    matchup = season_sched[
        (
            (season_sched["home_team"] == player_team)
            & (season_sched["away_team"] == opponent_team)
        )
        | (
            (season_sched["away_team"] == player_team)
            & (season_sched["home_team"] == opponent_team)
        )
    ].tail(1)

    if matchup.empty:
        return 1.0, "No Vegas line found"

    game = matchup.iloc[0]
    total_line = game.get("total_line", np.nan)
    spread_line = game.get("spread_line", np.nan)  # In nflverse, positive spread = home favored

    if pd.isna(total_line) or pd.isna(spread_line):
        return 1.0, "Vegas lines pending"

    # Determine player's team spread (negative = favorite, positive = underdog)
    is_home = game["home_team"] == player_team
    team_spread = -spread_line if is_home else spread_line

    # Baseline NFL game total is ~44.5
    total_factor = 1.0 + ((total_line - 44.5) * 0.005)  # +/- 0.5% per point off 44.5

    # Game script factor: Favorites run more, underdogs pass more
    script_factor = 1.0
    if stat_col == "rushing_yards":
        script_factor = 1.0 + (-team_spread * 0.006)  # -7 favorite gets ~+4.2% rush boost
    elif stat_col in ["passing_yards", "receiving_yards", "receptions"]:
        script_factor = 1.0 + (team_spread * 0.005)   # +7 underdog gets ~+3.5% pass boost

    combined_vegas = np.clip(total_factor * script_factor, 0.85, 1.18)
    note = f"Spread: {player_team} {team_spread:+.1f} | Total: {total_line}"
    return combined_vegas, note


def simulate_prop(player_name, opponent_team, stat_col, prop_line, n_sims=10000):
    opponent_team = opponent_team.upper().strip()
    p_logs = reg_stats[
        reg_stats["player_display_name"].str.contains(player_name, case=False, na=False)
    ]

    if p_logs.empty:
        print(f"\n[!] No player found matching '{player_name}'.")
        return

    # Sort chronologically and take last 15 games before filtering injury exits
    recent = p_logs.sort_values(["season", "week"]).tail(15).copy()
    full_name = recent["player_display_name"].iloc[-1]
    position = recent["position"].iloc[-1]
    player_team = recent["team"].iloc[-1] if "team" in recent.columns else recent["recent_team"].iloc[-1]

    # 1. Filter out early injury/decoy games if stat uses a volume metric
    if stat_col in VOLUME_MAP:
        vol_col, min_vol = VOLUME_MAP[stat_col]
        healthy_logs = recent[recent[vol_col] >= min_vol].tail(12).copy()
        if len(healthy_logs) < 4:
            healthy_logs = recent.tail(12).copy()
    else:
        healthy_logs = recent.tail(12).copy()

    # 2. Get Defense & Vegas Multipliers
    def_mult = get_defense_multiplier(opponent_team, position, stat_col)
    vegas_mult, vegas_note = get_vegas_script_multiplier(player_team, opponent_team, stat_col)

    # 3. Run Simulation (Volume x Efficiency vs. Direct Stat)
    if stat_col in VOLUME_MAP:
        vol_col, _ = VOLUME_MAP[stat_col]

        # Recency-weighted (EWMA) volume & efficiency
        ewm_vol = healthy_logs[vol_col].ewm(span=5).mean().iloc[-1]
        vol_std = max(healthy_logs[vol_col].std(), 1.0)

        # Efficiency = stat per opportunity (e.g., yards per target or catch rate)
        eff_series = healthy_logs[stat_col] / healthy_logs[vol_col].replace(0, np.nan)
        eff_series = eff_series.dropna()
        ewm_eff = eff_series.ewm(span=5).mean().iloc[-1]
        eff_std = max(eff_series.std(), 0.05)

        # Apply Vegas script adjustment to volume, and Defense matchup to efficiency
        adj_vol = ewm_vol * vegas_mult
        adj_eff = ewm_eff * def_mult

        sim_vol = np.clip(np.random.normal(adj_vol, vol_std, n_sims), 0, None)
        sim_eff = np.clip(np.random.normal(adj_eff, eff_std, n_sims), 0, None)
        sims = sim_vol * sim_eff

        model_note = f"Volume ({adj_vol:.1f} {vol_col}) x Efficiency ({adj_eff:.2f} {stat_col}/{vol_col[:-1]})"
    else:
        # Direct stat simulation (e.g., touchdowns, interceptions)
        ewm_avg = healthy_logs[stat_col].ewm(span=5).mean().iloc[-1]
        stat_std = max(healthy_logs[stat_col].std(), 0.5)
        adj_avg = ewm_avg * def_mult * vegas_mult
        sims = np.clip(np.random.normal(adj_avg, stat_std, n_sims), 0, None)
        model_note = f"Weighted Avg: {adj_avg:.2f}"

    over_prob = np.mean(sims > prop_line)
    under_prob = np.mean(sims < prop_line)
    median_proj = np.median(sims)
    edge = median_proj - prop_line

    print("\n" + "=" * 62)
    print(f" {full_name} ({position} - {player_team}) vs. {opponent_team} | {stat_col.upper()} {prop_line}")
    print("=" * 62)
    print(f" Healthy Games Used: {len(healthy_logs)} | {vegas_note}")
    print(f" Defense Matchup:    {def_mult:.2f}x | Vegas Script Factor: {vegas_mult:.2f}x")
    print(f" Engine Breakdown:   {model_note}")
    print("-" * 62)
    print(f" Median Projection:  {median_proj:.1f}  (Edge vs Line: {edge:+.1f})")
    print(f" OVER {prop_line:<5} Prob:  {over_prob:.1%}")
    print(f" UNDER {prop_line:<4} Prob:  {under_prob:.1%}")
    print("=" * 62)


# 4. Interactive Command-Line Loop
if __name__ == "__main__":
    print("\nReady! Available stats: passing_yards, rushing_yards, receiving_yards, receptions, passing_tds, rushing_tds, receiving_tds")
    while True:
        print("\nEnter prop details (or type 'quit' to exit):")
        player = input("Player Name (e.g., Josh Allen): ").strip()
        if player.lower() in ["quit", "exit", "q"]:
            break

        opp = input("Opponent Team Abbr (e.g., KC, BAL, DAL): ").strip()
        stat = input("Stat (e.g., passing_yards, rushing_yards, receiving_yards, receptions): ").strip().lower()

        try:
            line = float(input("Prop Line (e.g., 245.5): ").strip())
        except ValueError:
            print("[!] Invalid number for prop line. Try again.")
            continue

        simulate_prop(player, opp, stat, line)
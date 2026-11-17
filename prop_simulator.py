import nflreadpy as nfl
import numpy as np
import pandas as pd

# 1. Load weekly player stats using the official nflreadpy package
print("Fetching NFL player stats...")
# Convert Polars DataFrame to pandas so all your existing code works seamlessly
player_stats = nfl.load_player_stats([2024, 2025]).to_pandas()

# Filter to regular season games only
reg_stats = player_stats[player_stats["season_type"] == "REG"].copy()


def simulate_player_prop(player_name, stat_col, prop_line, n_sims=10000):
    # Filter logs for the specific player
    p_logs = reg_stats[
        reg_stats["player_display_name"].str.contains(player_name, case=False, na=False)
    ]

    if p_logs.empty:
        print(f"No data found for '{player_name}'. Check spelling.")
        return

    # Use their most recent 12 games so current role/usage is reflected
    recent_logs = p_logs.sort_values(["season", "week"]).tail(12)
    full_name = recent_logs["player_display_name"].iloc[-1]

    avg_val = recent_logs[stat_col].mean()
    std_val = recent_logs[stat_col].std()

    # Run 10,000 Monte Carlo simulations (clipped at 0 since yards/receptions can't be negative)
    sims = np.random.normal(loc=avg_val, scale=std_val, size=n_sims)
    sims = np.clip(sims, 0, None)

    over_prob = np.mean(sims > prop_line)
    under_prob = np.mean(sims < prop_line)
    median_proj = np.median(sims)

    print(f"\n--- {full_name} | {stat_col} vs. Line {prop_line} ({n_sims:,} Sims) ---")
    print(f"Last 12 Games Avg: {avg_val:.1f} (Std Dev: {std_val:.1f})")
    print(f"Median Projection: {median_proj:.1f}")
    print(f"OVER {prop_line} Prob:  {over_prob:.1%}")
    print(f"UNDER {prop_line} Prob: {under_prob:.1%}")


# Test examples:
simulate_player_prop("Josh Allen", "passing_yards", 245.5)
simulate_player_prop("Saquon Barkley", "rushing_yards", 88.5)
simulate_player_prop("Ja'Marr Chase", "receiving_yards", 82.5)
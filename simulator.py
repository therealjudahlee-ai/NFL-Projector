import numpy as np
import pandas as pd

# 1. Pull real NFL game results directly from nflverse
print("Fetching NFL data...")
url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
schedules = pd.read_csv(url)

# Filter to completed regular-season games from 2024 & 2025
completed = schedules[
    (schedules['season'].isin([2024, 2025])) &
    (schedules['game_type'] == 'REG') &
    (schedules['result'].notna())
].copy()

# 2. Calculate offensive & defensive stats for every team
home_stats = completed.groupby('home_team').agg(
    home_scored=('home_score', 'mean'),
    home_allowed=('away_score', 'mean'),
    home_std=('home_score', 'std')
)

away_stats = completed.groupby('away_team').agg(
    away_scored=('away_score', 'mean'),
    away_allowed=('home_score', 'mean'),
    away_std=('away_score', 'std')
)

# Combine into a single team ratings table
team_ratings = pd.concat([home_stats, away_stats], axis=1)
team_ratings['avg_scored'] = (team_ratings['home_scored'] + team_ratings['away_scored']) / 2
team_ratings['avg_allowed'] = (team_ratings['home_allowed'] + team_ratings['away_allowed']) / 2
team_ratings['std_dev'] = (team_ratings['home_std'] + team_ratings['away_std']) / 2

# 3. Simulate a matchup 10,000 times
def simulate_game(home_team, away_team, n_sims=10000):
    home = team_ratings.loc[home_team]
    away = team_ratings.loc[away_team]

    # Expected score = average of offense vs opponent defense (+2.0 home field edge)
    exp_home = ((home['avg_scored'] + away['avg_allowed']) / 2) + 2.0
    exp_away = (away['avg_scored'] + home['avg_allowed']) / 2

    # Run Monte Carlo simulations
    sim_home = np.round(np.random.normal(exp_home, home['std_dev'], n_sims))
    sim_away = np.round(np.random.normal(exp_away, away['std_dev'], n_sims))

    home_win_prob = np.mean(sim_home > sim_away)
    away_win_prob = np.mean(sim_away > sim_home)

    print(f"\n--- {away_team} @ {home_team} ({n_sims:,} Simulations) ---")
    print(f"{home_team} Win Prob: {home_win_prob:.1%}")
    print(f"{away_team} Win Prob: {away_win_prob:.1%}")
    print(f"Median Score: {home_team} {np.median(sim_home):.0f} - {away_team} {np.median(sim_away):.0f}")

# Test it out!
simulate_game("KC", "BUF")
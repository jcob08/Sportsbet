import pandas as pd

# Load rolling stat files from the correct folder
batter_7 = pd.read_csv('data/raw/processed/statcast/processed_batter_last_7.csv')
batter_15 = pd.read_csv('data/raw/processed/statcast/processed_batter_last_15.csv')
batter_30 = pd.read_csv('data/raw/processed/statcast/processed_batter_last_30.csv')

# Load main game file (choose the one with most features)
games = pd.read_csv('data/games_with_engineered_features.csv')

# Example: Merge rolling stats into one DataFrame
batter_7 = batter_7.add_suffix('_7')
batter_15 = batter_15.add_suffix('_15')
batter_30 = batter_30.add_suffix('_30')

# Assume 'player_id' and 'game_id' are the join keys
batter_7 = batter_7.rename(columns={'player_id_7': 'player_id', 'game_id_7': 'game_id'})
batter_15 = batter_15.rename(columns={'player_id_15': 'player_id', 'game_id_15': 'game_id'})
batter_30 = batter_30.rename(columns={'player_id_30': 'player_id', 'game_id_30': 'game_id'})

# Merge all rolling stats
player_features = batter_7.merge(batter_15, on=['player_id', 'game_id'], how='left') \
                          .merge(batter_30, on=['player_id', 'game_id'], how='left')

# Merge with game context
full_features = player_features.merge(games, on='game_id', how='left')

# Save to CSV
full_features.to_csv('data/player_game_features.csv', index=False)
print("Player-game features file created: data/player_game_features.csv")
import pandas as pd

# Load the 2024 game outcomes
outcomes_path = 'sports_data/mlb/contextual/game_outcomes_2024.csv'
outcomes_df = pd.read_csv(outcomes_path)
print(f"\nOutcomes data shape: {outcomes_df.shape}")
print("\nSample of outcomes data:")
print(outcomes_df.head())

# Load the engineered features
features_path = 'data/contextual/games_with_engineered_features.csv'
features_df = pd.read_csv(features_path)
print(f"\nFeatures data shape: {features_df.shape}")
print("\nSample of features data:")
print(features_df.head())

# Check game_id format in both dataframes
print("\nGame ID samples from outcomes:")
print(outcomes_df['game_id'].head())
print("\nGame ID samples from features:")
print(features_df['game_id'].head())

# Merge the dataframes on 'game_id'
merged_df = pd.merge(outcomes_df, features_df, on='game_id', how='inner')
print(f"\nMerged data shape: {merged_df.shape}")

if len(merged_df) == 0:
    print("\nWARNING: No matching game IDs found between the datasets!")
    
    # Check for potential game_id mismatches
    print("\nUnique game IDs in outcomes:", len(outcomes_df['game_id'].unique()))
    print("Unique game IDs in features:", len(features_df['game_id'].unique()))
    
    # Check for any common game IDs
    common_ids = set(outcomes_df['game_id']).intersection(set(features_df['game_id']))
    print(f"Number of common game IDs: {len(common_ids)}")
else:
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv('data/contextual/merged_game_data_2024.csv', index=False)
    print("\nMerged data saved to 'data/contextual/merged_game_data_2024.csv'") 
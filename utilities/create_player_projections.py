# utilities/create_player_projections.py
import os
import pandas as pd
import numpy as np
from datetime import datetime

def create_player_projections():
    print("Creating player projections for model testing...")
    
    # Define the output directory
    prediction_dir = os.path.join("data", "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Load player data if available
    player_data = None
    analysis_dir = os.path.join("sports_data", "mlb", "analysis")
    
    # Try to find player game stats files
    years = [2023, 2024]
    for year in years:
        stats_file = os.path.join(analysis_dir, f"player_game_stats_{year}.csv")
        if os.path.exists(stats_file):
            print(f"Found player stats for {year}. Loading data...")
            year_data = pd.read_csv(stats_file)
            if player_data is None:
                player_data = year_data
            else:
                player_data = pd.concat([player_data, year_data])
    
    if player_data is not None:
        print(f"Loaded {len(player_data)} player-game records")
        
        # Calculate player-level stats
        player_stats = player_data.groupby('player_id').agg({
            'game_id': 'nunique',
            'at_bats': 'sum',
            'hits': 'sum',
            'home_runs': 'sum',
            'got_hit': 'mean',
            'got_home_run': 'mean'
        }).reset_index()
        
        # Rename columns
        player_stats = player_stats.rename(columns={
            'game_id': 'games',
            'got_hit': 'hit_rate',
            'got_home_run': 'hr_rate'
        })
        
        # Calculate batting average
        player_stats['batting_avg'] = player_stats['hits'] / player_stats['at_bats']
        
        # Add player names if available
        if 'player_name' in player_data.columns:
            name_mapping = player_data.drop_duplicates('player_id')[['player_id', 'player_name']]
            player_stats = pd.merge(player_stats, name_mapping, on='player_id', how='left')
        else:
            player_stats['player_name'] = player_stats['player_id'].apply(lambda x: f"Player {x}")
        
        # Create projected rates (with some noise for realism)
        np.random.seed(42)
        player_stats['projected_hit_rate_2025'] = player_stats['hit_rate'] * np.random.uniform(0.9, 1.1, len(player_stats))
        player_stats['projected_hr_rate_2025'] = player_stats['hr_rate'] * np.random.uniform(0.85, 1.15, len(player_stats))
        
        # Save projections
        output_file = os.path.join(prediction_dir, "player_projections_2025.csv")
        player_stats.to_csv(output_file, index=False)
        print(f"Saved projections for {len(player_stats)} players to {output_file}")
        return player_stats
    else:
        print("No player stats found. Creating synthetic player projections...")
        
        # Create synthetic player data
        player_ids = range(10001, 10201)  # 200 players
        
        synthetic_data = []
        for player_id in player_ids:
            # Create realistic but random stats
            hit_rate = np.random.beta(7, 17)  # Beta distribution centered around ~0.30
            hr_rate = np.random.beta(1, 20)   # Beta distribution for HR rates (~0.05)
            games = np.random.randint(20, 162)
            at_bats = games * np.random.randint(3, 5)  # 3-4 at bats per game
            hits = int(at_bats * hit_rate)
            home_runs = int(at_bats * hr_rate)
            
            synthetic_data.append({
                'player_id': str(player_id),
                'player_name': f"Player {player_id}",
                'games': games,
                'at_bats': at_bats,
                'hits': hits,
                'home_runs': home_runs,
                'hit_rate': hit_rate,
                'hr_rate': hr_rate,
                'batting_avg': hits / at_bats,
                'projected_hit_rate_2025': hit_rate * np.random.uniform(0.9, 1.1),
                'projected_hr_rate_2025': hr_rate * np.random.uniform(0.85, 1.15)
            })
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Save projections
        output_file = os.path.join(prediction_dir, "player_projections_2025.csv")
        synthetic_df.to_csv(output_file, index=False)
        print(f"Saved synthetic projections for {len(synthetic_df)} players to {output_file}")
        return synthetic_df

if __name__ == "__main__":
    create_player_projections()
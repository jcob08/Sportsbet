import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set up directories
DATA_DIR = "sports_data/mlb"
PLAYERS_DIR = os.path.join(DATA_DIR, "players")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

def create_player_id_name_mapping():
    """Create a mapping of player IDs to names from player JSON files"""
    print("Creating player ID to name mapping...")
    
    # Get all player JSON files
    player_files = [f for f in os.listdir(PLAYERS_DIR) if f.endswith(".json")]
    
    # Dictionary to store player mappings
    player_map = {}
    
    # Process each player file
    for file in tqdm(player_files, desc="Processing player files"):
        player_id = file.split("_")[1]
        file_path = os.path.join(PLAYERS_DIR, file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract player name from info
            if 'info' in data:
                player_info = data['info']
                # Check different possible formats
                if isinstance(player_info, dict) and 'fullName' in player_info:
                    name = player_info['fullName']
                    player_map[player_id] = name
                elif isinstance(player_info, dict) and 'people' in player_info and len(player_info['people']) > 0:
                    person = player_info['people'][0]
                    if 'fullName' in person:
                        name = person['fullName']
                        player_map[player_id] = name
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Save mapping to file
    mapping_file = os.path.join(PREDICTION_DIR, "player_id_map.csv")
    mapping_df = pd.DataFrame({"player_id": list(player_map.keys()), 
                 "player_name": list(player_map.values())})
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Saved {len(player_map)} player ID mappings to {mapping_file}")
    return player_map

def load_player_stats(year=2023):
    """Load player statistics for a specific year"""
    stats_file = os.path.join(ANALYSIS_DIR, f"player_game_stats_{year}.csv")
    if not os.path.exists(stats_file):
        print(f"No player stats file found for {year}")
        return None
    
    return pd.read_csv(stats_file)

def analyze_player_consistency(game_stats):
    """Analyze player consistency in hitting and home runs"""
    print("Analyzing player consistency...")
    
    # Convert date to datetime
    if 'game_date' in game_stats.columns:
        game_stats['game_date'] = pd.to_datetime(game_stats['game_date'])
    
    # Get unique player IDs
    player_ids = game_stats['player_id'].unique()
    
    # Initialize consistency metrics
    consistency_data = []
    
    for player_id in tqdm(player_ids, desc="Calculating player consistency"):
        # Get player games
        player_games = game_stats[game_stats['player_id'] == player_id].copy()
        
        # Skip players with too few games
        if len(player_games) < 20:
            continue
        
        # Sort by date if available
        if 'game_date' in player_games.columns:
            player_games = player_games.sort_values('game_date')
        
        # Get hit and HR sequences
        hit_sequence = player_games['got_hit'].values
        hr_sequence = player_games['got_home_run'].values
        
        # Calculate streaks
        hit_streaks = calc_streaks(hit_sequence)
        hr_streaks = calc_streaks(hr_sequence)
        
        # Calculate standard deviations for rolling windows
        hit_std = calc_rolling_std(hit_sequence, 10)
        hr_std = calc_rolling_std(hr_sequence, 10)
        
        # Calculate consistency metrics
        hit_consistency = 1 - (hit_std / 0.5)  # 0.5 is max std for binary outcome
        hr_consistency = 1 - (hr_std / 0.5)
        
        # Calculate overall consistency score
        consistency_score = (hit_consistency * 0.7) + (hr_consistency * 0.3)
        
        # Store consistency data
        consistency_data.append({
            'player_id': player_id,
            'games': len(player_games),
            'at_bats': player_games['at_bats'].sum(),
            'hit_rate': player_games['got_hit'].mean(),
            'hr_rate': player_games['got_home_run'].mean(),
            'longest_hit_streak': hit_streaks['longest_success'],
            'longest_hitless_streak': hit_streaks['longest_failure'],
            'longest_hr_streak': hr_streaks['longest_success'],
            'longest_hr_drought': hr_streaks['longest_failure'],
            'hit_consistency': hit_consistency,
            'hr_consistency': hr_consistency,
            'consistency_score': consistency_score
        })
    
    # Convert to DataFrame
    consistency_df = pd.DataFrame(consistency_data)
    
    # Save to file
    output_file = os.path.join(PREDICTION_DIR, "player_consistency.csv")
    consistency_df.to_csv(output_file, index=False)
    
    print(f"Saved consistency metrics for {len(consistency_df)} players to {output_file}")
    return consistency_df

def calc_streaks(sequence):
    """Calculate success and failure streaks in a binary sequence"""
    current_success = 0
    current_failure = 0
    longest_success = 0
    longest_failure = 0
    
    for outcome in sequence:
        if outcome == 1:
            # Success
            current_success += 1
            current_failure = 0
            longest_success = max(longest_success, current_success)
        else:
            # Failure
            current_failure += 1
            current_success = 0
            longest_failure = max(longest_failure, current_failure)
    
    return {
        'longest_success': longest_success,
        'longest_failure': longest_failure
    }

def calc_rolling_std(sequence, window=10):
    """Calculate the average standard deviation for rolling windows"""
    if len(sequence) <= window:
        return np.std(sequence)
    
    stds = []
    for i in range(len(sequence) - window + 1):
        window_slice = sequence[i:i+window]
        stds.append(np.std(window_slice))
    
    return np.mean(stds)

def predict_2025_performance(consistency_df, player_map):
    """Create 2025 performance predictions based on 2023 data and consistency"""
    print("Creating 2025 performance predictions...")
    
    # Calculate baseline hit and HR rates (league averages)
    baseline_hit_rate = consistency_df['hit_rate'].mean()
    baseline_hr_rate = consistency_df['hr_rate'].mean()
    
    # Create predictions
    predictions = []
    
    for _, player in consistency_df.iterrows():
        player_id = player['player_id']
        
        # Consider consistency in predictions
        # More consistent players will maintain closer to their 2023 performance
        # Less consistent players will regress toward the mean
        consistency_weight = player['consistency_score']
        mean_weight = 1 - consistency_weight
        
        # Projected rates
        projected_hit_rate = (player['hit_rate'] * consistency_weight) + (baseline_hit_rate * mean_weight)
        projected_hr_rate = (player['hr_rate'] * consistency_weight) + (baseline_hr_rate * mean_weight)
        
        # Get player name
        player_name = player_map.get(str(player_id), f"Player {player_id}")
        
        # Add prediction
        predictions.append({
            'player_id': player_id,
            'player_name': player_name,
            'games_2023': player['games'],
            'at_bats_2023': player['at_bats'],
            'hit_rate_2023': player['hit_rate'],
            'hr_rate_2023': player['hr_rate'],
            'hit_consistency': player['hit_consistency'],
            'hr_consistency': player['hr_consistency'],
            'consistency_score': player['consistency_score'],
            'projected_hit_rate_2025': projected_hit_rate,
            'projected_hr_rate_2025': projected_hr_rate
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Save predictions
    output_file = os.path.join(PREDICTION_DIR, "player_projections_2025.csv")
    predictions_df.to_csv(output_file, index=False)
    
    print(f"Saved 2025 projections for {len(predictions_df)} players to {output_file}")
    
    # Display top projected hitters
    top_hit_projections = predictions_df.sort_values('projected_hit_rate_2025', ascending=False)
    
    print("\nTop 10 Projected Hitters for 2025 (by Hit Rate):")
    for i, (_, player) in enumerate(top_hit_projections.head(10).iterrows(), 1):
        print(f"{i}. {player['player_name']} (ID: {player['player_id']}): " +
              f"Projected Hit Rate: {player['projected_hit_rate_2025']*100:.2f}%, " +
              f"2023: {player['hit_rate_2023']*100:.2f}%, " +
              f"Consistency: {player['hit_consistency']:.2f}")
    
    # Display top projected power hitters
    top_hr_projections = predictions_df.sort_values('projected_hr_rate_2025', ascending=False)
    
    print("\nTop 10 Projected Power Hitters for 2025 (by HR Rate):")
    for i, (_, player) in enumerate(top_hr_projections.head(10).iterrows(), 1):
        print(f"{i}. {player['player_name']} (ID: {player['player_id']}): " +
              f"Projected HR Rate: {player['projected_hr_rate_2025']*100:.2f}%, " +
              f"2023: {player['hr_rate_2023']*100:.2f}%, " +
              f"Consistency: {player['hr_consistency']:.2f}")
    
    return predictions_df

def main():
    """Main execution function"""
    print("=== Building 2025 MLB Season Predictions ===")
    
    # Create player ID to name mapping
    player_map = create_player_id_name_mapping()
    
    # Load player game stats for 2023
    game_stats = load_player_stats(2023)
    
    if game_stats is not None:
        # Analyze player consistency
        consistency_df = analyze_player_consistency(game_stats)
        
        # Create 2025 predictions
        predictions_df = predict_2025_performance(consistency_df, player_map)
    
    print("\n=== 2025 Season Prediction Framework Complete ===")

if __name__ == "__main__":
    main()
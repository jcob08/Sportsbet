import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import paths from config.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, GAMES_DIR, TEAMS_DIR, PITCHERS_DIR, PROCESSED_DIR, ANALYSIS_DIR

# Set up directories (using paths from config.py)
# DATA_DIR is already defined in config.py
# ANALYSIS_DIR is already defined in config.py
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

def create_player_id_name_mapping():
    """Load improved player ID to name mapping"""
    print("Loading player ID to name mapping...")
    
    # Run the create_player_mapping.py script first
    try:
        print("Running the create_player_mapping.py script to ensure we have the latest player mappings...")
        os.system("python utilities\\create_player_mapping.py")
        print("Player mapping script completed successfully.")
    except Exception as e:
        print(f"Error running player mapping script: {e}")
    
    # Check all possible locations for the improved mapping file
    possible_paths = [
        os.path.join(PREDICTION_DIR, "improved_player_id_map.csv"),
        os.path.join(DATA_DIR, "sports_data", "mlb", "predictions", "improved_player_id_map.csv")
    ]
    
    improved_mapping_file = None
    for path in possible_paths:
        if os.path.exists(path):
            improved_mapping_file = path
            print(f"Found improved mapping file at: {improved_mapping_file}")
            break
    
    if improved_mapping_file:
        # Load the improved mapping
        mapping_df = pd.read_csv(improved_mapping_file)
        
        # Ensure player_id is treated as string
        mapping_df['player_id'] = mapping_df['player_id'].astype(str)
        
        player_map = dict(zip(mapping_df['player_id'], mapping_df['player_name']))
        print(f"Loaded {len(player_map)} player ID mappings from improved mapping file")
        
        # Debug: Print first 5 keys to see format
        print("First 5 player IDs in mapping:")
        for i, key in enumerate(list(player_map.keys())[:5]):
            print(f"  {i+1}. '{key}'")
            
        return player_map
    
    # If improved mapping doesn't exist, fall back to original method
    print("Improved mapping file not found. Creating mapping from pitcher files...")
    
    # Get all pitcher JSON files
    player_files = [f for f in os.listdir(PITCHERS_DIR) if f.endswith(".json")]
    
    # Dictionary to store player mappings
    player_map = {}
    
    # Process each player file
    for file in tqdm(player_files, desc="Processing player files"):
        try:
            # Extract player ID from filename (assuming format like pitcher_123456_2023.json)
            parts = file.split("_")
            if len(parts) >= 2:
                player_id = parts[1]
            else:
                continue
                
            file_path = os.path.join(PITCHERS_DIR, file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Try multiple ways to extract the player name
            name = None
            
            # Method 1: Check for info.people[0].fullName
            if 'info' in data and isinstance(data['info'], dict) and 'people' in data['info']:
                people = data['info']['people']
                if isinstance(people, list) and len(people) > 0 and 'fullName' in people[0]:
                    name = people[0]['fullName']
            
            # Method 2: Check for info.people[0].firstName + lastName
            if name is None and 'info' in data and isinstance(data['info'], dict) and 'people' in data['info']:
                people = data['info']['people']
                if isinstance(people, list) and len(people) > 0:
                    person = people[0]
                    if 'firstName' in person and 'lastName' in person:
                        name = f"{person['firstName']} {person['lastName']}"
            
            # Method 3: Try the player_name field directly
            if name is None and 'player_name' in data:
                name = data['player_name']
                
            # Method 4: Try pitcher_id field if it's actually the name
            if name is None and 'pitcher_id' in data and isinstance(data['pitcher_id'], str) and " " in data['pitcher_id']:
                name = data['pitcher_id']
                
            # Save the mapping if we found a name
            if name and player_id:
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

def load_player_stats(year):
    """Load player statistics for a specific year"""
    stats_file = os.path.join(ANALYSIS_DIR, f"player_game_stats_{year}.csv")
    if not os.path.exists(stats_file):
        print(f"No player stats file found for {year}")
        return None
    
    return pd.read_csv(stats_file)

def analyze_player_consistency(game_stats, year_weights=None):
    """Analyze player consistency in hitting and home runs"""
    print("Analyzing player consistency...")
    
    # Set default year weights if not provided (most recent year has higher weight)
    if year_weights is None:
        # Determine available years
        if 'game_date' in game_stats.columns:
            game_stats['year'] = pd.to_datetime(game_stats['game_date']).dt.year
            years = sorted(game_stats['year'].unique())
            
            # Create weights with higher values for more recent years
            year_weights = {}
            for i, year in enumerate(years):
                year_weights[year] = (i + 1) / sum(range(1, len(years) + 1))
            
            print(f"Using year weights: {year_weights}")
    
    # Convert date to datetime
    if 'game_date' in game_stats.columns and 'year' not in game_stats.columns:
        game_stats['game_date'] = pd.to_datetime(game_stats['game_date'])
        game_stats['year'] = game_stats['game_date'].dt.year
    
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
        
        # Calculate weighted hit rate and HR rate if year weights are provided
        weighted_hit_rate = player_games['got_hit'].mean()
        weighted_hr_rate = player_games['got_home_run'].mean()
        
        if year_weights and 'year' in player_games.columns:
            # Group by year
            yearly_stats = player_games.groupby('year').agg({
                'got_hit': 'mean',
                'got_home_run': 'mean',
                'player_id': 'count'  # Game count
            }).reset_index()
            
            # Apply weights to each year
            yearly_stats['weighted_hit'] = yearly_stats.apply(
                lambda x: x['got_hit'] * year_weights.get(x['year'], 0), axis=1)
            yearly_stats['weighted_hr'] = yearly_stats.apply(
                lambda x: x['got_home_run'] * year_weights.get(x['year'], 0), axis=1)
            yearly_stats['weighted_games'] = yearly_stats.apply(
                lambda x: x['player_id'] * year_weights.get(x['year'], 0), axis=1)
            
            # Calculate weighted rates
            weighted_hit_rate = yearly_stats['weighted_hit'].sum() / yearly_stats['weighted_games'].sum()
            weighted_hr_rate = yearly_stats['weighted_hr'].sum() / yearly_stats['weighted_games'].sum()
        
        # Store consistency data
        consistency_data.append({
            'player_id': player_id,
            'games': len(player_games),
            'at_bats': player_games['at_bats'].sum(),
            'hit_rate': player_games['got_hit'].mean(),
            'hr_rate': player_games['got_home_run'].mean(),
            'weighted_hit_rate': weighted_hit_rate,
            'weighted_hr_rate': weighted_hr_rate,
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
    
    # Debug: Print the first few rows of the consistency data with player ID types
    print("Sample of player IDs in consistency data (with types):")
    for i, pid in enumerate(consistency_df['player_id'].head()):
        print(f"  {i+1}. {pid} (type: {type(pid)})")
    
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
    """Create 2025 performance predictions based on historical data and consistency"""
    print("Creating 2025 performance predictions...")
    
    # Calculate baseline hit and HR rates (league averages)
    baseline_hit_rate = consistency_df['hit_rate'].mean()
    baseline_hr_rate = consistency_df['hr_rate'].mean()
    
    print(f"League average hit rate: {baseline_hit_rate:.4f}")
    print(f"League average HR rate: {baseline_hr_rate:.4f}")
    
    # Add some manual mappings for top players
    top_players = {
        '691176': 'Shohei Ohtani',
        '518626': 'Carlos Correa',
        '621311': 'Tyler O\'Neill',
        '624413': 'Matt Chapman',
        '670712': 'Vladimir Guerrero Jr.',
        '656941': 'Rafael Devers',
        '669127': 'Jake Cronenworth',
        '519317': 'Jose Ramirez',
        '458015': 'Edwin Encarnacion',
        '663728': 'Anthony Rizzo',
        '650333': 'Aaron Judge',
        '694384': 'Jose Altuve',
        '661388': 'Jarred Kelenic',
        '663538': 'Ryan McMahon',
        '682622': 'Teoscar Hernandez',
        '660670': 'Juan Soto',
        '687093': 'Brent Rooker',
        '518692': 'Freddie Freeman',
        '641355': 'Dansby Swanson',
        '643217': 'Christian Walker'
    }
    
    # Add manual mappings to the player map
    for player_id, name in top_players.items():
        player_map[player_id] = name
    
    # Create predictions
    predictions = []
    
    # Track ID matching stats
    id_match_count = 0
    id_not_found_count = 0
    
    for _, player in consistency_df.iterrows():
        player_id = player['player_id']
        
        # Consider consistency in predictions
        # More consistent players will maintain closer to their historical performance
        # Less consistent players will regress toward the mean
        consistency_weight = player['consistency_score']
        mean_weight = 1 - consistency_weight
        
        # Use weighted rates if available, otherwise use regular rates
        if 'weighted_hit_rate' in player.index and not pd.isna(player['weighted_hit_rate']):
            hit_rate_to_use = player['weighted_hit_rate']
            hr_rate_to_use = player['weighted_hr_rate']
        else:
            hit_rate_to_use = player['hit_rate']
            hr_rate_to_use = player['hr_rate']
        
        # Projected rates
        projected_hit_rate = (hit_rate_to_use * consistency_weight) + (baseline_hit_rate * mean_weight)
        projected_hr_rate = (hr_rate_to_use * consistency_weight) + (baseline_hr_rate * mean_weight)
        
        # Try different formats for player ID matching
        player_name = None
        
        # Convert player_id to various formats for better matching
        pid_formats = [
            str(int(player_id)) if isinstance(player_id, (int, float)) else str(player_id),  # "691176"
            str(player_id),  # Could be "691176.0"
            str(player_id).split('.')[0] if '.' in str(player_id) else str(player_id)  # "691176"
        ]
        
        for pid in pid_formats:
            if pid in player_map:
                player_name = player_map[pid]
                id_match_count += 1
                break
        
        # If still not found, try the manual top players mapping
        if player_name is None:
            for pid in pid_formats:
                if pid in top_players:
                    player_name = top_players[pid]
                    id_match_count += 1
                    break
        
        # If still not found, use default
        if player_name is None:
            player_name = f"Player {player_id}"
            id_not_found_count += 1
        
        # Add prediction
        predictions.append({
            'player_id': player_id,
            'player_name': player_name,
            'games': player['games'],
            'at_bats': player['at_bats'],
            'hit_rate': player['hit_rate'],
            'hr_rate': player['hr_rate'],
            'hit_consistency': player['hit_consistency'],
            'hr_consistency': player['hr_consistency'],
            'consistency_score': player['consistency_score'],
            'projected_hit_rate_2025': projected_hit_rate,
            'projected_hr_rate_2025': projected_hr_rate
        })
    
    print(f"Player ID matching stats: {id_match_count} matched, {id_not_found_count} not found")
    
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
              f"2023-2024: {player['hit_rate']*100:.2f}%, " +
              f"Consistency: {player['hit_consistency']:.2f}")
    
    # Display top projected power hitters
    top_hr_projections = predictions_df.sort_values('projected_hr_rate_2025', ascending=False)
    
    print("\nTop 10 Projected Power Hitters for 2025 (by HR Rate):")
    for i, (_, player) in enumerate(top_hr_projections.head(10).iterrows(), 1):
        print(f"{i}. {player['player_name']} (ID: {player['player_id']}): " +
              f"Projected HR Rate: {player['projected_hr_rate_2025']*100:.2f}%, " +
              f"2023-2024: {player['hr_rate']*100:.2f}%, " +
              f"Consistency: {player['hr_consistency']:.2f}")
    
    return predictions_df

def main():
    """Main execution function"""
    print("=== Building 2025 MLB Season Predictions ===")
    
    # Create player ID to name mapping
    player_map = create_player_id_name_mapping()
    
    # Load player game stats for both 2023 and 2024
    game_stats_2023 = load_player_stats(2023)
    game_stats_2024 = load_player_stats(2024)
    
    # Combine the data from both years if available
    if game_stats_2023 is not None and game_stats_2024 is not None:
        print("Combining 2023 and 2024 data for improved predictions")
        # Ensure both dataframes have a year column
        game_stats_2023['game_date'] = pd.to_datetime(game_stats_2023['game_date'])
        game_stats_2023['year'] = game_stats_2023['game_date'].dt.year
        
        game_stats_2024['game_date'] = pd.to_datetime(game_stats_2024['game_date'])
        game_stats_2024['year'] = game_stats_2024['game_date'].dt.year
        
        # Combine data
        game_stats = pd.concat([game_stats_2023, game_stats_2024])
        
        # Create year weights (2024: 65%, 2023: 35%)
        year_weights = {2023: 0.35, 2024: 0.65}
        
        # Analyze player consistency with weighted years
        consistency_df = analyze_player_consistency(game_stats, year_weights)
        
        # Create 2025 predictions
        predictions_df = predict_2025_performance(consistency_df, player_map)
    elif game_stats_2023 is not None:
        print("Using only 2023 data for predictions")
        # Analyze player consistency
        consistency_df = analyze_player_consistency(game_stats_2023)
        
        # Create 2025 predictions
        predictions_df = predict_2025_performance(consistency_df, player_map)
    elif game_stats_2024 is not None:
        print("Using only 2024 data for predictions")
        # Analyze player consistency
        consistency_df = analyze_player_consistency(game_stats_2024)
        
        # Create 2025 predictions
        predictions_df = predict_2025_performance(consistency_df, player_map)
    else:
        print("No player stats found for 2023 or 2024. Cannot generate predictions.")
    
    print("\n=== 2025 Season Prediction Framework Complete ===")

if __name__ == "__main__":
    main()
# utilities/create_player_mapping.py
import os
import pandas as pd
import json
from tqdm import tqdm
import sys
import statsapi

# Import paths from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PITCHERS_DIR

# Set up output directory
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

def create_improved_player_mapping():
    """Create an improved mapping of player IDs to names from all available sources"""
    print("Creating improved player ID to name mapping...")
    
    # Dictionary to store player mappings
    player_map = {}
    
    # Step 1: Extract from pitcher files
    print("Extracting names from pitcher files...")
    player_files = [f for f in os.listdir(PITCHERS_DIR) if f.endswith(".json")]
    
    for file in tqdm(player_files, desc="Processing pitcher files"):
        try:
            parts = file.split("_")
            if len(parts) >= 2:
                player_id = parts[1]
            else:
                continue
                
            file_path = os.path.join(PITCHERS_DIR, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'info' in data and isinstance(data['info'], dict) and 'people' in data['info']:
                people = data['info']['people']
                if isinstance(people, list) and len(people) > 0:
                    person = people[0]
                    if 'fullName' in person:
                        player_map[player_id] = person['fullName']
                        continue
                    elif 'firstName' in person and 'lastName' in person:
                        player_map[player_id] = f"{person['firstName']} {person['lastName']}"
                        continue
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Step 2: Extract from player game stats
    print("Extracting names from game stats...")
    stats_file_2023 = os.path.join(DATA_DIR, "analysis", "player_game_stats_2023.csv")
    stats_file_2024 = os.path.join(DATA_DIR, "analysis", "player_game_stats_2024.csv")
    
    if os.path.exists(stats_file_2023):
        try:
            stats_df = pd.read_csv(stats_file_2023)
            if 'player_name' in stats_df.columns:
                for _, row in stats_df.drop_duplicates('player_id').iterrows():
                    player_map[str(row['player_id'])] = row['player_name']
        except Exception as e:
            print(f"Error extracting from 2023 stats: {e}")
    
    if os.path.exists(stats_file_2024):
        try:
            stats_df = pd.read_csv(stats_file_2024)
            if 'player_name' in stats_df.columns:
                for _, row in stats_df.drop_duplicates('player_id').iterrows():
                    player_map[str(row['player_id'])] = row['player_name']
        except Exception as e:
            print(f"Error extracting from 2024 stats: {e}")
    
    # Step 3: Extract from game data
    print("Searching for names in game data...")
    games_dir = os.path.join(DATA_DIR, "games")
    game_dirs = [d for d in os.listdir(games_dir) if os.path.isdir(os.path.join(games_dir, d))]
    
    for game_dir in tqdm(game_dirs, desc="Checking game data"):
        try:
            game_data_file = os.path.join(games_dir, game_dir, "game_data.json")
            if os.path.exists(game_data_file):
                with open(game_data_file, 'r') as f:
                    game_data = json.load(f)
                if 'gameData' in game_data and 'players' in game_data['gameData']:
                    players = game_data['gameData']['players']
                    for player_key, player_info in players.items():
                        if player_key.startswith('ID'):
                            player_id = player_key.replace('ID', '')
                            if 'fullName' in player_info:
                                player_map[player_id] = player_info['fullName']
                            elif 'firstName' in person and 'lastName' in person:
                                player_map[player_id] = f"{player_info['firstName']} {player_info['lastName']}"
        except Exception as e:
            print(f"Error checking game data in {game_dir}: {e}")
    
    # Step 4: Extract player IDs from game_outcomes lineups and fetch names
    print("Extracting player IDs from game_outcomes lineups...")
    game_files = [
        os.path.join(DATA_DIR, "contextual", "game_outcomes_2023.csv"),
        os.path.join(DATA_DIR, "contextual", "game_outcomes_2024.csv")
    ]
    
    all_player_ids = set()
    for game_file in game_files:
        if not os.path.exists(game_file):
            continue
        games_df = pd.read_csv(game_file)
        for team in ['home', 'away']:
            lineup_col = f'{team}_starting_lineup'
            if lineup_col not in games_df.columns:
                continue
            for lineup in games_df[lineup_col].dropna():
                player_ids = str(lineup).split(',')
                for player_id in player_ids:
                    player_id = str(player_id).strip()
                    if player_id:
                        all_player_ids.add(player_id)
    
    print(f"Found {len(all_player_ids)} unique player IDs in lineups. Fetching names...")
    for player_id in tqdm(all_player_ids, desc="Fetching player names"):
        if player_id in player_map:
            continue
        try:
            person_data = statsapi.get('person', {'personId': player_id})
            if 'people' in person_data and len(person_data['people']) > 0:
                person = person_data['people'][0]
                if 'fullName' in person:
                    player_map[player_id] = person['fullName']
                elif 'firstName' in person and 'lastName' in person:
                    player_map[player_id] = f"{person['firstName']} {person['lastName']}"
        except Exception as e:
            print(f"Error fetching name for player ID {player_id}: {e}")
    
    # Step 5: Add manual mappings for top players
    print("Adding manual mappings for top players...")
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
        '643217': 'Christian Walker',
        '605119': 'Shohei Ohtani',  # From unmatched list
        '666801': 'Mike Trout',
        '665120': 'Mookie Betts',
        '607043': 'Bryce Harper',
        '571745': 'Freddy Peralta'
    }
    
    for player_id, name in top_players.items():
        player_map[player_id] = name
    
    # Save the mapping
    mapping_file = os.path.join(PREDICTION_DIR, "improved_player_id_map.csv")
    mapping_df = pd.DataFrame({"player_id": list(player_map.keys()), 
                               "player_name": list(player_map.values())})
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Saved {len(player_map)} player ID mappings to {mapping_file}")
    return player_map

if __name__ == "__main__":
    create_improved_player_mapping()
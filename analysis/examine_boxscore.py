import os
import json
import pandas as pd

# Set up directories
DATA_DIR = "sports_data/mlb"
GAMES_DIR = os.path.join(DATA_DIR, "games")

def examine_boxscore_structure():
    """Examine the structure of a single boxscore file to understand the data format"""
    print("Examining boxscore file structure...")
    
    # Get the first game directory
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    if not game_dirs:
        print("No game directories found.")
        return
    
    # Select first directory
    game_dir = game_dirs[0]
    print(f"Examining game directory: {game_dir}")
    
    # Check for boxscore file
    boxscore_file = os.path.join(GAMES_DIR, game_dir, "boxscore.json")
    
    if not os.path.exists(boxscore_file):
        print(f"No boxscore file found in {game_dir}")
        return
    
    # Load and examine the file
    try:
        with open(boxscore_file, 'r') as f:
            boxscore = json.load(f)
        
        print(f"\nBoxscore keys: {list(boxscore.keys())}")
        
        # Examine home team structure
        if 'home' in boxscore:
            home = boxscore['home']
            print(f"\nHome team keys: {list(home.keys())}")
            
            # Check for batting stats
            if 'battingStats' in home:
                # Get first player's batting stats
                batting_stats = home['battingStats']
                
                if batting_stats:
                    player_id = list(batting_stats.keys())[0]
                    print(f"\nExample player ID: {player_id}")
                    print(f"Player batting stats keys: {list(batting_stats[player_id].keys())}")
                    print(f"Player batting stats: {batting_stats[player_id]}")
                else:
                    print("No batting stats found for home team.")
            else:
                print("No 'battingStats' key in home team data.")
        else:
            print("No 'home' key in boxscore data.")
        
        # Save a sample boxscore for examination
        sample_file = "sample_boxscore.json"
        with open(sample_file, 'w') as f:
            json.dump(boxscore, f, indent=4)
        
        print(f"\nSaved sample boxscore to {sample_file} for detailed examination.")
        
    except Exception as e:
        print(f"Error examining boxscore file: {e}")

# Run the analysis
examine_boxscore_structure()
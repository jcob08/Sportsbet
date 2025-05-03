import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import collections

# Set up directories
DATA_DIR = "sports_data/mlb"
GAMES_DIR = os.path.join(DATA_DIR, "games")
TEAMS_DIR = os.path.join(DATA_DIR, "teams")
PITCHERS_DIR = os.path.join(DATA_DIR, "pitchers")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def analyze_directory_structure():
    """Check the existing directory structure and files"""
    print("=== Analyzing MLB Data Directory Structure ===")
    
    # Check if base directories exist
    directories = {
        "DATA_DIR": DATA_DIR,
        "GAMES_DIR": GAMES_DIR,
        "TEAMS_DIR": TEAMS_DIR,
        "PITCHERS_DIR": PITCHERS_DIR,
        "PROCESSED_DIR": PROCESSED_DIR
    }
    
    for name, directory in directories.items():
        if os.path.exists(directory):
            print(f"{name} exists: {directory}")
            if os.path.isdir(directory):
                num_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
                num_dirs = len([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
                print(f"  - Contains {num_files} files and {num_dirs} subdirectories")
            else:
                print(f"  - WARNING: {directory} is not a directory")
        else:
            print(f"{name} does not exist: {directory}")

def analyze_team_data():
    """Analyze team data files"""
    print("\n=== Analyzing Team Data ===")
    
    if not os.path.exists(TEAMS_DIR):
        print(f"Teams directory does not exist: {TEAMS_DIR}")
        return
    
    # Check team files
    team_files = [f for f in os.listdir(TEAMS_DIR) if os.path.isfile(os.path.join(TEAMS_DIR, f))]
    
    # Group files by year
    years_data = collections.defaultdict(list)
    for file in team_files:
        if file.endswith(".csv") or file.endswith(".json") or file.endswith(".txt"):
            if "_2023" in file:
                years_data[2023].append(file)
            elif "_2024" in file:
                years_data[2024].append(file)
            elif "_2025" in file:
                years_data[2025].append(file)
    
    # Display summary of files by year
    for year, files in sorted(years_data.items()):
        print(f"\nYear {year} - {len(files)} files:")
        
        # Count file types
        file_types = collections.Counter([f.split('_')[0] for f in files])
        for file_type, count in file_types.items():
            print(f"  - {file_type}: {count} files")
        
        # If team file exists, count teams
        teams_file = f"teams_{year}.csv"
        if teams_file in files:
            try:
                teams_df = pd.read_csv(os.path.join(TEAMS_DIR, teams_file))
                print(f"  - Number of teams: {len(teams_df)}")
            except Exception as e:
                print(f"  - Error reading teams file: {e}")

def analyze_game_data():
    """Analyze game data files"""
    print("\n=== Analyzing Game Data ===")
    
    if not os.path.exists(GAMES_DIR):
        print(f"Games directory does not exist: {GAMES_DIR}")
        return
    
    # Get all game directories
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    print(f"Found {len(game_dirs)} game directories")
    
    # Initialize counters
    years_count = collections.defaultdict(int)
    years_with_boxscore = collections.defaultdict(int)
    
    # Sample a few games to check their structure
    sample_size = min(20, len(game_dirs))
    sample_dirs = game_dirs[:sample_size]
    
    print(f"\nAnalyzing a sample of {sample_size} games:")
    
    for game_dir in sample_dirs:
        game_path = os.path.join(GAMES_DIR, game_dir)
        files = [f for f in os.listdir(game_path) if os.path.isfile(os.path.join(game_path, f))]
        
        print(f"\nGame directory: {game_dir}")
        print(f"  - Contains {len(files)} files: {', '.join(files)}")
        
        # Check game date/year
        game_data_file = os.path.join(game_path, "game_data.json")
        if os.path.exists(game_data_file):
            try:
                with open(game_data_file, 'r') as f:
                    game_data = json.load(f)
                    game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                    if game_date:
                        year = game_date.split('-')[0]
                        print(f"  - Game date: {game_date} (Year: {year})")
                    else:
                        print("  - No game date found")
            except Exception as e:
                print(f"  - Error reading game data: {e}")
    
    # Count games by year
    print("\nCounting games by year...")
    
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        game_data_file = os.path.join(GAMES_DIR, game_dir, "game_data.json")
        boxscore_file = os.path.join(GAMES_DIR, game_dir, "boxscore.json")
        
        if os.path.exists(game_data_file):
            try:
                with open(game_data_file, 'r') as f:
                    game_data = json.load(f)
                    game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                    if game_date:
                        year = int(game_date.split('-')[0])
                        years_count[year] += 1
                        
                        # Check if boxscore exists
                        if os.path.exists(boxscore_file):
                            years_with_boxscore[year] += 1
            except:
                continue
    
    # Display summary of games by year
    print("\nSummary of games by year:")
    for year, count in sorted(years_count.items()):
        boxscore_count = years_with_boxscore[year]
        print(f"  - Year {year}: {count} games, {boxscore_count} with boxscores ({boxscore_count/count*100:.1f}%)")

def main():
    """Main function to analyze the MLB data directory structure"""
    print("=== MLB Data Directory Analysis ===")
    
    analyze_directory_structure()
    analyze_team_data()
    analyze_game_data()
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
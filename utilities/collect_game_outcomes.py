# utilities/collect_game_outcomes.py

import os
import json
import pandas as pd
import statsapi
from tqdm import tqdm

def collect_game_outcomes(year, games_dir="sports_data/mlb/games", output_dir="sports_data/mlb/contextual"):
    """
    Collect actual game outcomes for MLB games.
    
    Args:
        year: Year to collect outcomes for
        games_dir: Directory containing game data
        output_dir: Directory to save outcomes data
        
    Returns:
        DataFrame: Game outcomes data
    """
    print(f"Collecting game outcomes for {year}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of game directories
    if os.path.exists(games_dir):
        game_dirs = [d for d in os.listdir(games_dir) if os.path.isdir(os.path.join(games_dir, d))]
    else:
        print(f"Error: Games directory {games_dir} not found.")
        return pd.DataFrame()
    
    # Initialize outcomes list
    outcomes = []
    
    # Process each game directory
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        # Extract game ID from directory name
        game_id = game_dir.replace("game_", "")
        
        # Path to game data file
        game_data_file = os.path.join(games_dir, game_dir, "game_data.json")
        
        # Skip if game data file doesn't exist
        if not os.path.exists(game_data_file):
            continue
        
        # Load game data to check year
        try:
            with open(game_data_file, 'r') as f:
                game_data = json.load(f)
                game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate', '')
                
                # Skip if game is not from the specified year
                if not game_date or not game_date.startswith(str(year)):
                    continue
        except Exception as e:
            print(f"Error reading game data file {game_data_file}: {e}")
            continue
        
        # Fetch game outcomes from statsapi
        try:
            # Get game data from MLB Stats API
            game = statsapi.get('game', {'gamePk': game_id})
            
            # Extract linescore data
            linescore = game.get('liveData', {}).get('linescore', {})
            
            # Extract teams data
            home_team = linescore.get('teams', {}).get('home', {})
            away_team = linescore.get('teams', {}).get('away', {})
            
            # Create outcome record
            outcome = {
                'game_id': game_id,
                'home_team_runs': home_team.get('runs', 0),
                'away_team_runs': away_team.get('runs', 0),
                'total_runs': home_team.get('runs', 0) + away_team.get('runs', 0),
                'home_team_win': 1 if home_team.get('runs', 0) > away_team.get('runs', 0) else 0,
                'run_differential': home_team.get('runs', 0) - away_team.get('runs', 0),
                'home_team_hits': home_team.get('hits', 0),
                'away_team_hits': away_team.get('hits', 0),
                'home_team_errors': home_team.get('errors', 0),
                'away_team_errors': away_team.get('errors', 0),
                'home_team_leftonbase': home_team.get('leftOnBase', 0),
                'away_team_leftonbase': away_team.get('leftOnBase', 0)
            }
            
            # Get boxscore for strikeout data
            boxscore = game.get('liveData', {}).get('boxscore', {})
            home_team_stats = boxscore.get('teams', {}).get('home', {}).get('teamStats', {}).get('pitching', {})
            away_team_stats = boxscore.get('teams', {}).get('away', {}).get('teamStats', {}).get('pitching', {})
            
            # Add strikeouts data
            outcome['home_team_strikeouts_pitched'] = home_team_stats.get('strikeOuts', 0)
            outcome['away_team_strikeouts_pitched'] = away_team_stats.get('strikeOuts', 0)
            outcome['total_strikeouts'] = home_team_stats.get('strikeOuts', 0) + away_team_stats.get('strikeOuts', 0)
            
            # Add to outcomes list
            outcomes.append(outcome)
            
        except Exception as e:
            print(f"Error fetching outcomes for game {game_id}: {e}")
    
    # Convert to DataFrame
    outcomes_df = pd.DataFrame(outcomes)
    
    # Save outcomes if we have data
    if not outcomes_df.empty:
        output_file = os.path.join(output_dir, f"game_outcomes_{year}.csv")
        outcomes_df.to_csv(output_file, index=False)
        print(f"Saved {len(outcomes_df)} game outcomes to {output_file}")
    else:
        print(f"No game outcomes found for {year}")
    
    return outcomes_df

if __name__ == "__main__":
    # Collect outcomes for recent years
    years = [2023, 2024]
    for year in years:
        collect_game_outcomes(year)
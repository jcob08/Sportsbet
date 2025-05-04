# utilities/collect_game_outcomes.py
import os
import json
import pandas as pd
import statsapi
from tqdm import tqdm

def collect_game_outcomes(year, games_dir="sports_data/mlb/games", output_dir="sports_data/mlb/contextual"):
    print(f"Collecting game outcomes for {year}")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(games_dir):
        game_dirs = [d for d in os.listdir(games_dir) if os.path.isdir(os.path.join(games_dir, d))]
    else:
        print(f"Error: Games directory {games_dir} not found.")
        return pd.DataFrame()
    
    outcomes = []
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        game_id = game_dir.replace("game_", "")
        game_data_file = os.path.join(games_dir, game_dir, "game_data.json")
        if not os.path.exists(game_data_file):
            continue
        
        try:
            with open(game_data_file, 'r') as f:
                game_data = json.load(f)
                game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate', '')
                if not game_date or not game_date.startswith(str(year)):
                    continue
        except Exception as e:
            print(f"Error reading game data file {game_data_file}: {e}")
            continue
        
        try:
            game = statsapi.get('game', {'gamePk': game_id})
            linescore = game.get('liveData', {}).get('linescore', {})
            home_team = linescore.get('teams', {}).get('home', {})
            away_team = linescore.get('teams', {}).get('away', {})
            
            # Get team abbreviations
            home_team_abbr = game.get('gameData', {}).get('teams', {}).get('home', {}).get('abbreviation', '')
            away_team_abbr = game.get('gameData', {}).get('teams', {}).get('away', {}).get('abbreviation', '')
            
            # Get starting pitchers and lineups
            boxscore = game.get('liveData', {}).get('boxscore', {})
            home_pitcher = boxscore.get('teams', {}).get('home', {}).get('pitchers', [None])[0]
            away_pitcher = boxscore.get('teams', {}).get('away', {}).get('pitchers', [None])[0]
            home_lineup = boxscore.get('teams', {}).get('home', {}).get('battingOrder', [])
            away_lineup = boxscore.get('teams', {}).get('away', {}).get('battingOrder', [])
            
            home_team_stats = boxscore.get('teams', {}).get('home', {}).get('teamStats', {}).get('pitching', {})
            away_team_stats = boxscore.get('teams', {}).get('away', {}).get('teamStats', {}).get('pitching', {})
            
            outcome = {
                'game_id': game_id,
                'home_team': home_team_abbr,
                'away_team': away_team_abbr,
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
                'away_team_leftonbase': away_team.get('leftOnBase', 0),
                'home_team_strikeouts_pitched': home_team_stats.get('strikeOuts', 0),
                'away_team_strikeouts_pitched': away_team_stats.get('strikeOuts', 0),
                'total_strikeouts': home_team_stats.get('strikeOuts', 0) + away_team_stats.get('strikeOuts', 0),
                'home_starting_pitcher_id': str(home_pitcher) if home_pitcher else None,
                'away_starting_pitcher_id': str(away_pitcher) if away_pitcher else None,
                'home_starting_lineup': ','.join(map(str, home_lineup)) if home_lineup else None,
                'away_starting_lineup': ','.join(map(str, away_lineup)) if away_lineup else None
            }
            outcomes.append(outcome)
        except Exception as e:
            print(f"Error fetching outcomes for game {game_id}: {e}")
    
    outcomes_df = pd.DataFrame(outcomes)
    if not outcomes_df.empty:
        output_file = os.path.join(output_dir, f"game_outcomes_{year}.csv")
        outcomes_df.to_csv(output_file, index=False)
        print(f"Saved {len(outcomes_df)} game outcomes to {output_file}")
    else:
        print(f"No game outcomes found for {year}")
    return outcomes_df

if __name__ == "__main__":
    years = [2023, 2024]
    for year in years:
        collect_game_outcomes(year)
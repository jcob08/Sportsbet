import os
import json
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime

# Import paths from config.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, GAMES_DIR, TEAMS_DIR, PITCHERS_DIR, PROCESSED_DIR, ANALYSIS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_data_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Comment out original directory paths since we're using config.py now
# DATA_DIR = "sports_data/mlb"
# TEAMS_DIR = os.path.join(DATA_DIR, "teams")
# GAMES_DIR = os.path.join(DATA_DIR, "games")
# PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
# ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

# Fixed version of add_team_trends function
def add_team_trends(df, team_id_col, team_score_col, opp_score_col, win_indicator, prefix):
    result_df = df.copy()
    
    # Get unique teams
    teams = df[team_id_col].unique()
    
    for team_id in teams:
        # Get games for this team
        team_mask = df[team_id_col] == team_id
        team_games = df[team_mask].copy()
        
        if len(team_games) > 0:
            # Handle both column names and functions for win_indicator
            if isinstance(win_indicator, str):
                # If win_indicator is a column name
                win_series = team_games[win_indicator]
            else:
                # If win_indicator is a function
                win_series = win_indicator(team_games)
                
            # Calculate rolling stats (last 10 games)
            team_games[f'{prefix}_last10_wins'] = win_series.rolling(10, min_periods=1).sum()
            team_games[f'{prefix}_last10_runs_scored'] = team_games[team_score_col].rolling(10, min_periods=1).mean()
            team_games[f'{prefix}_last10_runs_allowed'] = team_games[opp_score_col].rolling(10, min_periods=1).mean()
            
            # Calculate win percentage
            team_games[f'{prefix}_last10_win_pct'] = team_games[f'{prefix}_last10_wins'] / \
                                                 win_series.rolling(10, min_periods=1).count()
            
            # Update the result dataframe
            for col in [f'{prefix}_last10_wins', f'{prefix}_last10_runs_scored', 
                        f'{prefix}_last10_runs_allowed', f'{prefix}_last10_win_pct']:
                result_df.loc[team_mask, col] = team_games[col]
    
    return result_df

def calculate_matchup_history(df):
    """Calculate historical matchup statistics between teams."""
    result_df = df.copy()
    
    # Create matchup identifier (consistently ordered team IDs)
    result_df['matchup'] = result_df.apply(
        lambda row: f"{min(row['home_team_id'], row['away_team_id'])}-{max(row['home_team_id'], row['away_team_id'])}",
        axis=1
    )
    
    # Initialize matchup columns
    result_df['home_wins_vs_away'] = 0
    result_df['away_wins_vs_home'] = 0
    result_df['total_matchups'] = 0
    result_df['home_win_pct_vs_away'] = 0.0
    result_df['away_win_pct_vs_home'] = 0.0
    
    # Process each game in chronological order
    for idx, row in result_df.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        game_date = row['game_date']
        
        # Find previous games between these teams
        prev_matchups = result_df[
            (result_df['game_date'] < game_date) &
            (
                ((result_df['home_team_id'] == home_team) & (result_df['away_team_id'] == away_team)) |
                ((result_df['home_team_id'] == away_team) & (result_df['away_team_id'] == home_team))
            )
        ]
        
        if len(prev_matchups) > 0:
            # Count home team wins
            home_wins = prev_matchups[
                ((prev_matchups['home_team_id'] == home_team) & (prev_matchups['home_team_won'] == 1)) |
                ((prev_matchups['away_team_id'] == home_team) & (prev_matchups['home_team_won'] == 0))
            ].shape[0]
            
            # Count away team wins
            away_wins = prev_matchups[
                ((prev_matchups['home_team_id'] == away_team) & (prev_matchups['home_team_won'] == 1)) |
                ((prev_matchups['away_team_id'] == away_team) & (prev_matchups['home_team_won'] == 0))
            ].shape[0]
            
            # Total previous matchups
            total_matchups = len(prev_matchups)
            
            # Update the dataframe
            result_df.at[idx, 'home_wins_vs_away'] = home_wins
            result_df.at[idx, 'away_wins_vs_home'] = away_wins
            result_df.at[idx, 'total_matchups'] = total_matchups
            
            # Calculate win percentages
            if total_matchups > 0:
                result_df.at[idx, 'home_win_pct_vs_away'] = home_wins / total_matchups
                result_df.at[idx, 'away_win_pct_vs_home'] = away_wins / total_matchups
    
    return result_df

def process_data_for_model(year):
    """Process collected data into a format suitable for model training."""
    logger.info(f"Processing data for model training for {year}")
    
    # 1. Load all team schedules
    all_games = []
    teams_file = os.path.join(TEAMS_DIR, f"teams_{year}.csv")
    
    if not os.path.exists(teams_file):
        logger.error(f"Teams file not found: {teams_file}")
        return
    
    teams_df = pd.read_csv(teams_file)
    
    for _, team in teams_df.iterrows():
        team_id = team['id']
        schedule_file = os.path.join(TEAMS_DIR, f"team_{team_id}_schedule_{year}.csv")
        
        if os.path.exists(schedule_file):
            try:
                team_games = pd.read_csv(schedule_file)
                all_games.append(team_games)
            except Exception as e:
                logger.error(f"Error reading schedule file {schedule_file}: {e}")
    
    if not all_games:
        logger.error("No game data found")
        return
    
    # Combine all games and remove duplicates
    games_df = pd.concat(all_games)
    games_df = games_df.drop_duplicates(subset=['game_id'])
    
    # Filter for regular season and completed games
    games_df = games_df[(games_df['game_type'] == 'R') & 
                       (~games_df['home_score'].isna()) & 
                       (~games_df['away_score'].isna())]
    
    logger.info(f"Found {len(games_df)} completed regular season games for {year}")
    
    # 2. Create initial features dataframe
    features_data = []
    
    for _, game in tqdm(games_df.iterrows(), total=len(games_df), desc="Processing games"):
        game_id = game['game_id']
        game_date = game['game_date']
        home_id = game['home_id']
        away_id = game['away_id']
        
        # Basic game information
        game_features = {
            'game_id': game_id,
            'game_date': game_date,
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_score': game['home_score'],
            'away_score': game['away_score'],
            'home_team_won': 1 if game['home_score'] > game['away_score'] else 0
        }
        
        # Try to add weather information
        weather_file = os.path.join(GAMES_DIR, f"game_{game_id}", "weather.json")
        if os.path.exists(weather_file):
            try:
                with open(weather_file, 'r') as f:
                    weather = json.load(f)
                    game_features['temperature'] = weather.get('temperature')
                    game_features['weather_condition'] = weather.get('condition')
                    game_features['wind'] = weather.get('wind')
            except Exception as e:
                logger.error(f"Error reading weather file {weather_file}: {e}")
        
        # Try to add starting pitcher information
        starters_file = os.path.join(GAMES_DIR, f"game_{game_id}", "starting_pitchers.json")
        if os.path.exists(starters_file):
            try:
                with open(starters_file, 'r') as f:
                    starters = json.load(f)
                    game_features['home_starter_id'] = starters.get('home_starter_id')
                    game_features['away_starter_id'] = starters.get('away_starter_id')
            except Exception as e:
                logger.error(f"Error reading starters file {starters_file}: {e}")
        
        features_data.append(game_features)
    
    # Create features dataframe
    features_df = pd.DataFrame(features_data)
    
    # 3. Add team performance trends (last 10 games)
    # Convert date to datetime for sorting
    features_df['game_date'] = pd.to_datetime(features_df['game_date'])
    features_df = features_df.sort_values('game_date')
    
    # Add home team trends
    features_df = add_team_trends(
        df=features_df,
        team_id_col='home_team_id',
        team_score_col='home_score',
        opp_score_col='away_score',
        win_indicator='home_team_won',  # String - column name
        prefix='home'
    )
    
    # Add away team trends
    features_df = add_team_trends(
        df=features_df,
        team_id_col='away_team_id',
        team_score_col='away_score',
        opp_score_col='home_score',
        win_indicator=lambda x: 1 - x['home_team_won'],  # Function
        prefix='away'
    )
    
    # 4. Add matchup history
    features_df = calculate_matchup_history(features_df)
    
    # 5. Save the processed features
    # Convert date back to string for storage
    features_df['game_date'] = features_df['game_date'].dt.strftime('%Y-%m-%d')
    
    output_file = os.path.join(PROCESSED_DIR, f"game_features_{year}.csv")
    features_df.to_csv(output_file, index=False)
    logger.info(f"Saved processed features to {output_file}")
    
    return features_df

def extract_player_stats_from_boxscores(year):
    """Extract player batting statistics from boxscore files for a specific year"""
    logger.info(f"Extracting player stats from boxscores for {year}")
    
    # Get all game directories
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    # Initialize list to store player game stats
    player_game_stats = []
    processed_count = 0
    year_games_count = 0
    
    # Process each game directory
    for game_dir in tqdm(game_dirs, desc="Processing game boxscores"):
        game_id = game_dir.replace("game_", "")
        boxscore_file = os.path.join(GAMES_DIR, game_dir, "boxscore.json")
        
        if not os.path.exists(boxscore_file):
            continue
        
        # Check if game is from the target year
        game_data_file = os.path.join(GAMES_DIR, game_dir, "game_data.json")
        game_date = None
        
        if os.path.exists(game_data_file):
            try:
                with open(game_data_file, 'r') as f:
                    game_data = json.load(f)
                    game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                    
                    # Skip if not from target year
                    if not game_date or not game_date.startswith(str(year)):
                        continue
                    
                    # Count games for the year
                    year_games_count += 1
            except Exception as e:
                logger.error(f"Error checking game date for {game_dir}: {e}")
                continue
        
        # Get starting pitchers
        pitcher_file = os.path.join(GAMES_DIR, game_dir, "starting_pitchers.json")
        home_pitcher = None
        away_pitcher = None
        
        if os.path.exists(pitcher_file):
            try:
                with open(pitcher_file, 'r') as f:
                    pitcher_data = json.load(f)
                    home_pitcher = pitcher_data.get('home_starter_id')
                    away_pitcher = pitcher_data.get('away_starter_id')
            except Exception as e:
                logger.error(f"Error reading pitcher data for {game_dir}: {e}")
        
        # Process boxscore
        try:
            with open(boxscore_file, 'r') as f:
                boxscore = json.load(f)
            
            # Increment processed count
            processed_count += 1
            
            # Process home team batters
            if 'homeBatters' in boxscore and 'home' in boxscore:
                home_team = boxscore['home'].get('team', {})
                home_team_id = home_team.get('id')
                
                for batter in boxscore['homeBatters']:
                    if isinstance(batter, dict):
                        player_id = batter.get('personId')
                        
                        if player_id:
                            # Extract batting stats and convert to integers
                            try:
                                at_bats = int(batter.get('ab', 0))
                                hits = int(batter.get('h', 0))
                                doubles = int(batter.get('d', 0))
                                triples = int(batter.get('t', 0))
                                home_runs = int(batter.get('hr', 0))
                                rbi = int(batter.get('rbi', 0))
                                walks = int(batter.get('bb', 0))
                                strikeouts = int(batter.get('so', 0))
                            except (ValueError, TypeError):
                                # Skip if conversion fails
                                continue
                            
                            player_game_stats.append({
                                'game_id': game_id,
                                'game_date': game_date,
                                'player_id': str(player_id),
                                'team_id': home_team_id,
                                'is_home': True,
                                'opposing_pitcher': away_pitcher,
                                'at_bats': at_bats,
                                'hits': hits,
                                'doubles': doubles,
                                'triples': triples,
                                'home_runs': home_runs,
                                'runs_batted_in': rbi,
                                'walks': walks,
                                'strikeouts': strikeouts
                            })
            
            # Process away team batters
            if 'awayBatters' in boxscore and 'away' in boxscore:
                away_team = boxscore['away'].get('team', {})
                away_team_id = away_team.get('id')
                
                for batter in boxscore['awayBatters']:
                    if isinstance(batter, dict):
                        player_id = batter.get('personId')
                        
                        if player_id:
                            # Extract batting stats and convert to integers
                            try:
                                at_bats = int(batter.get('ab', 0))
                                hits = int(batter.get('h', 0))
                                doubles = int(batter.get('d', 0))
                                triples = int(batter.get('t', 0))
                                home_runs = int(batter.get('hr', 0))
                                rbi = int(batter.get('rbi', 0))
                                walks = int(batter.get('bb', 0))
                                strikeouts = int(batter.get('so', 0))
                            except (ValueError, TypeError):
                                # Skip if conversion fails
                                continue
                            
                            player_game_stats.append({
                                'game_id': game_id,
                                'game_date': game_date,
                                'player_id': str(player_id),
                                'team_id': away_team_id,
                                'is_home': False,
                                'opposing_pitcher': home_pitcher,
                                'at_bats': at_bats,
                                'hits': hits,
                                'doubles': doubles,
                                'triples': triples,
                                'home_runs': home_runs,
                                'runs_batted_in': rbi,
                                'walks': walks,
                                'strikeouts': strikeouts
                            })
                            
        except Exception as e:
            logger.error(f"Error processing boxscore for {game_dir}: {e}")
    
    logger.info(f"Found {year_games_count} games from {year}")
    logger.info(f"Processed {processed_count} games with boxscore data")
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(player_game_stats)
    
    if len(stats_df) > 0:
        # Add indicator fields for hits and home runs
        stats_df['got_hit'] = (stats_df['hits'] > 0).astype(int)
        stats_df['got_home_run'] = (stats_df['home_runs'] > 0).astype(int)
        
        # Save to CSV
        output_file = os.path.join(ANALYSIS_DIR, f"player_game_stats_{year}.csv")
        stats_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(stats_df)} player-game records to {output_file}")
        return stats_df
    else:
        logger.info(f"No player game stats found for {year}")
        return None

def main():
    """Main function to process data for 2024 and 2025 seasons"""
    years_to_process = [2024, 2025]
    
    for year in years_to_process:
        logger.info(f"==== Processing data for {year} season ====")
        
        # Process data for model training
        processed_df = process_data_for_model(year)
        
        if processed_df is not None:
            logger.info(f"Successfully processed features for {year}")
        
        # Extract player statistics from boxscores
        player_stats = extract_player_stats_from_boxscores(year)
        
        if player_stats is not None:
            logger.info(f"Successfully extracted player stats for {year}")
        
        logger.info(f"==== Completed processing for {year} season ====")
    
    logger.info("All data processing complete!")

if __name__ == "__main__":
    main()
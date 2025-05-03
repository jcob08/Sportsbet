# mlb_data_processor.py
import pandas as pd
import os
import glob
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = "sports_data/mlb"
PROCESSED_DIR = "sports_data/mlb/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_team_schedules(year):
    """Load all team schedules for a given year"""
    schedule_files = glob.glob(os.path.join(DATA_DIR, f"team_*_schedule_{year}.csv"))
    
    if not schedule_files:
        logger.warning(f"No schedule files found for {year}")
        return None
    
    all_games = []
    
    for file in schedule_files:
        try:
            df = pd.read_csv(file)
            all_games.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        # Remove duplicates (each game appears twice, once for each team)
        combined_df = combined_df.drop_duplicates(subset=['game_id'])
        return combined_df
    else:
        return None

def load_game_boxscores(game_ids):
    """Load boxscore data for specific games"""
    boxscores = {}
    
    for game_id in game_ids:
        file_path = os.path.join(DATA_DIR, f"game_{game_id}_boxscore.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    boxscore = json.load(f)
                boxscores[game_id] = boxscore
            except Exception as e:
                logger.error(f"Error loading boxscore for game {game_id}: {e}")
    
    return boxscores

def extract_team_stats(year):
    """
    Extract team statistics from schedule and game data
    """
    logger.info(f"Extracting team stats for {year}")
    
    # Load schedule data
    schedule_df = load_team_schedules(year)
    
    if schedule_df is None:
        logger.error("No schedule data available")
        return None
    
    # Get team IDs
    teams_file = os.path.join(DATA_DIR, f"teams_{year}.csv")
    if os.path.exists(teams_file):
        teams_df = pd.read_csv(teams_file)
        team_ids = teams_df['id'].tolist()
    else:
        # Extract unique team IDs from the schedule
        home_teams = schedule_df['home_id'].unique()
        away_teams = schedule_df['away_id'].unique()
        team_ids = list(set(home_teams) | set(away_teams))
    
    # Create team stats dictionary
    team_stats = {}
    
    for team_id in team_ids:
        # Find home games
        home_games = schedule_df[schedule_df['home_id'] == team_id]
        # Find away games
        away_games = schedule_df[schedule_df['away_id'] == team_id]
        
        # Calculate win-loss record
        home_wins = home_games[home_games['home_score'] > home_games['away_score']].shape[0]
        home_losses = home_games[home_games['home_score'] < home_games['away_score']].shape[0]
        away_wins = away_games[away_games['away_score'] > away_games['home_score']].shape[0]
        away_losses = away_games[away_games['away_score'] < away_games['home_score']].shape[0]
        
        total_wins = home_wins + away_wins
        total_losses = home_losses + away_losses
        total_games = total_wins + total_losses
        
        if total_games > 0:
            win_pct = total_wins / total_games
        else:
            win_pct = 0.0
        
        # Calculate home/away performance
        home_games_played = home_wins + home_losses
        if home_games_played > 0:
            home_win_pct = home_wins / home_games_played
        else:
            home_win_pct = 0.0
            
        away_games_played = away_wins + away_losses
        if away_games_played > 0:
            away_win_pct = away_wins / away_games_played
        else:
            away_win_pct = 0.0
        
        # Add to team stats
        team_stats[team_id] = {
            'team_id': team_id,
            'total_games': total_games,
            'wins': total_wins,
            'losses': total_losses,
            'win_pct': win_pct,
            'home_games': home_games_played,
            'home_wins': home_wins,
            'home_losses': home_losses,
            'home_win_pct': home_win_pct,
            'away_games': away_games_played,
            'away_wins': away_wins,
            'away_losses': away_losses,
            'away_win_pct': away_win_pct
        }
    
    # Convert to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Save to file
    output_file = os.path.join(PROCESSED_DIR, f"team_stats_{year}.csv")
    team_stats_df.to_csv(output_file, index=False)
    logger.info(f"Saved team stats to {output_file}")
    
    return team_stats_df

def prepare_game_features(year):
    """
    Prepare features for each game in the given year
    """
    logger.info(f"Preparing game features for {year}")
    
    # Load schedule data
    schedule_df = load_team_schedules(year)
    
    if schedule_df is None:
        logger.error("No schedule data available")
        return None
    
    # Filter for completed games
    completed_games = schedule_df[~schedule_df['winning_team'].isna()].copy()
    
    if completed_games.empty:
        logger.warning("No completed games found")
        return None
    
    # Load team stats
    team_stats_file = os.path.join(PROCESSED_DIR, f"team_stats_{year}.csv")
    if not os.path.exists(team_stats_file):
        logger.info("Team stats file not found, generating now")
        team_stats_df = extract_team_stats(year)
    else:
        team_stats_df = pd.read_csv(team_stats_file)
    
    if team_stats_df is None:
        logger.error("Unable to load or generate team stats")
        return None
    
    # Prepare features DataFrame
    features = []
    
    for _, game in completed_games.iterrows():
        game_id = game['game_id']
        home_id = game['home_id']
        away_id = game['away_id']
        
        # Get team statistics
        home_stats = team_stats_df[team_stats_df['team_id'] == home_id].iloc[0].to_dict() if not team_stats_df[team_stats_df['team_id'] == home_id].empty else {}
        away_stats = team_stats_df[team_stats_df['team_id'] == away_id].iloc[0].to_dict() if not team_stats_df[team_stats_df['team_id'] == away_id].empty else {}
        
        if not home_stats or not away_stats:
            logger.warning(f"Missing team stats for game {game_id}")
            continue
        
        # Create feature dictionary
        game_features = {
            'game_id': game_id,
            'game_date': game['game_date'],
            'home_team_id': home_id,
            'away_team_id': away_id,
            
            # Home team features
            'home_win_pct': home_stats.get('win_pct', 0.0),
            'home_home_win_pct': home_stats.get('home_win_pct', 0.0),
            
            # Away team features
            'away_win_pct': away_stats.get('win_pct', 0.0),
            'away_away_win_pct': away_stats.get('away_win_pct', 0.0),
            
            # Matchup features
            'win_pct_diff': home_stats.get('win_pct', 0.0) - away_stats.get('win_pct', 0.0),
            
            # Outcome (target variable)
            'home_team_won': 1 if game['winning_team'] == game['home_name'] else 0
        }
        
        features.append(game_features)
    
    features_df = pd.DataFrame(features)
    
    # Save to file
    output_file = os.path.join(PROCESSED_DIR, f"game_features_{year}.csv")
    features_df.to_csv(output_file, index=False)
    logger.info(f"Saved game features to {output_file}")
    
    return features_df

def main():
    """Main execution function"""
    current_year = datetime.now().year
    # Use previous year if we're in the offseason (before April)
    if datetime.now().month < 4:
        year = current_year - 1
    else:
        year = current_year
    
    logger.info(f"Starting MLB data processing for {year}")
    
    # Extract team statistics
    team_stats_df = extract_team_stats(year)
    
    # Prepare game features
    game_features_df = prepare_game_features(year)
    
    logger.info("MLB data processing completed")

if __name__ == "__main__":
    main()
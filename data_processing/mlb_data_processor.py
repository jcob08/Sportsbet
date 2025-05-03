import pandas as pd
import os
import glob
import json
import logging
from datetime import datetime

# Set up logging to show messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories for data
DATA_DIR = "sports_data/mlb"
PROCESSED_DIR = "sports_data/mlb/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_team_schedules(year):
    """Load all team schedules for a given year, filtering for regular season games and completed games."""
    schedule_files = glob.glob(os.path.join(DATA_DIR, f"team_*_schedule_{year}.csv"))
    if not schedule_files:
        logger.warning(f"No schedule files found for {year}")
        return None
    all_games = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    for file in schedule_files:
        try:
            df = pd.read_csv(file)
            # Filter for regular season games only
            df = df[df['game_type'] == 'R']
            # Ensure team IDs and scores are valid
            df = df[df['home_id'].notna() & df['away_id'].notna() & df['home_score'].notna() & df['away_score'].notna()]
            # Filter for games on or before current date
            df = df[df['game_date'] <= current_date]
            all_games.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['game_id'])
        logger.info(f"Loaded {len(combined_df)} unique regular season games for {year}")
        return combined_df
    return None

def load_game_boxscores(game_ids):
    """Load boxscore data for specific games."""
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
    """Extract team statistics from schedule and game data."""
    logger.info(f"Extracting team stats for {year}")
    schedule_df = load_team_schedules(year)
    if schedule_df is None:
        logger.error("No schedule data available")
        return None
    teams_file = os.path.join(DATA_DIR, f"teams_{year}.csv")
    if os.path.exists(teams_file):
        teams_df = pd.read_csv(teams_file)
        team_ids = teams_df['id'].tolist()
    else:
        home_teams = schedule_df['home_id'].unique()
        away_teams = schedule_df['away_id'].unique()
        team_ids = list(set(home_teams) | set(away_teams))
    team_stats = {}
    for team_id in team_ids:
        # Filter games where team is home or away
        home_games = schedule_df[schedule_df['home_id'] == team_id]
        away_games = schedule_df[schedule_df['away_id'] == team_id]
        # Calculate wins and losses
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
            logger.warning(f"Team ID {team_id} has no games played in {year}")
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
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    output_file = os.path.join(PROCESSED_DIR, f"team_stats_{year}.csv")
    team_stats_df.to_csv(output_file, index=False)
    logger.info(f"Saved team stats to {output_file}")
    return team_stats_df

def prepare_game_features(year):
    """Prepare features for each game in the given year."""
    logger.info(f"Preparing game features for {year}")
    schedule_df = load_team_schedules(year)
    if schedule_df is None:
        logger.error("No schedule data available")
        return None
    completed_games = schedule_df[~schedule_df['winning_team'].isna()].copy()
    if completed_games.empty:
        logger.warning("No completed games found")
        return None
    team_stats_file = os.path.join(PROCESSED_DIR, f"team_stats_{year}.csv")
    if not os.path.exists(team_stats_file):
        logger.info("Team stats file not found, generating now")
        team_stats_df = extract_team_stats(year)
    else:
        team_stats_df = pd.read_csv(team_stats_file)
    if team_stats_df is None:
        logger.error("Unable to load or generate team stats")
        return None
    features = []
    for _, game in completed_games.iterrows():
        game_id = game['game_id']
        home_id = game['home_id']
        away_id = game['away_id']
        home_stats = team_stats_df[team_stats_df['team_id'] == home_id].iloc[0].to_dict() if not team_stats_df[team_stats_df['team_id'] == home_id].empty else {'win_pct': 0.0, 'home_win_pct': 0.0}
        away_stats = team_stats_df[team_stats_df['team_id'] == away_id].iloc[0].to_dict() if not team_stats_df[team_stats_df['team_id'] == away_id].empty else {'win_pct': 0.0, 'away_win_pct': 0.0}
        if not home_stats or not away_stats:
            logger.info(f"Using default stats for game {game_id} (Home ID: {home_id}, Away ID: {away_id})")
        game_features = {
            'game_id': game_id,
            'game_date': game['game_date'],
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_win_pct': home_stats.get('win_pct', 0.0),
            'home_home_win_pct': home_stats.get('home_win_pct', 0.0),
            'away_win_pct': away_stats.get('win_pct', 0.0),
            'away_away_win_pct': away_stats.get('away_win_pct', 0.0),
            'win_pct_diff': home_stats.get('win_pct', 0.0) - away_stats.get('win_pct', 0.0),
            'home_team_won': 1 if game['winning_team'] == game['home_name'] else 0
        }
        features.append(game_features)
    features_df = pd.DataFrame(features)
    output_file = os.path.join(PROCESSED_DIR, f"game_features_{year}.csv")
    features_df.to_csv(output_file, index=False)
    logger.info(f"Saved game features to {output_file}")
    return features_df

def main():
    """Main execution function"""
    years = [2024, 2025]
    for year in years:
        logger.info(f"Starting MLB data processing for {year}")
        team_stats_df = extract_team_stats(year)
        game_features_df = prepare_game_features(year)
        logger.info(f"MLB data processing completed for {year}")

if __name__ == "__main__":
    main()
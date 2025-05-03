import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "sports_data/mlb"

def load_team_schedules(year):
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
        combined_df = combined_df.drop_duplicates(subset=['game_id'])
        return combined_df
    return None

def check_games(year, game_ids):
    schedule_df = load_team_schedules(year)
    if schedule_df is None:
        logger.error("No schedule data available")
        return
    for game_id in game_ids:
        game = schedule_df[schedule_df['game_id'] == game_id]
        if not game.empty:
            home_id = game['home_id'].iloc[0]
            away_id = game['away_id'].iloc[0]
            home_name = game['home_name'].iloc[0] if 'home_name' in game else 'Unknown'
            away_name = game['away_name'].iloc[0] if 'away_name' in game else 'Unknown'
            game_date = game['game_date'].iloc[0] if 'game_date' in game else 'Unknown'
            game_type = game['game_type'].iloc[0] if 'game_type' in game else 'Unknown'
            logger.info(f"Game {game_id} (Date: {game_date}, Type: {game_type}):")
            logger.info(f"  Home Team: {home_name} (ID: {home_id})")
            logger.info(f"  Away Team: {away_name} (ID: {away_id})")
        else:
            logger.warning(f"Game {game_id} not found in schedule data")

if __name__ == "__main__":
    year = 2025
    problematic_games = [778780, 794238, 794289, 790404, 790401, 796341, 787927, 787928, 790403, 790402, 789325, 791797]
    check_games(year, problematic_games)
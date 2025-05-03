import os
import logging
import pandas as pd
import time
from tqdm import tqdm
import statsapi
import json

# Set up directories
DATA_DIR = "sports_data/mlb"
PLAYERS_DIR = os.path.join(DATA_DIR, "players")
os.makedirs(PLAYERS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_player_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def api_call_with_retry(func, *args, retries=3, delay=2, **kwargs):
    """Execute an API call with retry logic"""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed after {retries} attempts")
                raise

def get_player_ids_from_boxscores(year):
    """Extract player IDs directly from boxscore files instead of roster files"""
    logger.info(f"Extracting player IDs from boxscores for {year}")
    
    # Get game directories
    game_dirs = [d for d in os.listdir(os.path.join(DATA_DIR, "games")) 
                if os.path.isdir(os.path.join(DATA_DIR, "games", d))]
    
    player_ids = set()
    processed_count = 0
    
    for game_dir in tqdm(game_dirs, desc=f"Scanning boxscores for {year}"):
        boxscore_file = os.path.join(DATA_DIR, "games", game_dir, "boxscore.json")
        
        # Check if the game is from the target year
        game_data_file = os.path.join(DATA_DIR, "games", game_dir, "game_data.json")
        if os.path.exists(game_data_file):
            try:
                with open(game_data_file, 'r') as f:
                    game_data = json.load(f)
                    game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                    
                    # Skip if not from the target year
                    if not game_date or not game_date.startswith(str(year)):
                        continue
            except Exception as e:
                logger.error(f"Error checking game date for {game_dir}: {e}")
                continue
        
        if not os.path.exists(boxscore_file):
            continue
        
        processed_count += 1
        
        try:
            with open(boxscore_file, 'r') as f:
                boxscore = json.load(f)
            
            # Extract player IDs from home team
            if 'home' in boxscore:
                # Extract batters
                if 'batters' in boxscore['home']:
                    for batter_id in boxscore['home']['batters']:
                        player_ids.add(str(batter_id))
                
                # Extract pitchers
                if 'pitchers' in boxscore['home']:
                    for pitcher_id in boxscore['home']['pitchers']:
                        player_ids.add(str(pitcher_id))
                
                # Extract from batting stats
                if 'battingStats' in boxscore['home']:
                    for player_id in boxscore['home']['battingStats'].keys():
                        player_ids.add(str(player_id))
            
            # Extract player IDs from away team
            if 'away' in boxscore:
                # Extract batters
                if 'batters' in boxscore['away']:
                    for batter_id in boxscore['away']['batters']:
                        player_ids.add(str(batter_id))
                
                # Extract pitchers
                if 'pitchers' in boxscore['away']:
                    for pitcher_id in boxscore['away']['pitchers']:
                        player_ids.add(str(pitcher_id))
                
                # Extract from batting stats
                if 'battingStats' in boxscore['away']:
                    for player_id in boxscore['away']['battingStats'].keys():
                        player_ids.add(str(player_id))
        except Exception as e:
            logger.error(f"Error processing boxscore for {game_dir}: {e}")
    
    logger.info(f"Processed {processed_count} games for {year}")
    logger.info(f"Found {len(player_ids)} unique player IDs from boxscores for {year}")
    
    # Save to CSV for reference
    output_file = os.path.join(PLAYERS_DIR, f"player_ids_{year}_from_boxscores.csv")
    pd.DataFrame({"player_id": list(player_ids)}).to_csv(output_file, index=False)
    
    return list(player_ids)

def collect_player_basic_info(player_id, year):
    """Collect basic information for a player"""
    logger.info(f"Collecting basic info for player {player_id} in {year}")
    
    # Skip if already collected
    player_file = os.path.join(PLAYERS_DIR, f"player_{player_id}_{year}.json")
    if os.path.exists(player_file):
        logger.info(f"Player {player_id} already collected for {year}. Skipping.")
        return True
    
    try:
        # Get player info using the correct endpoint
        player_info = api_call_with_retry(statsapi.get, 'person', {'personId': player_id})
        
        # Get player stats using a different approach
        season_stats = None
        try:
            # Use the correct endpoint for player stats by season
            stats_params = {
                'personId': player_id,
                'hydrate': f'stats(type=season,season={year})'
            }
            stats_response = api_call_with_retry(statsapi.get, 'person', stats_params)
            
            # Extract stats from the response if available
            if 'stats' in stats_response:
                season_stats = stats_response['stats']
        except Exception as e:
            logger.warning(f"Could not get season stats for player {player_id}: {e}")
        
        # Store player data
        player_data = {
            'player_id': player_id,
            'year': year,
            'info': player_info,
            'season_stats': season_stats
        }
        
        # Save to file
        with open(player_file, 'w') as f:
            json.dump(player_data, f, indent=4)
        
        logger.info(f"Saved basic info for player {player_id} to {player_file}")
        return True
    except Exception as e:
        logger.error(f"Error collecting basic info for player {player_id}: {e}")
        return False

def extract_player_game_stats(years):
    """Extract game-by-game stats for players from boxscores for multiple seasons"""
    for year in years:
        logger.info(f"Extracting game-by-game stats for {year}")
        
        # Get all game directories for this year
        game_dirs = []
        all_dirs = [d for d in os.listdir(os.path.join(DATA_DIR, "games")) 
                    if os.path.isdir(os.path.join(DATA_DIR, "games", d))]
        
        # Filter game directories by year
        for game_dir in all_dirs:
            game_data_file = os.path.join(DATA_DIR, "games", game_dir, "game_data.json")
            if os.path.exists(game_data_file):
                try:
                    with open(game_data_file, 'r') as f:
                        game_data = json.load(f)
                        game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                        
                        if game_date and game_date.startswith(str(year)):
                            game_dirs.append(game_dir)
                except Exception as e:
                    logger.error(f"Error checking game year for {game_dir}: {e}")
        
        logger.info(f"Found {len(game_dirs)} games for {year}")
        
        # Initialize data collection
        player_game_stats = []
        
        for game_dir in tqdm(game_dirs, desc=f"Processing {year} game boxscores"):
            game_id = game_dir.replace('game_', '')
            boxscore_file = os.path.join(DATA_DIR, "games", game_dir, "boxscore.json")
            
            if not os.path.exists(boxscore_file):
                continue
            
            try:
                with open(boxscore_file, 'r') as f:
                    boxscore = json.load(f)
                
                # Extract game date
                game_data_file = os.path.join(DATA_DIR, "games", game_dir, "game_data.json")
                game_date = None
                
                if os.path.exists(game_data_file):
                    with open(game_data_file, 'r') as f:
                        game_data = json.load(f)
                        game_date = game_data.get('gameData', {}).get('datetime', {}).get('officialDate')
                
                # Process home team batting stats
                if 'home' in boxscore and 'battingStats' in boxscore['home']:
                    home_team_id = None
                    # Try different paths to get team ID
                    if 'team' in boxscore['home'] and isinstance(boxscore['home']['team'], dict):
                        home_team_id = boxscore['home']['team'].get('id')
                    elif 'id' in boxscore['home']:
                        home_team_id = boxscore['home'].get('id')
                    
                    if home_team_id is not None:
                        home_batting = boxscore['home']['battingStats']
                        
                        for player_id, stats in home_batting.items():
                            if isinstance(stats, dict):
                                player_game_stats.append({
                                    'game_id': game_id,
                                    'game_date': game_date,
                                    'player_id': player_id,
                                    'team_id': home_team_id,
                                    'is_home': True,
                                    'year': year,
                                    'at_bats': stats.get('atBats', 0),
                                    'hits': stats.get('hits', 0),
                                    'doubles': stats.get('doubles', 0),
                                    'triples': stats.get('triples', 0),
                                    'home_runs': stats.get('homeRuns', 0),
                                    'runs_batted_in': stats.get('rbi', 0),
                                    'walks': stats.get('baseOnBalls', 0),
                                    'strikeouts': stats.get('strikeOuts', 0)
                                })
                
                # Process away team batting stats
                if 'away' in boxscore and 'battingStats' in boxscore['away']:
                    away_team_id = None
                    # Try different paths to get team ID
                    if 'team' in boxscore['away'] and isinstance(boxscore['away']['team'], dict):
                        away_team_id = boxscore['away']['team'].get('id')
                    elif 'id' in boxscore['away']:
                        away_team_id = boxscore['away'].get('id')
                    
                    if away_team_id is not None:
                        away_batting = boxscore['away']['battingStats']
                        
                        for player_id, stats in away_batting.items():
                            if isinstance(stats, dict):
                                player_game_stats.append({
                                    'game_id': game_id,
                                    'game_date': game_date,
                                    'player_id': player_id,
                                    'team_id': away_team_id,
                                    'is_home': False,
                                    'year': year,
                                    'at_bats': stats.get('atBats', 0),
                                    'hits': stats.get('hits', 0),
                                    'doubles': stats.get('doubles', 0),
                                    'triples': stats.get('triples', 0),
                                    'home_runs': stats.get('homeRuns', 0),
                                    'runs_batted_in': stats.get('rbi', 0),
                                    'walks': stats.get('baseOnBalls', 0),
                                    'strikeouts': stats.get('strikeOuts', 0)
                                })
                    
            except Exception as e:
                logger.error(f"Error processing boxscore for game {game_id}: {e}")
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(player_game_stats)
        
        if len(stats_df) > 0:
            # Add indicator fields for hits and home runs
            stats_df['got_hit'] = (stats_df['hits'] > 0).astype(int)
            stats_df['got_home_run'] = (stats_df['home_runs'] > 0).astype(int)
            
            # Ensure processed directory exists
            processed_dir = os.path.join(DATA_DIR, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Save to file
            output_file = os.path.join(processed_dir, f"player_game_stats_{year}.csv")
            stats_df.to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(stats_df)} player-game records for {year} to {output_file}")
        else:
            logger.warning(f"No player game stats found for {year}")

def main():
    """Main execution function for player data collection"""
    # Process both 2023 and 2024 seasons
    years = [2023, 2024]
    
    print(f"=== MLB Player Data Collection for {', '.join(map(str, years))} Seasons ===")
    
    # Step 1: Extract player IDs from boxscores for both seasons
    player_ids_by_year = {}
    for year in years:
        player_ids = get_player_ids_from_boxscores(year)
        player_ids_by_year[year] = player_ids
    
    # Save combined player IDs
    all_player_ids = set()
    for player_ids in player_ids_by_year.values():
        all_player_ids.update(player_ids)
    
    output_file = os.path.join(PLAYERS_DIR, "all_player_ids.csv")
    pd.DataFrame({"player_id": list(all_player_ids)}).to_csv(output_file, index=False)
    logger.info(f"Saved {len(all_player_ids)} unique player IDs across all years to {output_file}")
    
    # Step 2: Collect basic player information for each season
    for year in years:
        player_ids = player_ids_by_year.get(year, [])
        if player_ids:
            print(f"\nProcessing {len(player_ids)} players for {year} season")
            
            for player_id in tqdm(player_ids, desc=f"Collecting player data for {year}"):
                try:
                    collect_player_basic_info(player_id, year)
                    time.sleep(1)  # Delay to avoid overwhelming the API
                except KeyboardInterrupt:
                    print("\nOperation interrupted by user. Continuing to next step...")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing player {player_id}: {e}")
                    continue
    
    # Step 3: Extract player game-by-game statistics from boxscores
    try:
        extract_player_game_stats(years)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Exiting...")
        return
    
    print("\n=== Player Data Collection Complete ===")
    print("Player IDs extracted and saved to all_player_ids.csv")
    print("Player basic information saved to individual JSON files")
    print("Player game statistics extracted and saved to CSV files")

if __name__ == "__main__":
    main()
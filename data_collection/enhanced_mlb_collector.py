import statsapi
import pandas as pd
import os
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm  # For progress bars

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_2024_2025_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directory for data
DATA_DIR = "sports_data/mlb"
os.makedirs(DATA_DIR, exist_ok=True)

# Create subdirectories for better organization
TEAMS_DIR = os.path.join(DATA_DIR, "teams")
GAMES_DIR = os.path.join(DATA_DIR, "games")
PITCHERS_DIR = os.path.join(DATA_DIR, "pitchers")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")  # Added this line

for directory in [TEAMS_DIR, GAMES_DIR, PITCHERS_DIR, PROCESSED_DIR, ANALYSIS_DIR]:  # Updated this line
    os.makedirs(directory, exist_ok=True)

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

def collect_league_data(year):
    """Collect list of all MLB teams for a given year."""
    logger.info(f"Collecting league data for {year}")
    try:
        teams = api_call_with_retry(statsapi.get, 'teams', {'sportId': 1, 'season': year})['teams']
        teams_data = []
        for team in teams:
            teams_data.append({
                'id': team['id'],
                'name': team['name'],
                'teamCode': team.get('teamCode', ''),
                'abbreviation': team.get('abbreviation', ''),
                'division': team.get('division', {}).get('name', ''),
                'league': team.get('league', {}).get('name', ''),
                'venue': team.get('venue', {}).get('name', ''),
                'year': year
            })
        teams_df = pd.DataFrame(teams_data)
        output_file = os.path.join(TEAMS_DIR, f"teams_{year}.csv")
        teams_df.to_csv(output_file, index=False)
        logger.info(f"Saved teams data to {output_file}")
        return teams_df
    except Exception as e:
        logger.error(f"Error collecting league data for {year}: {e}")
        return None

def collect_team_data(team_id, year):
    """Collect schedule and roster data for a team."""
    logger.info(f"Collecting schedule for team {team_id} in {year}")
    try:
        # Get team schedule with retry
        schedule = api_call_with_retry(statsapi.schedule, team=team_id, season=year)
        schedule_data = []
        for game in schedule:
            schedule_data.append({
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'game_type': game['game_type'],
                'home_id': game['home_id'],
                'home_name': game['home_name'],
                'away_id': game['away_id'],
                'away_name': game['away_name'],
                'home_score': game.get('home_score', None),
                'away_score': game.get('away_score', None),
                'winning_team': game.get('winning_team', None),
                'losing_team': game.get('losing_team', None),
                'status': game.get('status', ''),
                'doubleheader': game.get('doubleheader', 'N'),
                'venue': game.get('venue_name', ''),
                'year': year
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        
        # Convert date string to datetime for sorting
        if not schedule_df.empty:
            schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'])
            schedule_df = schedule_df.sort_values('game_date')
            # Convert back to string for storage
            schedule_df['game_date'] = schedule_df['game_date'].dt.strftime('%Y-%m-%d')
        
        output_file = os.path.join(TEAMS_DIR, f"team_{team_id}_schedule_{year}.csv")
        schedule_df.to_csv(output_file, index=False)
        logger.info(f"Saved schedule to {output_file}")

        # Get team roster
        logger.info(f"Collecting roster for team {team_id} in {year}")
        roster = api_call_with_retry(statsapi.roster, team_id, season=year)
        roster_file = os.path.join(TEAMS_DIR, f"team_{team_id}_roster_{year}.txt")
        with open(roster_file, 'w') as f:
            f.write(roster)
        logger.info(f"Saved roster to {roster_file}")
        
        # Try to get team stats via general team endpoint
        logger.info(f"Collecting team info for team {team_id} in {year}")
        try:
            # Use the team endpoint to get available team info
            team_info = api_call_with_retry(statsapi.get, 'team', {'teamId': team_id, 'season': year})
            
            # Save whatever team info is available
            team_info_file = os.path.join(TEAMS_DIR, f"team_{team_id}_info_{year}.json")
            with open(team_info_file, 'w') as f:
                json.dump(team_info, f, indent=4)
            logger.info(f"Saved team info to {team_info_file}")
            
        except Exception as e:
            logger.error(f"Error collecting team info for team {team_id}: {e}")
        
        return schedule_df
    except Exception as e:
        logger.error(f"Error collecting team data for team {team_id} in {year}: {e}")
        return None

def collect_game_data(game_id, year):
    """Collect comprehensive game data including boxscore and linescore."""
    logger.info(f"Collecting game data for game {game_id}")
    
    # Skip if already collected
    game_dir = os.path.join(GAMES_DIR, f"game_{game_id}")
    boxscore_file = os.path.join(game_dir, "boxscore.json")
    if os.path.exists(boxscore_file):
        logger.info(f"Game {game_id} already collected. Skipping.")
        return
        
    try:
        # Create game directory
        os.makedirs(game_dir, exist_ok=True)
        
        # Get basic game data
        try:
            game_data = api_call_with_retry(statsapi.get, 'game', {'gamePk': game_id})
            game_file = os.path.join(game_dir, "game_data.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f, indent=4)
            
            # Extract weather information if available
            try:
                weather_data = {
                    'game_id': game_id,
                    'year': year,
                    'temperature': None,
                    'condition': None,
                    'wind': None
                }
                
                game_info = game_data.get('gameData', {})
                weather = game_info.get('weather', {})
                
                if weather:
                    weather_data['temperature'] = weather.get('temp')
                    weather_data['condition'] = weather.get('condition')
                    weather_data['wind'] = weather.get('wind')
                
                weather_file = os.path.join(game_dir, "weather.json")
                with open(weather_file, 'w') as f:
                    json.dump(weather_data, f, indent=4)
            except Exception as e:
                logger.error(f"Error extracting weather data for game {game_id}: {e}")
        except Exception as e:
            logger.error(f"Error collecting game data for game {game_id}: {e}")
        
        # Get boxscore data
        try:
            boxscore = api_call_with_retry(statsapi.boxscore_data, game_id)
            with open(boxscore_file, 'w') as f:
                json.dump(boxscore, f, indent=4)
        except Exception as e:
            logger.error(f"Error collecting boxscore for game {game_id}: {e}")
        
        # Get linescore
        try:
            linescore = api_call_with_retry(statsapi.linescore, game_id)
            linescore_file = os.path.join(game_dir, "linescore.txt")
            with open(linescore_file, 'w') as f:
                f.write(linescore)
        except Exception as e:
            logger.error(f"Error collecting linescore for game {game_id}: {e}")
        
        # Extract starting pitchers from boxscore if available
        try:
            if os.path.exists(boxscore_file):
                with open(boxscore_file, 'r') as f:
                    boxscore = json.load(f)
                
                home_team = boxscore.get('home', {})
                away_team = boxscore.get('away', {})
                
                home_pitchers = home_team.get('pitchers', [])
                away_pitchers = away_team.get('pitchers', [])
                
                home_starter_id = home_pitchers[0] if home_pitchers else None
                away_starter_id = away_pitchers[0] if away_pitchers else None
                
                starters = {
                    'game_id': game_id,
                    'year': year,
                    'home_starter_id': home_starter_id,
                    'away_starter_id': away_starter_id
                }
                
                starters_file = os.path.join(game_dir, "starting_pitchers.json")
                with open(starters_file, 'w') as f:
                    json.dump(starters, f, indent=4)
        except Exception as e:
            logger.error(f"Error extracting starting pitchers for game {game_id}: {e}")
        
        # Get play-by-play data if available
        try:
            plays = api_call_with_retry(statsapi.get, 'game_playByPlay', {'gamePk': game_id})
            plays_file = os.path.join(game_dir, "plays.json")
            with open(plays_file, 'w') as f:
                json.dump(plays, f, indent=4)
        except Exception as e:
            logger.error(f"Error collecting play-by-play for game {game_id}: {e}")
        
        logger.info(f"Successfully collected data for game {game_id}")
        return True
    except Exception as e:
        logger.error(f"Error collecting game data for game {game_id}: {e}")
        return False

def collect_pitcher_data(pitcher_id, year):
    """Collect data for an individual pitcher."""
    logger.info(f"Collecting data for pitcher {pitcher_id} for {year}")
    
    # Skip if already collected
    pitcher_file = os.path.join(PITCHERS_DIR, f"pitcher_{pitcher_id}_{year}.json")
    if os.path.exists(pitcher_file):
        logger.info(f"Pitcher {pitcher_id} already collected for {year}. Skipping.")
        return True
    
    try:
        # Get player info
        player_info = api_call_with_retry(statsapi.get, 'person', {'personId': pitcher_id})
        
        # Get season stats
        try:
            # Try to use player_stats endpoint
            player_stats = api_call_with_retry(statsapi.player_stats, 
                                             personId=pitcher_id,
                                             group="pitching", 
                                             type="season", 
                                             season=year)
        except Exception:
            # If player_stats doesn't work, try to extract from player_info
            logger.warning(f"Could not get stats using player_stats endpoint for pitcher {pitcher_id}")
            player_stats = player_info.get('stats', [])
        
        # Combine into one record
        pitcher_data = {
            'pitcher_id': pitcher_id,
            'year': year,
            'info': player_info,
            'stats': player_stats
        }
        
        with open(pitcher_file, 'w') as f:
            json.dump(pitcher_data, f, indent=4)
        
        logger.info(f"Saved pitcher data to {pitcher_file}")
        return True
    except Exception as e:
        logger.error(f"Error collecting data for pitcher {pitcher_id}: {e}")
        return False

def identify_starting_pitchers(year):
    """Identify all starting pitchers from game data."""
    logger.info(f"Identifying starting pitchers for {year}")
    
    starting_pitchers = set()
    
    # Scan all game directories
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    for game_dir in game_dirs:
        starters_file = os.path.join(GAMES_DIR, game_dir, "starting_pitchers.json")
        if os.path.exists(starters_file):
            try:
                with open(starters_file, 'r') as f:
                    starters = json.load(f)
                    
                game_year = starters.get('year')
                if game_year == year:
                    home_starter = starters.get('home_starter_id')
                    away_starter = starters.get('away_starter_id')
                    
                    if home_starter:
                        starting_pitchers.add(home_starter)
                    if away_starter:
                        starting_pitchers.add(away_starter)
            except Exception as e:
                logger.error(f"Error reading starters file {starters_file}: {e}")
    
    logger.info(f"Found {len(starting_pitchers)} unique starting pitchers for {year}")
    return list(starting_pitchers)

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
    
    # Function to calculate rolling stats for a team
    def add_team_trends(df, team_id_col, team_score_col, opp_score_col, win_indicator, prefix):
        result_df = df.copy()
        
        # Get unique teams
        teams = df[team_id_col].unique()
        
        for team_id in teams:
            # Get games for this team
            team_mask = df[team_id_col] == team_id
            team_games = df[team_mask].copy()
            
            if len(team_games) > 0:
                # Calculate rolling stats (last 10 games)
                team_games[f'{prefix}_last10_wins'] = team_games[win_indicator].rolling(10, min_periods=1).sum()
                team_games[f'{prefix}_last10_runs_scored'] = team_games[team_score_col].rolling(10, min_periods=1).mean()
                team_games[f'{prefix}_last10_runs_allowed'] = team_games[opp_score_col].rolling(10, min_periods=1).mean()
                
                # Calculate win percentage
                team_games[f'{prefix}_last10_win_pct'] = team_games[f'{prefix}_last10_wins'] / \
                                                     team_games[win_indicator].rolling(10, min_periods=1).count()
                
                # Update the result dataframe
                for col in [f'{prefix}_last10_wins', f'{prefix}_last10_runs_scored', 
                            f'{prefix}_last10_runs_allowed', f'{prefix}_last10_win_pct']:
                    result_df.loc[team_mask, col] = team_games[col]
        
        return result_df
    
    # Add home team trends
    features_df = add_team_trends(
        df=features_df,
        team_id_col='home_team_id',
        team_score_col='home_score',
        opp_score_col='away_score',
        win_indicator='home_team_won',
        prefix='home'
    )
    
    # Add away team trends
    features_df = add_team_trends(
        df=features_df,
        team_id_col='away_team_id',
        team_score_col='away_score',
        opp_score_col='home_score',
        win_indicator=lambda x: 1 - x['home_team_won'],
        prefix='away'
    )
    
    # 4. Add matchup history
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
    
    # Add matchup history
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
    print(f"Extracting player stats from boxscores for {year}")
    
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
                print(f"Error checking game date for {game_dir}: {e}")
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
                print(f"Error reading pitcher data for {game_dir}: {e}")
        
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
                                'triples': triples,'home_runs': home_runs,
                                'runs_batted_in': rbi,
                                'walks': walks,
                                'strikeouts': strikeouts
                            })
                            
        except Exception as e:
            print(f"Error processing boxscore for {game_dir}: {e}")
    
    print(f"Found {year_games_count} games from {year}")
    print(f"Processed {processed_count} games with boxscore data")
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(player_game_stats)
    
    if len(stats_df) > 0:
        # Add indicator fields for hits and home runs
        stats_df['got_hit'] = (stats_df['hits'] > 0).astype(int)
        stats_df['got_home_run'] = (stats_df['home_runs'] > 0).astype(int)
        
        # Save to CSV
        output_file = os.path.join(ANALYSIS_DIR, f"player_game_stats_{year}.csv")
        stats_df.to_csv(output_file, index=False)
        
        print(f"Saved {len(stats_df)} player-game records to {output_file}")
        return stats_df
    else:
        print(f"No player game stats found for {year}")
        return None

def main():
    """Main execution function"""
    # Define years to collect (2024 and 2025)
    years_to_collect = [2024, 2025]
    
    for year in years_to_collect:
        logger.info(f"==== Starting MLB data collection for {year} season ====")
        
        # 1. Collect league data (teams)
        teams_df = collect_league_data(year)
        
        if teams_df is None:
            logger.error(f"Failed to collect league data for {year}. Skipping year.")
            continue
        
        # 2. Collect team schedules and rosters
        all_game_ids = set()
        
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_name = team['name']
            
            logger.info(f"Processing {team_name} (ID: {team_id}) for {year}")
            schedule_df = collect_team_data(team_id, year)
            
            if schedule_df is not None:
                # For 2025, only collect games that have been played already
                if year == 2025:
                    # Filter for games with scores (completed games)
                    completed_games = schedule_df[
                        (~schedule_df['home_score'].isna()) & 
                        (~schedule_df['away_score'].isna()) &
                        (schedule_df['game_type'] == 'R')
                    ]
                    all_game_ids.update(completed_games['game_id'].tolist())
                    logger.info(f"Found {len(completed_games)} completed games for {team_name} in 2025")
                else:
                    # For 2024, collect all regular season games
                    regular_season_games = schedule_df[schedule_df['game_type'] == 'R']
                    all_game_ids.update(regular_season_games['game_id'].tolist())
            
            # Add delay between teams to avoid overwhelming the API
            time.sleep(3)
        
        logger.info(f"Found {len(all_game_ids)} games to collect for {year}")
        
        # 3. Collect detailed game data
        for game_id in tqdm(all_game_ids, desc=f"Collecting game data for {year}"):
            collect_game_data(game_id, year)
            # Brief delay between game data collection
            time.sleep(1)
        
        # 4. Identify and collect starting pitcher data
        starting_pitchers = identify_starting_pitchers(year)
        
        for pitcher_id in tqdm(starting_pitchers, desc=f"Collecting pitcher data for {year}"):
            collect_pitcher_data(pitcher_id, year)
            time.sleep(1)
        
        # 5. Process data for model training
        process_data_for_model(year)
        
        # 6. Extract player statistics from boxscores
        extract_player_stats_from_boxscores(year)
        
        logger.info(f"==== Completed MLB data collection for {year} season ====")
    
    logger.info("All data collection complete!")

if __name__ == "__main__":
    main()
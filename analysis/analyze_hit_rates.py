import os
import json
import pandas as pd
from tqdm import tqdm

# Set up directories
DATA_DIR = "sports_data/mlb"
GAMES_DIR = os.path.join(DATA_DIR, "games")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def extract_player_stats_from_boxscores(year):
    """Extract player batting statistics from boxscore files with the correct structure"""
    print(f"Extracting player stats from boxscores for {year}")
    
    # Get all game directories
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    # Initialize list to store player game stats
    player_game_stats = []
    processed_count = 0
    
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
            
            # Get all player info
            player_info = boxscore.get('playerInfo', {})
            
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
            print(f"Error processing boxscore for {game_dir}: {e}")
    
    print(f"Processed {processed_count} games from {year}")
    
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
        
        # Analyze player-level stats
        analyze_player_hitting(stats_df, year)
    else:
        print(f"No player game stats found for {year}")
    
    return stats_df

def analyze_player_hitting(stats_df, year):
    """Analyze player hitting statistics"""
    print(f"\nAnalyzing hitting statistics for {year}")
    
    # Calculate player-level stats
    player_stats = stats_df.groupby('player_id').agg({
        'game_id': 'nunique',
        'at_bats': 'sum',
        'hits': 'sum',
        'home_runs': 'sum',
        'got_hit': 'mean',
        'got_home_run': 'mean'
    }).reset_index()
    
    # Rename columns
    player_stats.rename(columns={
        'game_id': 'games',
        'got_hit': 'hit_rate',
        'got_home_run': 'hr_rate'
    }, inplace=True)
    
    # Calculate batting average
    player_stats['batting_avg'] = player_stats['hits'] / player_stats['at_bats']
    
    # Filter to players with minimum at-bats
    min_at_bats = 50
    qualified_players = player_stats[player_stats['at_bats'] >= min_at_bats].copy()
    
    print(f"Found {len(qualified_players)} players with at least {min_at_bats} at-bats")
    
    # Sort by hit rate
    top_hitters = qualified_players.sort_values('hit_rate', ascending=False)
    
    print(f"\nTop 10 Hitters by Hit Rate (min. {min_at_bats} at-bats):")
    for i, (_, player) in enumerate(top_hitters.head(10).iterrows(), 1):
        print(f"{i}. Player ID: {player['player_id']}, " +
              f"Hit Rate: {player['hit_rate']*100:.2f}%, " +
              f"Batting Avg: {player['batting_avg']:.3f}, " +
              f"Games: {player['games']}, At-Bats: {player['at_bats']}")
    
    # Sort by home run rate
    top_hr_hitters = qualified_players.sort_values('hr_rate', ascending=False)
    
    print(f"\nTop 10 Power Hitters by HR Rate (min. {min_at_bats} at-bats):")
    for i, (_, player) in enumerate(top_hr_hitters.head(10).iterrows(), 1):
        print(f"{i}. Player ID: {player['player_id']}, " +
              f"HR Rate: {player['hr_rate']*100:.2f}%, " +
              f"HRs: {player['home_runs']}, " +
              f"Games: {player['games']}, At-Bats: {player['at_bats']}")
    
    # Save top hitter lists
    top_hitters_file = os.path.join(ANALYSIS_DIR, f"top_hitters_{year}.csv")
    top_hitters.head(50).to_csv(top_hitters_file, index=False)
    
    top_hr_file = os.path.join(ANALYSIS_DIR, f"top_hr_hitters_{year}.csv")
    top_hr_hitters.head(50).to_csv(top_hr_file, index=False)
    
    # Analyze home vs. away performance
    home_away_analysis(stats_df, year)
    
    return qualified_players

def home_away_analysis(stats_df, year):
    """Analyze home vs. away performance differences"""
    print(f"\nAnalyzing home vs. away performance for {year}")
    
    # Group by player and home/away
    home_away_stats = stats_df.groupby(['player_id', 'is_home']).agg({
        'game_id': 'nunique',
        'at_bats': 'sum',
        'hits': 'sum',
        'home_runs': 'sum',
        'got_hit': 'mean',
        'got_home_run': 'mean'
    }).reset_index()
    
    # Filter to players with minimum games both home and away
    min_games = 10
    
    # Get players with enough home games
    home_players = home_away_stats[
        (home_away_stats['is_home'] == True) & 
        (home_away_stats['game_id'] >= min_games)
    ]['player_id'].tolist()
    
    # Get players with enough away games
    away_players = home_away_stats[
        (home_away_stats['is_home'] == False) & 
        (home_away_stats['game_id'] >= min_games)
    ]['player_id'].tolist()
    
    # Get players with enough games both home and away
    qualified_players = list(set(home_players) & set(away_players))
    
    if not qualified_players:
        print(f"No players found with at least {min_games} games both home and away")
        return
    
    # Filter to qualified players
    qualified_stats = home_away_stats[home_away_stats['player_id'].isin(qualified_players)].copy()
    
    # Reshape data to have home and away stats in separate columns
    home_stats = qualified_stats[qualified_stats['is_home'] == True].copy()
    away_stats = qualified_stats[qualified_stats['is_home'] == False].copy()
    
    # Rename columns for clarity
    home_stats.rename(columns={
        'game_id': 'home_games',
        'at_bats': 'home_at_bats',
        'hits': 'home_hits',
        'home_runs': 'home_hrs',
        'got_hit': 'home_hit_rate',
        'got_home_run': 'home_hr_rate'
    }, inplace=True)
    
    away_stats.rename(columns={
        'game_id': 'away_games',
        'at_bats': 'away_at_bats',
        'hits': 'away_hits',
        'home_runs': 'away_hrs',
        'got_hit': 'away_hit_rate',
        'got_home_run': 'away_hr_rate'
    }, inplace=True)
    
    # Drop is_home column
    home_stats.drop('is_home', axis=1, inplace=True)
    away_stats.drop('is_home', axis=1, inplace=True)
    
    # Merge home and away stats
    player_home_away = pd.merge(home_stats, away_stats, on='player_id', how='inner')
    
    # Calculate home/away differences
    player_home_away['hit_rate_diff'] = player_home_away['home_hit_rate'] - player_home_away['away_hit_rate']
    player_home_away['hr_rate_diff'] = player_home_away['home_hr_rate'] - player_home_away['away_hr_rate']
    
    # Sort by hit rate difference
    home_advantage_hitters = player_home_away.sort_values('hit_rate_diff', ascending=False)
    
    print(f"\nPlayers with Biggest Home Advantage for Hits (min. {min_games} games each):")
    for i, (_, player) in enumerate(home_advantage_hitters.head(10).iterrows(), 1):
        print(f"{i}. Player ID: {player['player_id']}, " +
              f"Home Hit Rate: {player['home_hit_rate']*100:.2f}%, " +
              f"Away Hit Rate: {player['away_hit_rate']*100:.2f}%, " +
              f"Difference: {player['hit_rate_diff']*100:.2f}%")
    
    # Sort by HR rate difference
    home_advantage_hr = player_home_away.sort_values('hr_rate_diff', ascending=False)
    
    print(f"\nPlayers with Biggest Home Advantage for HRs (min. {min_games} games each):")
    for i, (_, player) in enumerate(home_advantage_hr.head(10).iterrows(), 1):
        print(f"{i}. Player ID: {player['player_id']}, " +
              f"Home HR Rate: {player['home_hr_rate']*100:.2f}%, " +
              f"Away HR Rate: {player['away_hr_rate']*100:.2f}%, " +
              f"Difference: {player['hr_rate_diff']*100:.2f}%")
    
    # Save the analysis
    home_away_file = os.path.join(ANALYSIS_DIR, f"home_away_splits_{year}.csv")
    player_home_away.to_csv(home_away_file, index=False)
    
    return player_home_away

# Run the analysis for 2023
player_data = extract_player_stats_from_boxscores(2023)

# If that doesn't work, try 2024
if player_data is None or len(player_data) == 0:
    player_data = extract_player_stats_from_boxscores(2024)
import os
import json
import pandas as pd
from tqdm import tqdm

# Set up directories
DATA_DIR = "sports_data/mlb"
GAMES_DIR = os.path.join(DATA_DIR, "games")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def extract_player_stats_for_year(year):
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
                                'triples': triples,
                                'home_runs': home_runs,
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

def analyze_basic_stats(stats_df, year):
    """Analyze basic statistics from player game stats"""
    print(f"\nBasic statistics for {year} season:")
    
    # Number of unique players
    player_count = stats_df['player_id'].nunique()
    print(f"Number of players: {player_count}")
    
    # Overall hit and home run rates
    hit_rate = stats_df['got_hit'].mean() * 100
    hr_rate = stats_df['got_home_run'].mean() * 100
    print(f"Overall hit rate: {hit_rate:.2f}%")
    print(f"Overall home run rate: {hr_rate:.2f}%")
    
    # Calculate player-level season stats
    season_stats = stats_df.groupby('player_id').agg({
        'game_id': 'nunique',
        'at_bats': 'sum',
        'hits': 'sum',
        'home_runs': 'sum',
        'got_hit': 'mean',
        'got_home_run': 'mean'
    }).reset_index()
    
    # Rename columns
    season_stats.rename(columns={
        'game_id': 'games',
        'got_hit': 'hit_rate',
        'got_home_run': 'hr_rate'
    }, inplace=True)
    
    # Calculate batting average
    season_stats['batting_avg'] = season_stats['hits'] / season_stats['at_bats']
    
    # Filter to players with minimum at-bats
    min_at_bats = 50
    qualified_players = season_stats[season_stats['at_bats'] >= min_at_bats].copy()
    
    print(f"Players with at least {min_at_bats} at-bats: {len(qualified_players)}")
    
    # Save season stats to file
    output_file = os.path.join(ANALYSIS_DIR, f"player_season_stats_{year}.csv")
    season_stats.to_csv(output_file, index=False)
    
    print(f"Saved season stats for {len(season_stats)} players to {output_file}")
    
    return qualified_players

def main():
    """Main execution function"""
    print("=== Extracting MLB Player Stats for Multiple Years ===")
    
    # List of years to extract
    years = [2023, 2024, 2025]
    
    for year in years:
        # Extract player stats for the year
        stats_df = extract_player_stats_for_year(year)
        
        if stats_df is not None and len(stats_df) > 0:
            # Analyze basic statistics
            qualified_players = analyze_basic_stats(stats_df, year)
    
    print("\n=== Player Stats Extraction Complete ===")

if __name__ == "__main__":
    main()
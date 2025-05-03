import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up paths
DATA_DIR = "sports_data/mlb"
TEAMS_DIR = os.path.join(DATA_DIR, "teams")
GAMES_DIR = os.path.join(DATA_DIR, "games")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def load_team_data():
    """Load team schedules and merge into a complete game dataset"""
    print("Loading team schedules...")
    
    # Get team information
    team_files = glob.glob(os.path.join(TEAMS_DIR, "teams_*.csv"))
    teams_df = pd.DataFrame()
    
    for file in team_files:
        year = os.path.basename(file).replace("teams_", "").replace(".csv", "")
        df = pd.read_csv(file)
        df['year'] = year
        teams_df = pd.concat([teams_df, df])
    
    # Get schedule information
    all_games = []
    schedule_files = glob.glob(os.path.join(TEAMS_DIR, "team_*_schedule_*.csv"))
    
    for file in schedule_files:
        try:
            games_df = pd.read_csv(file)
            all_games.append(games_df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_games:
        print("No game data found.")
        return None, None
    
    # Combine games and remove duplicates
    combined_games = pd.concat(all_games, ignore_index=True)
    combined_games['game_date'] = pd.to_datetime(combined_games['game_date'])
    
    # Ensure game_id is string type for consistent merging later
    combined_games['game_id'] = combined_games['game_id'].astype(str)
    
    # Drop duplicates based on game_id
    unique_games = combined_games.drop_duplicates(subset=['game_id'])
    
    # Get only completed regular season games
    completed_games = unique_games[
        (unique_games['game_type'] == 'R') & 
        (~unique_games['home_score'].isna()) & 
        (~unique_games['away_score'].isna())
    ].copy()
    
    # Add additional columns
    completed_games['home_team_won'] = (completed_games['home_score'] > completed_games['away_score']).astype(int)
    completed_games['total_runs'] = completed_games['home_score'] + completed_games['away_score']
    completed_games['run_diff'] = completed_games['home_score'] - completed_games['away_score']
    
    print(f"Loaded {len(unique_games)} unique games, {len(completed_games)} completed regular season games")
    
    return teams_df, completed_games

def analyze_home_field_advantage(games_df):
    """Analyze home field advantage"""
    print("\nAnalyzing home field advantage...")
    
    # Overall home win percentage
    home_win_pct = games_df['home_team_won'].mean() * 100
    print(f"Overall home team win percentage: {home_win_pct:.2f}%")
    
    # Home win percentage by year
    yearly_home_adv = games_df.groupby('year')['home_team_won'].mean() * 100
    print("\nHome win percentage by year:")
    print(yearly_home_adv)
    
    # Home win percentage by team
    team_home_adv = games_df.groupby('home_id')['home_team_won'].agg(['mean', 'count'])
    team_home_adv['win_pct'] = team_home_adv['mean'] * 100
    team_home_adv = team_home_adv.sort_values('win_pct', ascending=False)
    
    print("\nTop 5 teams with highest home win percentage:")
    print(team_home_adv.head(5)[['win_pct', 'count']])
    
    print("\nBottom 5 teams with lowest home win percentage:")
    print(team_home_adv.tail(5)[['win_pct', 'count']])
    
    # Plot home win percentage distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(team_home_adv['win_pct'], bins=10)
    plt.title('Distribution of Home Win Percentage by Team')
    plt.xlabel('Home Win Percentage')
    plt.ylabel('Number of Teams')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'home_win_pct_distribution.png'))
    
    return team_home_adv

def analyze_runs_scored(games_df):
    """Analyze runs scored distribution and factors"""
    print("\nAnalyzing runs scored...")
    
    # Overall runs distribution
    avg_runs = games_df['total_runs'].mean()
    median_runs = games_df['total_runs'].median()
    std_runs = games_df['total_runs'].std()
    
    print(f"Average runs per game: {avg_runs:.2f}")
    print(f"Median runs per game: {median_runs:.1f}")
    print(f"Standard deviation: {std_runs:.2f}")
    
    # Plot total runs distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(games_df['total_runs'], kde=True, bins=20)
    plt.axvline(avg_runs, color='r', linestyle='--', label=f'Mean: {avg_runs:.2f}')
    plt.axvline(median_runs, color='g', linestyle='--', label=f'Median: {median_runs:.1f}')
    plt.title('Distribution of Total Runs per Game')
    plt.xlabel('Total Runs')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'total_runs_distribution.png'))
    
    # Runs by month
    games_df['month'] = games_df['game_date'].dt.month
    monthly_runs = games_df.groupby('month')['total_runs'].mean()
    
    plt.figure(figsize=(10, 6))
    monthly_runs.plot(kind='bar')
    plt.title('Average Runs per Game by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Runs')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'monthly_runs.png'))
    
    return monthly_runs

def analyze_pitcher_impact(games_df):
    """Analyze starting pitcher impact from available data"""
    print("\nAnalyzing pitcher impact...")
    
    # Check if we have starting pitcher data
    pitcher_data_available = False
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    starting_pitchers = {}
    
    for game_dir in tqdm(game_dirs, desc="Scanning pitcher data"):
        game_id = game_dir.replace('game_', '')
        starters_file = os.path.join(GAMES_DIR, game_dir, "starting_pitchers.json")
        
        if os.path.exists(starters_file):
            pitcher_data_available = True
            try:
                with open(starters_file, 'r') as f:
                    starters = json.load(f)
                    starting_pitchers[game_id] = {
                        'home_starter_id': starters.get('home_starter_id'),
                        'away_starter_id': starters.get('away_starter_id')
                    }
            except Exception as e:
                print(f"Error reading {starters_file}: {e}")
    
    if not pitcher_data_available:
        print("No starting pitcher data available.")
        return None
    
    # Convert to DataFrame
    pitchers_df = pd.DataFrame.from_dict(starting_pitchers, orient='index')
    pitchers_df.reset_index(inplace=True)
    pitchers_df.rename(columns={'index': 'game_id'}, inplace=True)
    
    # Ensure game_id is string type in both dataframes
    pitchers_df['game_id'] = pitchers_df['game_id'].astype(str)
    
    # Merge with games data
    print(f"Game data shape: {games_df.shape}, Pitcher data shape: {pitchers_df.shape}")
    
    # Check for games with multiple pitcher entries
    pitcher_counts = pitchers_df['game_id'].value_counts()
    duplicate_games = pitcher_counts[pitcher_counts > 1]
    if len(duplicate_games) > 0:
        print(f"Found {len(duplicate_games)} games with multiple pitcher entries. Keeping first entry for each.")
        pitchers_df = pitchers_df.drop_duplicates(subset=['game_id'], keep='first')
    
    # Verify data types before merge
    print(f"Game ID type in games_df: {games_df['game_id'].dtype}")
    print(f"Game ID type in pitchers_df: {pitchers_df['game_id'].dtype}")
    
    # Merge data
    merged_df = pd.merge(games_df, pitchers_df, on='game_id', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    
    # Analyze home pitcher performance
    home_pitcher_stats = merged_df.groupby('home_starter_id').agg({
        'home_team_won': ['mean', 'count'],
        'home_score': 'mean',
        'away_score': 'mean'
    })
    
    # Flatten multi-index columns
    home_pitcher_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in home_pitcher_stats.columns]
    
    # Reset index for easier handling
    home_pitcher_stats = home_pitcher_stats.reset_index()
    
    # Rename columns for clarity
    home_pitcher_stats.rename(columns={
        'home_team_won_mean': 'win_pct',
        'home_team_won_count': 'games',
        'home_score_mean': 'runs_for',
        'away_score_mean': 'runs_against'
    }, inplace=True)
    
    # Filter pitchers with at least 5 starts
    home_pitcher_stats = home_pitcher_stats[home_pitcher_stats['games'] >= 5].copy()
    
    # Calculate run differential
    home_pitcher_stats['run_diff'] = home_pitcher_stats['runs_for'] - home_pitcher_stats['runs_against']
    
    # Sort by win percentage
    home_pitcher_stats = home_pitcher_stats.sort_values('win_pct', ascending=False)
    
    print("\nTop 10 home pitchers by win percentage (min. 5 starts):")
    print(home_pitcher_stats.head(10))
    
    # Away pitcher performance
    away_pitcher_stats = merged_df.groupby('away_starter_id').agg({
        'home_team_won': ['mean', 'count'],
        'away_score': 'mean',
        'home_score': 'mean'
    })
    
    # Flatten multi-index columns
    away_pitcher_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in away_pitcher_stats.columns]
    
    # Reset index for easier handling
    away_pitcher_stats = away_pitcher_stats.reset_index()
    
    # Rename columns
    away_pitcher_stats.rename(columns={
        'home_team_won_mean': 'loss_pct',
        'home_team_won_count': 'games',
        'away_score_mean': 'runs_for',
        'home_score_mean': 'runs_against'
    }, inplace=True)
    
    away_pitcher_stats['win_pct'] = 1 - away_pitcher_stats['loss_pct']
    
    # Filter pitchers with at least 5 starts
    away_pitcher_stats = away_pitcher_stats[away_pitcher_stats['games'] >= 5].copy()
    
    # Calculate run differential
    away_pitcher_stats['run_diff'] = away_pitcher_stats['runs_for'] - away_pitcher_stats['runs_against']
    
    # Sort by win percentage
    away_pitcher_stats = away_pitcher_stats.sort_values('win_pct', ascending=False)
    
    print("\nTop 10 away pitchers by win percentage (min. 5 starts):")
    print(away_pitcher_stats.head(10))
    
    # Plot pitcher impact
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=home_pitcher_stats, x='runs_against', y='win_pct', size='games', sizes=(50, 200), alpha=0.7)
    plt.title('Home Pitcher Win Percentage vs. Runs Allowed')
    plt.xlabel('Average Runs Allowed')
    plt.ylabel('Win Percentage')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'home_pitcher_impact.png'))
    
    return home_pitcher_stats, away_pitcher_stats

def analyze_player_batting(games_df):
    """Analyze individual player batting performance from boxscore data"""
    print("\nAnalyzing player batting performance...")
    
    # Check if we have boxscore data
    game_dirs = [d for d in os.listdir(GAMES_DIR) if os.path.isdir(os.path.join(GAMES_DIR, d))]
    
    # Initialize player data collection
    player_games = []
    
    # Process a subset of games for analysis (adjust number as needed)
    sample_size = min(500, len(game_dirs))
    sample_dirs = game_dirs[:sample_size]
    
    for game_dir in tqdm(sample_dirs, desc="Processing boxscores"):
        game_id = game_dir.replace('game_', '')
        boxscore_file = os.path.join(GAMES_DIR, game_dir, "boxscore.json")
        
        if os.path.exists(boxscore_file):
            try:
                with open(boxscore_file, 'r') as f:
                    boxscore = json.load(f)
                
                # Process home team batters
                if 'home' in boxscore and 'battingStats' in boxscore['home']:
                    home_team_id = boxscore['home'].get('team', {}).get('id', None)
                    if home_team_id is None and isinstance(boxscore['home'], dict):
                        # Try alternate structure
                        home_team_id = boxscore['home'].get('id', None)
                    
                    home_batting = boxscore['home'].get('battingStats', {})
                    
                    for player_id, stats in home_batting.items():
                        if isinstance(stats, dict):  # Ensure stats is a dictionary
                            player_games.append({
                                'game_id': game_id,
                                'player_id': player_id,
                                'team_id': home_team_id,
                                'is_home': True,
                                'at_bats': stats.get('atBats', 0),
                                'hits': stats.get('hits', 0),
                                'home_runs': stats.get('homeRuns', 0),
                                'doubles': stats.get('doubles', 0),
                                'triples': stats.get('triples', 0),
                                'rbi': stats.get('rbi', 0),
                                'walks': stats.get('baseOnBalls', 0),
                                'strikeouts': stats.get('strikeOuts', 0)
                            })
                
                # Process away team batters
                if 'away' in boxscore and 'battingStats' in boxscore['away']:
                    away_team_id = boxscore['away'].get('team', {}).get('id', None) 
                    if away_team_id is None and isinstance(boxscore['away'], dict):
                        # Try alternate structure
                        away_team_id = boxscore['away'].get('id', None)
                    
                    away_batting = boxscore['away'].get('battingStats', {})
                    
                    for player_id, stats in away_batting.items():
                        if isinstance(stats, dict):  # Ensure stats is a dictionary
                            player_games.append({
                                'game_id': game_id,
                                'player_id': player_id,
                                'team_id': away_team_id,
                                'is_home': False,
                                'at_bats': stats.get('atBats', 0),
                                'hits': stats.get('hits', 0),
                                'home_runs': stats.get('homeRuns', 0),
                                'doubles': stats.get('doubles', 0),
                                'triples': stats.get('triples', 0),
                                'rbi': stats.get('rbi', 0),
                                'walks': stats.get('baseOnBalls', 0),
                                'strikeouts': stats.get('strikeOuts', 0)
                            })
                
            except Exception as e:
                print(f"Error processing {boxscore_file}: {e}")
    
    if not player_games:
        print("No player batting data found in boxscores.")
        return None
    
    # Convert to DataFrame
    players_df = pd.DataFrame(player_games)
    
    # Calculate derived statistics
    players_df['hit_indicator'] = (players_df['hits'] > 0).astype(int)
    players_df['hr_indicator'] = (players_df['home_runs'] > 0).astype(int)
    players_df['xbh'] = players_df['doubles'] + players_df['triples'] + players_df['home_runs']  # Extra base hits
    
    # Calculate rates (with handling for divide-by-zero)
    players_df['batting_avg'] = players_df['hits'] / players_df['at_bats'].replace(0, np.nan)
    players_df['hr_rate'] = players_df['home_runs'] / players_df['at_bats'].replace(0, np.nan)
    
    # Player performance summary
    player_summary = players_df.groupby('player_id').agg({
        'game_id': 'count',
        'at_bats': 'sum',
        'hits': 'sum',
        'home_runs': 'sum',
        'hit_indicator': 'sum',
        'hr_indicator': 'sum'
    }).reset_index()
    
    player_summary.columns = ['player_id', 'games', 'at_bats', 'hits', 'home_runs', 'games_with_hit', 'games_with_hr']
    
    # Calculate percentages
    player_summary['batting_avg'] = player_summary['hits'] / player_summary['at_bats']
    player_summary['hit_pct'] = player_summary['games_with_hit'] / player_summary['games']
    player_summary['hr_pct'] = player_summary['games_with_hr'] / player_summary['games']
    
    # Filter players with at least 20 at-bats
    qualified_players = player_summary[player_summary['at_bats'] >= 20].copy()
    
    # Sort by batting average
    top_hitters = qualified_players.sort_values('batting_avg', ascending=False)
    
    print("\nTop 10 hitters by batting average (min. 20 at-bats):")
    print(top_hitters[['player_id', 'games', 'at_bats', 'hits', 'batting_avg']].head(10))
    
    # Sort by home run percentage
    top_power = qualified_players.sort_values('hr_pct', ascending=False)
    
    print("\nTop 10 power hitters by HR percentage (min. 20 at-bats):")
    print(top_power[['player_id', 'games', 'at_bats', 'home_runs', 'games_with_hr', 'hr_pct']].head(10))
    
    # Plot batting average distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(qualified_players['batting_avg'].dropna(), kde=True, bins=20)
    plt.title('Distribution of Batting Averages (min. 20 at-bats)')
    plt.xlabel('Batting Average')
    plt.ylabel('Number of Players')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'batting_avg_distribution.png'))
    
    # Plot home vs. away performance
    home_away = players_df.groupby('is_home').agg({
        'batting_avg': 'mean',
        'hr_rate': 'mean',
        'hit_indicator': 'mean'
    }).reset_index()
    
    home_away['is_home'] = home_away['is_home'].map({True: 'Home', False: 'Away'})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='is_home', y='batting_avg', data=home_away)
    plt.title('Average Batting Performance: Home vs. Away')
    plt.xlabel('')
    plt.ylabel('Average Batting Average')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'home_away_batting.png'))
    
    return players_df, qualified_players

def identify_data_gaps():
    """Identify gaps in current data needed for predictive modeling"""
    print("\nIdentifying data gaps for prediction models...")
    
    data_needs = [
        {
            "category": "Player Performance Prediction",
            "missing_data": [
                "Advanced batting metrics (exit velocity, launch angle, barrel %)",
                "Detailed platoon splits (vs. LHP/RHP)",
                "Batting order position for each game",
                "Player injury status and rest days",
                "Batter vs. pitcher historical matchups",
                "Equipment information (bat types, models)",
                "Recent form metrics (7/15/30 day rolling stats)"
            ]
        },
        {
            "category": "Game Outcome Prediction",
            "missing_data": [
                "Historical betting odds and line movements",
                "Bullpen usage and availability",
                "Detailed weather conditions for all games",
                "Umpire strike zone tendencies",
                "Travel schedule and rest days",
                "Team lineup cards with batting order",
                "Ballpark-specific factors (dimensions, playing surface)"
            ]
        }
    ]
    
    for category in data_needs:
        print(f"\n{category['category']} - Missing Data Elements:")
        for item in category['missing_data']:
            print(f"  • {item}")
    
    print("\nRecommended next steps for data collection:")
    next_steps = [
        "Extend collector to gather individual player game statistics",
        "Incorporate Statcast data for advanced metrics",
        "Find API sources for historical betting odds",
        "Create a database of ballpark dimensions and characteristics",
        "Develop pitcher-batter matchup history collection",
        "Implement collection of detailed weather data for game locations"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"{i}. {step}")

def main():
    """Main analysis function"""
    print("=== MLB Data Analysis ===")
    
    # Load team and game data
    teams_df, games_df = load_team_data()
    
    if games_df is None:
        print("No game data available. Please run data collection first.")
        return
    
    # Analyze home field advantage
    team_home_adv = analyze_home_field_advantage(games_df)
    
    # Analyze runs scored
    monthly_runs = analyze_runs_scored(games_df)
    
    try:
        # Analyze pitcher impact
        pitcher_stats = analyze_pitcher_impact(games_df)
    except Exception as e:
        print(f"Error in pitcher analysis: {e}")
        print("Skipping pitcher analysis.")
        pitcher_stats = None
    
    try:
        # Analyze player batting
        player_stats = analyze_player_batting(games_df)
    except Exception as e:
        print(f"Error in player batting analysis: {e}")
        print("Skipping player batting analysis.")
        player_stats = None
    
    # Identify data gaps for predictive modeling
    identify_data_gaps()
    
    print("\nAnalysis complete. Charts saved to:", ANALYSIS_DIR)

if __name__ == "__main__":
    main()
# explore_mlb_data.py
import pandas as pd
import os
import glob

def explore_batting_stats(year):
    """Explore the batting statistics we've collected"""
    file_path = f"sports_data/mlb/batting_stats_{year}.csv"
    
    if not os.path.exists(file_path):
        print(f"No batting stats file found for {year}")
        return
    
    # Load the data
    batting = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Batting stats for {year}:")
    print(f"Number of players: {len(batting)}")
    print(f"Number of teams: {batting['Team'].nunique()}")
    
    # Display column names
    print("\nColumns available:")
    print(batting.columns.tolist())
    
    # Display top 10 home run hitters
    print("\nTop 10 Home Run Hitters:")
    top_hr = batting.sort_values(by='HR', ascending=False).head(10)
    print(top_hr[['Name', 'Team', 'HR', 'RBI', 'AVG', 'OBP', 'SLG']].to_string(index=False))
    
    return batting

def explore_pitching_stats(year):
    """Explore the pitching statistics we've collected"""
    file_path = f"sports_data/mlb/pitching_stats_{year}.csv"
    
    if not os.path.exists(file_path):
        print(f"No pitching stats file found for {year}")
        return
    
    # Load the data
    pitching = pd.read_csv(file_path)
    
    # Display basic info
    print(f"\nPitching stats for {year}:")
    print(f"Number of pitchers: {len(pitching)}")
    print(f"Number of teams: {pitching['Team'].nunique()}")
    
    # Display column names
    print("\nColumns available:")
    print(pitching.columns.tolist())
    
    # Display top 10 pitchers by ERA (min 50 innings)
    print("\nTop 10 Pitchers by ERA (min 50 IP):")
    qualified = pitching[pitching['IP'] >= 50]
    top_era = qualified.sort_values(by='ERA').head(10)
    print(top_era[['Name', 'Team', 'W', 'L', 'ERA', 'WHIP', 'SO']].to_string(index=False))
    
    return pitching

def explore_team_schedules():
    """Explore the team schedules we've collected"""
    # Find all schedule files
    schedule_files = glob.glob("sports_data/mlb/*_schedule_*.csv")
    
    if not schedule_files:
        print("No team schedule files found")
        return
    
    print(f"\nFound {len(schedule_files)} team schedule files:")
    
    for file in schedule_files:
        team_name = os.path.basename(file).replace("_schedule_", " - ").replace(".csv", "")
        data = pd.read_csv(file)
        
        print(f"\n{team_name}:")
        print(f"Number of games: {len(data)}")
        
        # Check for columns related to wins/losses
        if 'W/L' in data.columns:
            wins = data[data['W/L'] == 'W'].shape[0]
            losses = data[data['W/L'] == 'L'].shape[0]
            print(f"Record: {wins}-{losses}")
        
        # Show a few sample columns if available
        print("Sample columns:")
        print(data.columns.tolist()[:10])  # Show first 10 columns

if __name__ == "__main__":
    # Set the year
    year = 2023
    
    # Explore batting and pitching stats
    batting = explore_batting_stats(year)
    pitching = explore_pitching_stats(year)
    
    # Explore team schedules
    explore_team_schedules()
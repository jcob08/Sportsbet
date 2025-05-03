import os
import pandas as pd
import numpy as np
from datetime import datetime

def integrate_statcast_with_player_data(statcast_dir, player_data_dir, output_dir=None, time_period=None):
    """
    Integrate processed Statcast data with existing player statistics
    
    Args:
        statcast_dir (str): Directory containing processed Statcast data
        player_data_dir (str): Directory containing player game statistics
        output_dir (str, optional): Directory to save integrated data
        time_period (str, optional): Time period to use ('all_time', 'last_7', 'last_15', 'last_30')
        
    Returns:
        tuple: (batter_df, pitcher_df) DataFrames with integrated player and Statcast data
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(player_data_dir), 'enhanced')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default time period
    if time_period is None:
        time_period = 'all_time'
    
    print(f"Integrating Statcast data ({time_period}) with player statistics...")
    
    # Find appropriate processed Statcast files
    batter_filename = f"processed_batter_{time_period}.csv"
    pitcher_filename = f"processed_pitcher_{time_period}.csv"
    
    batter_file = os.path.join(statcast_dir, batter_filename)
    pitcher_file = os.path.join(statcast_dir, pitcher_filename)
    
    # Load Statcast data if available
    batter_statcast = None
    if os.path.exists(batter_file):
        print(f"Loading batter Statcast data from: {batter_file}")
        batter_statcast = pd.read_csv(batter_file)
    else:
        print(f"No batter Statcast file found: {batter_file}")
        # Try to find any processed batter file
        batter_files = [f for f in os.listdir(statcast_dir) if f.startswith('processed_batter_')]
        if batter_files:
            batter_files.sort(key=lambda x: os.path.getmtime(os.path.join(statcast_dir, x)), reverse=True)
            alt_batter_file = os.path.join(statcast_dir, batter_files[0])
            print(f"Using alternative batter file: {alt_batter_file}")
            batter_statcast = pd.read_csv(alt_batter_file)
    
    pitcher_statcast = None
    if os.path.exists(pitcher_file):
        print(f"Loading pitcher Statcast data from: {pitcher_file}")
        pitcher_statcast = pd.read_csv(pitcher_file)
    else:
        print(f"No pitcher Statcast file found: {pitcher_file}")
        # Try to find any processed pitcher file
        pitcher_files = [f for f in os.listdir(statcast_dir) if f.startswith('processed_pitcher_')]
        if pitcher_files:
            pitcher_files.sort(key=lambda x: os.path.getmtime(os.path.join(statcast_dir, x)), reverse=True)
            alt_pitcher_file = os.path.join(statcast_dir, pitcher_files[0])
            print(f"Using alternative pitcher file: {alt_pitcher_file}")
            pitcher_statcast = pd.read_csv(alt_pitcher_file)
    
    # Find player statistics files
    current_year = datetime.now().year
    player_files = []
    
    # Check for different years of player data
    for year in range(current_year-1, current_year+1):  # Look for last year and current year
        filename = f"player_game_stats_{year}.csv"
        file_path = os.path.join(player_data_dir, filename)
        if os.path.exists(file_path):
            player_files.append((year, file_path))
    
    if not player_files:
        print(f"No player statistics files found in {player_data_dir}")
        return None, None
    
    # Process each player statistics file
    batter_results = []
    pitcher_results = []
    
    for year, player_file in player_files:
        print(f"Processing player statistics for {year} from {player_file}")
        
        # Load player data
        player_data = pd.read_csv(player_file)
        
        # Print column names to debug
        print(f"Available columns in player data: {player_data.columns.tolist()}")
        
        # Identify batters
        is_batter_data = 'at_bats' in player_data.columns or 'got_hit' in player_data.columns
        
        # Process batter data
        if is_batter_data and batter_statcast is not None:
            print(f"Processing batter statistics for {year}")
            
            # Define aggregation columns based on available columns
            agg_dict = {}
            
            # Handle required ID column first
            if 'player_id' not in player_data.columns:
                print("Error: No player_id column found in player data")
                continue
            
            # Build aggregation dictionary based on available columns
            if 'at_bats' in player_data.columns:
                agg_dict['at_bats'] = 'sum'
            
            if 'got_hit' in player_data.columns:
                agg_dict['got_hit'] = ['sum', 'mean']
                
            if 'got_home_run' in player_data.columns:
                agg_dict['got_home_run'] = ['sum', 'mean']
                
            if 'game_date' in player_data.columns:
                agg_dict['game_date'] = ['min', 'max', 'count']
            
            # Add any player identifying columns if they exist
            for col in ['player_name', 'team', 'position']:
                if col in player_data.columns:
                    agg_dict[col] = 'first'
            
            # Skip if we don't have enough data to aggregate
            if len(agg_dict) == 0:
                print(f"Not enough data columns to aggregate for {year}")
                continue
                
            # Aggregate batter stats
            print(f"Aggregating with dictionary: {agg_dict}")
            batter_stats = player_data.groupby('player_id').agg(agg_dict).reset_index()
            
            # Flatten multi-level column names if we have any
            if any(isinstance(x, tuple) for x in batter_stats.columns):
                batter_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                      for col in batter_stats.columns]
            
            # Rename columns for clarity if they exist
            rename_dict = {}
            if 'got_hit_sum' in batter_stats.columns:
                rename_dict['got_hit_sum'] = 'hits'
            if 'got_hit_mean' in batter_stats.columns:
                rename_dict['got_hit_mean'] = 'batting_avg'
            if 'got_home_run_sum' in batter_stats.columns:
                rename_dict['got_home_run_sum'] = 'home_runs'
            if 'got_home_run_mean' in batter_stats.columns:
                rename_dict['got_home_run_mean'] = 'hr_rate'
            if 'game_date_min' in batter_stats.columns:
                rename_dict['game_date_min'] = 'first_game_date'
            if 'game_date_max' in batter_stats.columns:
                rename_dict['game_date_max'] = 'last_game_date'
            if 'game_date_count' in batter_stats.columns:
                rename_dict['game_date_count'] = 'games_played'
                
            if rename_dict:
                batter_stats = batter_stats.rename(columns=rename_dict)
            
            print(f"Aggregated columns: {batter_stats.columns.tolist()}")
            
            # Ensure player_id is string type for merging
            batter_stats['player_id'] = batter_stats['player_id'].astype(str)
            batter_statcast['player_id'] = batter_statcast['player_id'].astype(str)
            
            # Define key Statcast metrics to include (adapt based on what's available)
            available_statcast_cols = batter_statcast.columns.tolist()
            print(f"Available Statcast columns: {available_statcast_cols}")
            
            # Select all statcast columns except player_name (to avoid conflicts)
            statcast_cols_to_use = [col for col in available_statcast_cols 
                                 if col != 'player_name' or 'player_name' not in batter_stats.columns]
            
            # Create subset of Statcast metrics
            batter_statcast_subset = batter_statcast[statcast_cols_to_use]
            
            # Merge player stats with Statcast metrics
            enhanced_batter_data = batter_stats.merge(
                batter_statcast_subset,
                on='player_id',
                how='left'
            )
            
            print(f"Enhanced {len(enhanced_batter_data)} batter records with Statcast metrics")
            
            # Add year column and save
            enhanced_batter_data['year'] = year
            enhanced_batter_data['time_period'] = time_period
            
            batter_results.append(enhanced_batter_data)
            
            # Save enhanced batter data
            output_file = os.path.join(output_dir, f"enhanced_batter_stats_{year}_{time_period}.csv")
            enhanced_batter_data.to_csv(output_file, index=False)
            print(f"Saved enhanced batter data to {output_file}")
            
            # Print sample data
            print("\nSample of enhanced batter data:")
            print(enhanced_batter_data.head(3))
    
    # Combine data across years if multiple years exist
    combined_batter_data = pd.concat(batter_results) if batter_results else None
    combined_pitcher_data = pd.concat(pitcher_results) if pitcher_results else None
    
    # If multiple years, save combined data
    if len(batter_results) > 1 and combined_batter_data is not None:
        combined_file = os.path.join(output_dir, f"enhanced_batter_stats_combined_{time_period}.csv")
        combined_batter_data.to_csv(combined_file, index=False)
        print(f"Saved combined enhanced batter data to {combined_file}")
    
    if len(pitcher_results) > 1 and combined_pitcher_data is not None:
        combined_file = os.path.join(output_dir, f"enhanced_pitcher_stats_combined_{time_period}.csv")
        combined_pitcher_data.to_csv(combined_file, index=False)
        print(f"Saved combined enhanced pitcher data to {combined_file}")
    
    return combined_batter_data, combined_pitcher_data

def test_statcast_integration():
    """
    Test the integration of Statcast data with player statistics
    """
    # Set up paths based on your project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    statcast_dir = os.path.join(base_dir, 'data', 'raw', 'processed', 'statcast')
    player_data_dir = os.path.join(base_dir, 'sports_data', 'mlb', 'analysis')
    output_dir = os.path.join(base_dir, 'sports_data', 'mlb', 'enhanced')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test integration with different time periods
    time_periods = ['all_time', 'last_30', 'last_15', 'last_7']
    
    for period in time_periods:
        print(f"\n=== Testing integration with {period} Statcast data ===")
        batter_df, pitcher_df = integrate_statcast_with_player_data(
            statcast_dir,
            player_data_dir,
            output_dir,
            time_period=period
        )
    
    print("\nIntegration testing complete!")

if __name__ == "__main__":
    test_statcast_integration()
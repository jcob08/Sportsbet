import os
import requests
import pandas as pd
from datetime import datetime

def get_statcast_data(start_date, end_date, player_type='batter'):
    """
    Retrieve Statcast data for all players within a date range
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        player_type (str): Either 'batter' or 'pitcher'
        
    Returns:
        pandas DataFrame with Statcast data
    """
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    
    params = {
        'all': 'true',
        'hfPT': '',
        'hfAB': '',
        'hfGT': '',
        'hfPR': '',
        'hfZ': '',
        'stadium': '',
        'hfBBL': '',
        'hfNewZones': '',
        'hfPull': '',
        'hfC': '',
        'hfSea': f'{start_date[:4]}|',
        'hfSit': '',
        'player_type': player_type,
        'hfOuts': '',
        'opponent': '',
        'pitcher_throws': '',
        'batter_stands': '',
        'hfSA': '',
        'game_date_gt': start_date,
        'game_date_lt': end_date,
        'hfInfield': '',
        'team': '',
        'position': '',
        'hfOutfield': '',
        'hfRO': '',
        'home_road': '',
        'hfFlag': '',
        'hfBBT': '',
        'metric_1': '',
        'hfInn': '',
        'min_pitches': '0',
        'min_results': '0',
        'group_by': 'name',
        'sort_col': 'pitches',
        'player_event_sort': 'api_p_release_speed',
        'sort_order': 'desc',
        'min_pas': '0',
        'type': 'details'
    }
    
    print(f"Requesting Statcast data for {player_type}s from {start_date} to {end_date}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Save raw CSV
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'raw', 'statcast')
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = f"statcast_{player_type}_{start_date}_to_{end_date}.csv"
        file_path = os.path.join(raw_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Read CSV into DataFrame
        df = pd.read_csv(file_path)
        print(f"Successfully retrieved {len(df)} Statcast records")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Statcast data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Simple test function
def test_statcast_fetcher():
    # Test with a small date range
    test_start_date = "2024-04-01"
    test_end_date = "2024-04-03"
    
    # Test for batters
    batter_data = get_statcast_data(test_start_date, test_end_date, 'batter')
    if not batter_data.empty:
        print(f"Successfully retrieved batter data with {len(batter_data)} rows")
        print(f"Columns: {batter_data.columns.tolist()[:5]}...")
    else:
        print("Failed to retrieve batter data")
    
    # Test for pitchers
    pitcher_data = get_statcast_data(test_start_date, test_end_date, 'pitcher')
    if not pitcher_data.empty:
        print(f"Successfully retrieved pitcher data with {len(pitcher_data)} rows")
        print(f"Columns: {pitcher_data.columns.tolist()[:5]}...")
    else:
        print("Failed to retrieve pitcher data")

if __name__ == "__main__":
    test_statcast_fetcher()
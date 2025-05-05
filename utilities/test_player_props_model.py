# test_player_props_model.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the player props model
from models.player_props_model import MLBPlayerPropsModel

def test_player_props_model():
    """Test the MLB player props model functionality"""
    print("=== Testing MLB Player Props Model ===")
    
    # Create the model
    props_model = MLBPlayerPropsModel()
    
    # Load data
    props_model.load_data()
    
    # Generate sample games if needed
    if props_model.game_data is None:
        print("Creating sample games for testing...")
        sample_games = create_sample_games()
    else:
        print(f"Using {len(props_model.game_data)} existing games")
        # Just use a subset for faster testing
        sample_games = props_model.game_data.sample(min(5, len(props_model.game_data)))
    
    # Test individual prop types
    print("\n=== Testing Individual Prop Types ===")
    
    # Hits props
    print("\nTesting hits props...")
    hit_props = props_model.predict_batter_hit_model(sample_games, threshold=1)
    if not hit_props.empty:
        print(f"Generated {len(hit_props)} hit prop predictions")
        print("\nSample hit props:")
        print(hit_props[['player_name', 'team', 'expected_value', 'line', 'over_prob', 'edge']].head(3))
    
    # Home run props
    print("\nTesting home run props...")
    hr_props = props_model.predict_batter_home_run_model(sample_games)
    if not hr_props.empty:
        print(f"Generated {len(hr_props)} home run prop predictions")
        print("\nSample home run props:")
        print(hr_props[['player_name', 'team', 'expected_value', 'line', 'over_prob', 'edge']].head(3))
    
    # Strikeout props
    print("\nTesting strikeout props...")
    k_props = props_model.predict_pitcher_strikeouts(sample_games)
    if not k_props.empty:
        print(f"Generated {len(k_props)} strikeout prop predictions")
        print("\nSample strikeout props:")
        print(k_props[['player_name', 'team', 'expected_value', 'line', 'over_prob', 'edge']].head(3))
    
    # Try the full workflow
    print("\n=== Testing Full Props Workflow ===")
    # Use a valid date string from the sample_games or game_data
    if not sample_games.empty:
        test_date = str(sample_games.iloc[0]['game_date'])[:10]
    else:
        test_date = "2024-04-01"
    all_props = props_model.run_daily_workflow(date=test_date)
    
    # Get best betting opportunities
    print("\n=== Finding Best Betting Opportunities ===")
    best_props = props_model.get_best_props_by_edge(min_edge=0.15)
    
    if not best_props.empty:
        print(f"Found {len(best_props)} props with edge > 15%")
        print("\nTop betting opportunities:")
        cols = ['player_name', 'team', 'prop_type', 'line', 'expected_value', 'edge']
        available_cols = [col for col in cols if col in best_props.columns]
        print(best_props[available_cols].head(5))
    
    print("\n=== Player Props Model Testing Complete ===")

def create_sample_games():
    """Create sample games data for testing"""
    sample_games = pd.DataFrame({
        'game_id': ['2025-001', '2025-002', '2025-003', '2025-004', '2025-005'],
        'game_date': ['2025-05-01', '2025-05-01', '2025-05-02', '2025-05-02', '2025-05-03'],
        'home_team': ['NYY', 'LAD', 'BOS', 'CHC', 'HOU'],
        'away_team': ['BOS', 'SF', 'NYY', 'STL', 'TEX'],
        'home_team_id': [147, 119, 111, 112, 117],
        'away_team_id': [111, 137, 147, 138, 140],
        'home_win_pct': [0.600, 0.580, 0.520, 0.490, 0.610],
        'away_win_pct': [0.520, 0.510, 0.600, 0.550, 0.480],
        'ballpark_run_factor': [0.05, -0.02, 0.08, 0.03, -0.01],
        'ballpark_hr_factor': [0.10, -0.03, 0.05, 0.02, 0.00],
        'weather_score': [0.02, 0.01, -0.01, 0.03, 0.00],
        'umpire_strikeout_boost': [0.05, -0.03, 0.02, 0.00, 0.04],
        'umpire_runs_boost': [-0.02, 0.01, -0.01, 0.00, 0.02],
        'home_starting_pitcher_id': ['543243', '477132', '592789', '608566', '572971'],
        'away_starting_pitcher_id': ['656756', '621244', '543243', '608566', '592789']
    })
    
    return sample_games

if __name__ == "__main__":
    test_player_props_model()
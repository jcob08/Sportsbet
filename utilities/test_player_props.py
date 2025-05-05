# utilities/test_player_props.py
import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlb_betting_model import MLBBettingModel

def test_player_props():
    print("=== Testing Player Props Predictions ===")
    
    # Create a betting model instance
    betting_model = MLBBettingModel()
    
    # Load sample game data
    games_df = pd.DataFrame({
        'game_id': ['12345', '12346', '12347'],
        'game_date': ['2025-04-01', '2025-04-01', '2025-04-02'],
        'home_team': ['NYY', 'LAD', 'BOS'],
        'away_team': ['BOS', 'SF', 'NYY'],
        'ballpark_run_factor': [0.05, 0.02, 0.03],
        'ballpark_hr_factor': [0.10, 0.08, 0.02],
        'weather_score': [0.02, 0.01, -0.01]
    })
    
    # Create sample player projections
    player_projections = pd.DataFrame({
        'player_id': [str(id) for id in range(10001, 10021)],
        'player_name': [f"Player {i}" for i in range(1, 21)],
        'hit_rate': np.random.uniform(0.3, 0.6, 20),
        'hr_rate': np.random.uniform(0.01, 0.1, 20),
        'projected_hit_rate_2025': np.random.uniform(0.3, 0.6, 20),
        'projected_hr_rate_2025': np.random.uniform(0.01, 0.1, 20)
    })
    
    # Test hits props
    print("\nTesting hits props predictions...")
    hits_props = betting_model.predict_player_props(
        games_df=games_df,
        player_projections=player_projections,
        prop_type='hits'
    )
    
    if not hits_props.empty:
        print("Successfully generated hits props!")
        print(f"Number of predictions: {len(hits_props)}")
        print("\nSample hits predictions:")
        sample_cols = ['game_id', 'player_name', 'team', 'prop_type', 'expected_value', 'line', 'over_prob']
        print(hits_props[sample_cols].head())
    else:
        print("Failed to generate hits props.")
    
    # Test home run props
    print("\nTesting home run props predictions...")
    hr_props = betting_model.predict_player_props(
        games_df=games_df,
        player_projections=player_projections,
        prop_type='hr'
    )
    
    if not hr_props.empty:
        print("Successfully generated home run props!")
        print(f"Number of predictions: {len(hr_props)}")
        print("\nSample home run predictions:")
        print(hr_props[sample_cols].head())
    else:
        print("Failed to generate home run props.")
    
    print("\n=== Player Props Testing Complete ===")

if __name__ == "__main__":
    test_player_props()
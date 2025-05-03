import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import from models directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import if the model file exists, otherwise we'll handle the error
try:
    from models.mlb_betting_model import MLBBettingModel
except ImportError:
    print("MLBBettingModel not found. Please create the file first.")
    sys.exit(1)

def test_model_integration():
    """
    Test the integration of contextual features into MLB betting models
    """
    # Initialize the betting model
    betting_model = MLBBettingModel()
    
    # Load game data with contextual features
    # Let's specify the exact filename to avoid the error
    data = betting_model.load_data("games_with_engineered_features.csv")
    
    # Compare models with and without contextual features
    comparison = betting_model.compare_models(data)
    
    # Train and evaluate a model for total runs
    betting_model.train_model(data, target_type='totals', include_contextual=True)
    betting_model.evaluate_model(data)
    
    # Analyze feature importance
    importance = betting_model.feature_importance()
    
    # Make predictions for new games
    new_games = data.sample(5)  # Simulate new games
    predictions = betting_model.make_predictions(new_games)
    
    print("\nSample predictions:")
    display_cols = ['game_id', 'home_team', 'away_team', 
                   'predicted_total_runs', 'total_line', 'bet_recommendation', 'edge']
    available_cols = [col for col in display_cols if col in predictions.columns]
    print(predictions[available_cols])
    
    return betting_model

if __name__ == "__main__":
    betting_model = test_model_integration()
# main.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project components
from models.mlb_betting_model import MLBBettingModel
from utilities.contextual_feature_engineer import ContextualFeatureEngineer
from utilities.statcast_processor import process_statcast_data
from utilities.collect_game_outcomes import collect_game_outcomes

def main():
    print("=== MLB Betting Model Execution ===")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Step 1: Collect latest game outcomes
    year = datetime.now().year
    print(f"\n1. Collecting game outcomes for {year}...")
    game_outcomes = collect_game_outcomes(year)
    
    if game_outcomes.empty:
        print("No game outcomes available. Please run data collection first.")
        return
    
    # Step 2: Apply contextual feature engineering
    print("\n2. Engineering contextual features...")
    engineer = ContextualFeatureEngineer()
    output_file = "data/contextual/games_with_engineered_features.csv"
    engineered_features = engineer.engineer_features(game_outcomes, output_file=output_file)
    
    # Step 3: Load contextual data
    print("\n3. Loading contextual data...")
    try:
        contextual_data = pd.read_csv(output_file)
        print(f"Loaded {len(contextual_data)} games with contextual features")
    except Exception as e:
        print(f"Error loading contextual data: {e}")
        return
    
    # Step 4: Initialize betting model
    print("\n4. Initializing betting model...")
    betting_model = MLBBettingModel()
    
    # Step 5: Train and evaluate models
    print("\n5. Training and evaluating models...")
    model_types = ['totals', 'moneyline', 'strikeouts']
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        betting_model.train_model(contextual_data, target_type=model_type, include_contextual=True)
        model_id = f"{model_type}_context_gbm"
        betting_model.evaluate_model(contextual_data, model_id=model_id)
    
    # Step 6: Generate player prop predictions
    print("\n6. Generating player prop predictions...")
    player_projections, batter_statcast, pitcher_statcast = betting_model.load_player_data()
    
    for prop_type in ['hits', 'hr', 'strikeouts']:
        print(f"\nPredicting {prop_type} props...")
        prop_predictions = betting_model.predict_player_props(
            contextual_data, player_projections, batter_statcast, pitcher_statcast, prop_type=prop_type
        )
        if not prop_predictions.empty:
            print(f"Generated {len(prop_predictions)} {prop_type} prop predictions")
    
    print("\n=== MLB Betting Model Execution Complete ===")

if __name__ == "__main__":
    main()
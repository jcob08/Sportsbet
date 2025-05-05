# utilities/test_model_integration.py
import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory (C:\MLB-Betting) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlb_betting_model import MLBBettingModel
from utilities.contextual_feature_engineer import ContextualFeatureEngineer

def test_model_integration():
    print("Loading real game outcomes from C:\\MLB-Betting\\sports_data\\mlb\\contextual\\game_outcomes_2024.csv")
    game_outcomes_2024 = pd.read_csv("C:\\MLB-Betting\\sports_data\\mlb\\contextual\\game_outcomes_2024.csv")
    game_outcomes_2024['game_id'] = game_outcomes_2024['game_id'].astype(str)
    
    print("Creating contextual data with real game_ids at data/contextual\\games_with_engineered_features.csv")
    engineer = ContextualFeatureEngineer()
    output_file = "data/contextual/games_with_engineered_features.csv"
    engineered_features = engineer.engineer_features(game_outcomes_2024, output_file=output_file)
    
    print("Loading contextual data from data/contextual\\games_with_engineered_features.csv")
    contextual_data = pd.read_csv("data/contextual/games_with_engineered_features.csv")
    contextual_data['game_id'] = contextual_data['game_id'].astype(str)
    
    data = pd.merge(game_outcomes_2024, contextual_data, on=['game_id', 'home_team', 'away_team'], how='inner')
    print(f"Merged {len(data)} real game outcomes (after inner join)")
    print("Columns after merge:", data.columns.tolist())
    
    # Resolve duplicate columns from the merge (e.g., home_starting_pitcher_id_x, home_starting_pitcher_id_y)
    duplicate_cols = ['home_starting_pitcher_id', 'away_starting_pitcher_id', 'home_starting_lineup', 'away_starting_lineup']
    for col in duplicate_cols:
        col_x = f"{col}_x"
        col_y = f"{col}_y"
        if col_x in data.columns and col_y in data.columns:
            if data[col_x].equals(data[col_y]):
                data = data.drop(columns=[col_y])
                data = data.rename(columns={col_x: col})
            else:
                print(f"Warning: {col_x} and {col_y} differ. Using {col_x} by default.")
                data = data.drop(columns=[col_y])
                data = data.rename(columns={col_x: col})
    
    # Drop other duplicate columns (e.g., home_team_runs_x, home_team_runs_y)
    for col in data.columns:
        if col.endswith('_y') and col[:-2] + '_x' in data.columns:
            data = data.drop(columns=[col])
            data = data.rename(columns={col[:-2] + '_x': col[:-2]})
    
    print("Columns after resolving duplicates:", data.columns.tolist())
    
    betting_model = MLBBettingModel()
    
    batter_statcast_file = "C:\\MLB-Betting\\data\\processed\\statcast\\processed_batter_all_time.csv"
    pitcher_statcast_file = "C:\\MLB-Betting\\data\\processed\\statcast\\processed_pitcher_all_time.csv"
    
    if os.path.exists(batter_statcast_file):
        print(f"Found batter Statcast file: {batter_statcast_file}")
        batter_statcast = pd.read_csv(batter_statcast_file)
        batter_statcast['player_id'] = batter_statcast['player_id'].astype(str)
    else:
        print("Batter Statcast file not found. Creating placeholder...")
        batter_statcast = pd.DataFrame(columns=['player_id', 'xwoba', 'barrel_rate'])
    
    if os.path.exists(pitcher_statcast_file):
        print(f"Found pitcher Statcast file: {pitcher_statcast_file}")
        pitcher_statcast = pd.read_csv(pitcher_statcast_file)
        pitcher_statcast['player_id'] = pitcher_statcast['player_id'].astype(str)
    else:
        print("Pitcher Statcast file not found. Creating placeholder...")
        pitcher_statcast = pd.DataFrame(columns=['player_id', 'velocity_avg', 'whiff_rate', 'stuff_plus'])
    
    print("Adding batter Statcast features...")
    # Initialize Statcast columns with default values to avoid NaN
    data['home_team_xwoba'] = 0.32
    data['away_team_xwoba'] = 0.32
    data['home_team_barrel_rate'] = 10.0
    data['away_team_barrel_rate'] = 10.0
    
    for idx, row in data.iterrows():
        for team in ['home', 'away']:
            lineup_col = f"{team}_starting_lineup"
            if lineup_col in row and pd.notna(row[lineup_col]):
                player_ids = str(row[lineup_col]).split(',')
                player_ids = [pid.strip() for pid in player_ids if pid.strip()]
                
                team_xwoba = []
                team_barrel_rate = []
                
                for pid in player_ids:
                    if pid in batter_statcast['player_id'].values:
                        player_data = batter_statcast[batter_statcast['player_id'] == pid]
                        if 'xwoba' in player_data.columns:
                            team_xwoba.append(player_data['xwoba'].iloc[0])
                        if 'barrel_rate' in player_data.columns:
                            team_barrel_rate.append(player_data['barrel_rate'].iloc[0])
                
                data.loc[idx, f"{team}_team_xwoba"] = np.mean(team_xwoba) if team_xwoba else 0.32
                data.loc[idx, f"{team}_team_barrel_rate"] = np.mean(team_barrel_rate) if team_barrel_rate else 10.0
            else:
                data.loc[idx, f"{team}_team_xwoba"] = 0.32
                data.loc[idx, f"{team}_team_barrel_rate"] = 10.0
    
    print("Adding pitcher Statcast features...")
    # Initialize pitcher Statcast columns with default values to avoid NaN
    data['home_pitcher_velocity_avg'] = 93.0
    data['away_pitcher_velocity_avg'] = 93.0
    data['home_pitcher_whiff_rate'] = 0.25
    data['away_pitcher_whiff_rate'] = 0.25
    data['home_pitcher_stuff_plus'] = 100.0
    data['away_pitcher_stuff_plus'] = 100.0
    
    for idx, row in data.iterrows():
        for team in ['home', 'away']:
            pitcher_id = row.get(f"{team}_starting_pitcher_id")
            if pd.notna(pitcher_id):
                pitcher_id = str(pitcher_id)
                if pitcher_id in pitcher_statcast['player_id'].values:
                    pitcher_data = pitcher_statcast[pitcher_statcast['player_id'] == pid]
                    data.loc[idx, f"{team}_pitcher_velocity_avg"] = pitcher_data['velocity_avg'].iloc[0] if 'velocity_avg' in pitcher_data.columns else 93.0
                    data.loc[idx, f"{team}_pitcher_whiff_rate"] = pitcher_data['whiff_rate'].iloc[0] if 'whiff_rate' in pitcher_data.columns else 0.25
                    data.loc[idx, f"{team}_pitcher_stuff_plus"] = pitcher_data['stuff_plus'].iloc[0] if 'stuff_plus' in pitcher_data.columns else 100.0
                else:
                    data.loc[idx, f"{team}_pitcher_velocity_avg"] = 93.0
                    data.loc[idx, f"{team}_pitcher_whiff_rate"] = 0.25
                    data.loc[idx, f"{team}_pitcher_stuff_plus"] = 100.0
            else:
                data.loc[idx, f"{team}_pitcher_velocity_avg"] = 93.0
                data.loc[idx, f"{team}_pitcher_whiff_rate"] = 0.25
                data.loc[idx, f"{team}_pitcher_stuff_plus"] = 100.0
    
    data.to_csv("data/contextual/games_with_statcast_features.csv", index=False)
    print("Saved enhanced data with Statcast features to data/contextual\\games_with_statcast_features.csv")
    
    print("\n--- Comparing models with and without contextual features ---")
    comparison_results = betting_model.compare_models(data)
    
    for model_id in ['totals_context_gbm', 'moneyline_context_gbm', 'strikeouts_context_gbm']:
        print(f"\n--- Training and evaluating {model_id.split('_')[0]} model ---")
        betting_model.train_model(data, target_type=model_id.split('_')[0], include_contextual=True)
        betting_model.evaluate_model(data, model_id=model_id)
        importance = betting_model.feature_importance(model_id=model_id)
    
    print("\n--- Sample predictions ---")
    sample_games = data.sample(5, random_state=42)
    
    # Map model target names to actual column names
    target_mapping = {
        'totals': 'total_runs',
        'moneyline': 'home_team_win',
        'strikeouts': 'total_strikeouts'
    }
    
    for model_id in ['totals_context_gbm', 'moneyline_context_gbm', 'strikeouts_context_gbm']:
        predictions = betting_model.make_predictions(sample_games, model_id=model_id)
        target_key = model_id.split('_')[0]  # e.g., 'totals', 'moneyline', 'strikeouts'
        target = target_mapping[target_key]  # e.g., 'total_runs', 'home_team_win', 'total_strikeouts'
        columns_to_show = ['game_id', 'home_team', 'away_team', target]
        predicted_col = f"predicted_{target_key}" if target_key == 'moneyline' else f"predicted_{target}"
        if predicted_col in predictions.columns:
            columns_to_show.append(predicted_col)
        print(f"\nSample {target_key} predictions:")
        print(predictions[columns_to_show])
    
    print("\n--- Predicting player props ---")

    try:
        player_projections, batter_statcast, pitcher_statcast = betting_model.load_player_data()
        
        # First, create player projections if they don't exist
        if player_projections is None:
            print("Creating player projections file...")
            from utilities.create_player_projections import create_player_projections
            player_projections = create_player_projections()
        
        for prop_type in ['hits', 'hr', 'strikeouts']:
            print(f"\nPredicting {prop_type} props...")
            try:
                prop_predictions = betting_model.predict_player_props(
                    sample_games, player_projections, batter_statcast, pitcher_statcast, prop_type=prop_type
                )
                if prop_predictions is not None and not prop_predictions.empty:
                    print(f"Successfully generated {len(prop_predictions)} {prop_type} prop predictions")
                    print(f"\nSample {prop_type} prop predictions:")
                    
                    cols_to_show = ['game_id', 'team', 'player_id', 'player_name', 'prop_type']
                    if 'expected_value' in prop_predictions.columns:
                        cols_to_show.append('expected_value')
                    if 'line' in prop_predictions.columns:
                        cols_to_show.append('line')
                    if 'over_prob' in prop_predictions.columns:
                        cols_to_show.append('over_prob')
                    
                    available_cols = [col for col in cols_to_show if col in prop_predictions.columns]
                    print(prop_predictions[available_cols].head())
                else:
                    print(f"No {prop_type} prop predictions generated.")
            except Exception as e:
                print(f"Error predicting {prop_type} props: {e}")
    except Exception as e:
        print(f"Error in player props prediction: {e}")

    print("\n--- Model integration testing complete ---")
    return betting_model

if __name__ == "__main__":
    betting_model = test_model_integration()
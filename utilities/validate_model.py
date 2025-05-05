import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.mlb_betting_model import MLBBettingModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the merged game data
merged_data_path = 'data/contextual/merged_game_data_2024.csv'
merged_df = pd.read_csv(merged_data_path)

# Rename columns to match what the model expects
merged_df = merged_df.rename(columns={
    'home_team_x': 'home_team',
    'away_team_x': 'away_team',
    'home_team_y': 'home_team_alt',
    'away_team_y': 'away_team_alt'
})

print("\nDataset Info:")
print(f"Total rows: {len(merged_df)}")
print("\nSample of the data:")
print(merged_df.head())

print("\nMissing values:")
print(merged_df.isnull().sum())

# Initialize the model
model = MLBBettingModel()

# Train the model on the merged data
pipe, features, _ = model.train_model(merged_df)

# Evaluate the model
print("\nModel Evaluation:")
model.evaluate_model(merged_df)

# Print feature importance
print("\nFeature Importance:")
model.feature_importance() 
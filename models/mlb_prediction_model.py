# mlb_prediction_model.py
import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = "sports_data/mlb/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_training_data(year):
    """Load processed game features for training"""
    features_file = os.path.join(DATA_DIR, f"game_features_{year}.csv")
    
    if not os.path.exists(features_file):
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} game records for training")
        return df
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def train_model(features_df):
    """Train the prediction model"""
    logger.info("Training prediction model")
    
    # Drop non-feature columns
    X = features_df.drop(['game_id', 'game_date', 'home_team_id', 'away_team_id', 'home_team_won'], axis=1)
    y = features_df['home_team_won']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Feature importance:")
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance:.4f}")
    
    return model

def save_model(model, year):
    """Save the trained model"""
    model_file = os.path.join(MODEL_DIR, f"mlb_model_{year}.pkl")
    
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def predict_upcoming_games(model, year):
    """Make predictions for upcoming games"""
    # This would typically use the MLB API to get upcoming games
    # For this example, we'll simulate upcoming games
    
    logger.info("Predicting upcoming games")
    
    # Load team stats
    team_stats_file = os.path.join(DATA_DIR, f"team_stats_{year}.csv")
    
    if not os.path.exists(team_stats_file):
        logger.error(f"Team stats file not found: {team_stats_file}")
        return
    
    team_stats_df = pd.read_csv(team_stats_file)
    
    # Create sample upcoming games (in a real implementation, you would fetch these from the API)
    # For this example, let's create matchups between the top teams
    top_teams = team_stats_df.sort_values('win_pct', ascending=False).head(10)
    
    upcoming_games = []
    for i in range(len(top_teams) - 1):
        for j in range(i + 1, len(top_teams)):
            home_team = top_teams.iloc[i]
            away_team = top_teams.iloc[j]
            
            # Create feature dictionary for home advantage scenario
            game_features = {
                'home_win_pct': home_team['win_pct'],
                'home_home_win_pct': home_team['home_win_pct'],
                'away_win_pct': away_team['win_pct'],
                'away_away_win_pct': away_team['away_win_pct'],
                'win_pct_diff': home_team['win_pct'] - away_team['win_pct']
            }
            
            upcoming_games.append({
                'home_team_id': home_team['team_id'],
                'away_team_id': away_team['team_id'],
                'features': game_features
            })
    
    # Make predictions
    for game in upcoming_games:
        features_df = pd.DataFrame([game['features']])
        
        # Predict probability of home team winning
        win_prob = model.predict_proba(features_df)[0][1]
        
        logger.info(f"Game: Team {game['home_team_id']} vs Team {game['away_team_id']}")
        logger.info(f"Prediction: {win_prob:.2f} probability of home team winning")
        logger.info("---")

def main():
    """Main execution function"""
    current_year = datetime.now().year
    # Use previous year if we're in the offseason (before April)
    if datetime.now().month < 4:
        year = current_year - 1
    else:
        year = current_year
    
    logger.info(f"Starting MLB prediction model for {year}")
    
    # Load training data
    features_df = load_training_data(year)
    
    if features_df is not None:
        # Train the model
        model = train_model(features_df)
        
        # Save the model
        save_model(model, year)
        
        # Make predictions for upcoming games
        predict_upcoming_games(model, year)
    
    logger.info("MLB prediction model completed")

if __name__ == "__main__":
    main()
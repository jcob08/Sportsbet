# mlb_prediction_framework.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBPredictionModel:
    def __init__(self):
        self.data_dir = "sports_data/mlb"
        self.model = None
        self.scaler = None
        self.features = None
        
    def load_data(self):
        """Load and prepare data for the model"""
        logger.info("Loading data for MLB prediction model...")
        
        # Check if batting and pitching data exist
        batting_file = os.path.join(self.data_dir, "batting_stats_2022.csv")
        pitching_file = os.path.join(self.data_dir, "pitching_stats_2022.csv")
        
        if not (os.path.exists(batting_file) and os.path.exists(pitching_file)):
            logger.error("Required data files not found. Please run data collector first.")
            return False
        
        # Load data
        try:
            self.batting_data = pd.read_csv(batting_file)
            self.pitching_data = pd.read_csv(pitching_file)
            logger.info("Data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for model training"""
        logger.info("Preparing features...")
        
        # For demonstration purposes, we'll create dummy data for game outcomes
        # In a real model, you would use actual game results
        
        # Create team-level features
        team_batting = self.batting_data.groupby('Team').agg({
            'AVG': 'mean',
            'OBP': 'mean',
            'SLG': 'mean',
            'HR': 'sum',
            'RBI': 'sum'
        }).reset_index()
        
        team_pitching = self.pitching_data.groupby('Team').agg({
            'ERA': 'mean',
            'WHIP': 'mean',
            'SO': 'sum',
            'BB': 'sum'
        }).reset_index()
        
        # Merge team stats
        team_stats = pd.merge(team_batting, team_pitching, on='Team')
        
        # Create dummy game data (for demonstration)
        np.random.seed(42)
        games = []
        teams = team_stats['Team'].tolist()
        
        for i in range(500):  # Generate 500 sample games
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Binary outcome: 1 if home team wins, 0 if away team wins
            outcome = np.random.randint(0, 2)
            
            games.append({
                'home_team': home_team,
                'away_team': away_team,
                'outcome': outcome
            })
        
        games_df = pd.DataFrame(games)
        
        # Join with team stats
        games_with_stats = games_df.copy()
        
        # Add home team stats
        home_team_stats = games_df.merge(
            team_stats, 
            left_on='home_team', 
            right_on='Team'
        ).drop(['Team', 'home_team', 'away_team', 'outcome'], axis=1)
        
        # Add away team stats
        away_team_stats = games_df.merge(
            team_stats, 
            left_on='away_team', 
            right_on='Team'
        ).drop(['Team', 'home_team', 'away_team', 'outcome'], axis=1)
        
        # Create feature columns
        feature_cols = []
        for col in team_stats.columns:
            if col != 'Team':
                home_team_stats = home_team_stats.rename(columns={col: f'home_{col}'})
                away_team_stats = away_team_stats.rename(columns={col: f'away_{col}'})
                feature_cols.append(f'home_{col}')
                feature_cols.append(f'away_{col}')
        
        # Combine all features
        X = pd.concat([home_team_stats.reset_index(drop=True), 
                      away_team_stats.reset_index(drop=True)], axis=1)
        y = games_df['outcome']
        
        self.features = feature_cols
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self):
        """Train the prediction model"""
        logger.info("Training MLB prediction model...")
        
        if not self.load_data():
            return False
        
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Train a Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Accuracy: {accuracy:.2f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("\nTop 10 Most Important Features:")
        for i in range(min(10, len(self.features))):
            logger.info(f"{self.features[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return True
    
    def make_predictions(self, home_team, away_team):
        """Make a prediction for a specific matchup"""
        if self.model is None:
            logger.error("Model not trained. Please train the model first.")
            return None
        
        logger.info(f"Predicting outcome for {home_team} vs {away_team}...")
        
        # [Implement logic to prepare features for the specific matchup]
        # This would require preprocessing similar to the training data
        
        # For now, return a dummy prediction
        return {
            'home_win_probability': 0.65,
            'prediction': 'Home team win'
        }

if __name__ == "__main__":
    logger.info("Initializing MLB prediction framework...")
    
    model = MLBPredictionModel()
    model.train_model()
    
    # Example prediction
    prediction = model.make_predictions('NYY', 'BOS')
    if prediction:
        logger.info(f"Prediction: {prediction['prediction']}")
        logger.info(f"Home team win probability: {prediction['home_win_probability']:.2f}")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLBBettingModel:
    """
    A class to build, train and evaluate MLB betting models with contextual features
    """
    def __init__(self, data_dir=None, model_dir=None):
        """
        Initialize the MLB betting model
        
        Args:
            data_dir (str, optional): Directory containing the game data
            model_dir (str, optional): Directory to save trained models
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, 'data', 'contextual')
        else:
            self.data_dir = data_dir
            
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, 'models', 'betting')
        else:
            self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model containers
        self.models = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print("MLB Betting Model initialized")
    
    def load_data(self, filename=None):
        """
        Load game data with contextual features
        
        Args:
            filename (str, optional): Name of the data file
            
        Returns:
            DataFrame: Loaded game data
        """
        if filename is None:
            # Try to find the most recent data file
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.startswith('games_with_engineered_features') and f.endswith('.csv')]
            
            if data_files:
                # Sort by creation time (newest first)
                data_files.sort(key=lambda x: os.path.getmtime(
                    os.path.join(self.data_dir, x)), reverse=True)
                
                filename = data_files[0]
            else:
                raise FileNotFoundError("No game data files found")
        
        data_path = os.path.join(self.data_dir, filename)
        print(f"Loading data from {data_path}")
        
        # Load the data
        data = pd.read_csv(data_path)
        
        # Check if we have game outcome data (for training)
        # If not, we'll need to simulate it for demonstration
        if 'home_team_runs' not in data.columns or 'away_team_runs' not in data.columns:
            print("Outcome data not found. Simulating outcomes for demonstration.")
            # Simulate game outcomes based on contextual factors
            data = self._simulate_outcomes(data)
        
        return data
    
    def _simulate_outcomes(self, data):
        """
        Simulate game outcomes based on contextual factors for demonstration
        
        Args:
            data (DataFrame): Game data without outcomes
            
        Returns:
            DataFrame: Game data with simulated outcomes
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Average MLB game has about 4.5 runs per team
        base_runs = 4.5
        
        # Simulate home team runs
        data['home_team_runs'] = data.apply(
            lambda row: max(0, int(np.random.poisson(
                base_runs * (1 + 0.3 * row.get('home_advantage_score', 0) + 
                           0.2 * row.get('total_runs_context_factor', 0))))),
            axis=1
        )
        
        # Simulate away team runs
        data['away_team_runs'] = data.apply(
            lambda row: max(0, int(np.random.poisson(
                base_runs * (1 + 0.3 * row.get('away_advantage_score', 0) + 
                           0.2 * row.get('total_runs_context_factor', 0))))),
            axis=1
        )
        
        # Calculate total runs
        data['total_runs'] = data['home_team_runs'] + data['away_team_runs']
        
        # Calculate run differential
        data['run_differential'] = data['home_team_runs'] - data['away_team_runs']
        
        # Determine game winner
        data['home_team_win'] = (data['run_differential'] > 0).astype(int)
        
        # Simulate strikeouts (average ~8 per team per game)
        base_strikeouts = 8
        
        data['home_pitcher_strikeouts'] = data.apply(
            lambda row: max(0, int(np.random.poisson(
                base_strikeouts * (1 + 0.2 * row.get('total_strikeouts_context_factor', 0) + 
                                 0.3 * row.get('home_pitcher_context_advantage', 0))))),
            axis=1
        )
        
        data['away_pitcher_strikeouts'] = data.apply(
            lambda row: max(0, int(np.random.poisson(
                base_strikeouts * (1 + 0.2 * row.get('total_strikeouts_context_factor', 0) + 
                                 0.3 * row.get('away_pitcher_context_advantage', 0))))),
            axis=1
        )
        
        # Total strikeouts
        data['total_strikeouts'] = data['home_pitcher_strikeouts'] + data['away_pitcher_strikeouts']
        
        # Simulate betting lines
        # Assume a standard -110 line for both sides of total (implied probability ~52.4%)
        data['total_line'] = data['total_runs'].apply(
            lambda x: x + np.random.uniform(-1, 1)  # Add some noise
        ).round(0) + 0.5  # Always use x.5 for totals
        
        data['home_team_odds'] = data.apply(
            lambda row: -110 if row['home_team_win'] == 1 else 100,
            axis=1
        )
        
        data['away_team_odds'] = data.apply(
            lambda row: -110 if row['home_team_win'] == 0 else 100,
            axis=1
        )
        
        # Over/under results
        data['over'] = (data['total_runs'] > data['total_line']).astype(int)
        
        return data

    def _preprocess_data(self, X):
        """
        Preprocess data to handle missing values
        
        Args:
            X (DataFrame): Features to preprocess
            
        Returns:
            DataFrame: Processed features with missing values filled
        """
        # Make a copy to avoid modifying the original
        X_processed = X.copy()
        
        # For each column, fill NaN values with appropriate values
        for col in X_processed.columns:
            # If the column has NaN values
            if X_processed[col].isna().any():
                # Fill NaNs with the column median for numeric features
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    # Get median excluding NaNs
                    median_val = X_processed[col].median()
                    # Fill NaNs with median
                    X_processed[col] = X_processed[col].fillna(median_val)
                    print(f"Warning: {col} contains NaN values. Filling with median value: {median_val:.4f}")
                else:
                    # For categorical features, fill with the most frequent value
                    mode_val = X_processed[col].mode()[0]
                    X_processed[col] = X_processed[col].fillna(mode_val)
                    print(f"Warning: {col} contains NaN values. Filling with mode value: {mode_val}")
        
        return X_processed
    
    def prepare_features(self, data, target_type='totals', include_contextual=True):
        """
        Prepare features for model training
        
        Args:
            data (DataFrame): Game data
            target_type (str): Type of prediction ('totals', 'spread', 'moneyline', 'strikeouts')
            include_contextual (bool): Whether to include contextual features
            
        Returns:
            tuple: X (features), y (target)
        """
        # Define basic features (non-contextual)
        basic_features = ['home_team', 'away_team']
        
        # Convert team columns to categorical
        data_encoded = data.copy()
        for col in ['home_team', 'away_team']:
            data_encoded[col] = data_encoded[col].astype('category').cat.codes
        
        # Define contextual features based on what's available
        contextual_features = []
        
        # Add available contextual features
        if include_contextual:
            # Ballpark features
            if 'ballpark_run_factor' in data.columns:
                contextual_features.extend(['ballpark_run_factor', 'ballpark_hr_factor'])
            
            # Weather features
            weather_features = ['temp_factor', 'wind_factor', 'humidity_factor', 
                               'precipitation_factor', 'weather_score']
            contextual_features.extend([f for f in weather_features if f in data.columns])
            
            # Umpire features
            umpire_features = ['umpire_strikeout_boost', 'umpire_runs_boost', 
                              'umpire_consistency_factor']
            contextual_features.extend([f for f in umpire_features if f in data.columns])
            
            # Team advantage features
            team_features = ['home_power_context_advantage', 'away_power_context_advantage',
                            'home_contact_context_advantage', 'away_contact_context_advantage']
            contextual_features.extend([f for f in team_features if f in data.columns])
            
            # Pitcher features
            pitcher_features = ['home_pitcher_context_advantage', 'away_pitcher_context_advantage', 
                               'pitcher_matchup_strikeout_boost']
            contextual_features.extend([f for f in pitcher_features if f in data.columns])
            
            # Combined features
            combined_features = ['total_runs_context_factor', 'total_strikeouts_context_factor',
                                'home_advantage_score', 'away_advantage_score']
            contextual_features.extend([f for f in combined_features if f in data.columns])
        
        # Combine features
        features = basic_features + contextual_features
        
        # Select target based on betting type
        if target_type == 'totals':
            # Predict total runs
            y = data['total_runs']
            target_name = 'total_runs'
        elif target_type == 'spread':
            # Predict run differential
            y = data['run_differential']
            target_name = 'run_differential'
        elif target_type == 'moneyline':
            # Predict home team win
            y = data['home_team_win']
            target_name = 'home_team_win'
        elif target_type == 'strikeouts':
            # Predict total strikeouts
            y = data['total_strikeouts']
            target_name = 'total_strikeouts'
        else:
            raise ValueError(f"Invalid target_type: {target_type}")
        
        # Select features
        X = data_encoded[features]
        
        print(f"Prepared features for {target_type} model:")
        print(f"- Basic features: {basic_features}")
        print(f"- Contextual features: {contextual_features if include_contextual else 'None'}")
        print(f"- Target: {target_name}")
        
        return X, y, features, target_name
    
    def train_model(self, data, target_type='totals', include_contextual=True, model_type='gbm'):
        """
        Train a betting model
        
        Args:
            data (DataFrame): Game data
            target_type (str): Type of prediction
            include_contextual (bool): Whether to include contextual features
            model_type (str): Type of model ('rf', 'gbm', 'ridge')
            
        Returns:
            tuple: Trained model, features, cross-validation scores
        """
        # Prepare features
        X, y, features, target_name = self.prepare_features(
            data, target_type, include_contextual)
            
        # Preprocess the features to handle missing values
        X_processed = self._preprocess_data(X)
        
        # Choose model type
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = 'Random Forest'
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_name = 'Gradient Boosting'
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
            model_name = 'Ridge Regression'
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # Train on full dataset
        pipeline.fit(X_processed, y)
        
        # Save model info
        model_id = f"{target_type}_{'context' if include_contextual else 'basic'}_{model_type}"
        self.models[model_id] = {
            'pipeline': pipeline,
            'features': features,
            'target': target_name,
            'cv_rmse': rmse_scores.mean(),
            'include_contextual': include_contextual,
            'model_type': model_type,
            'target_type': target_type
        }
        
        print(f"\nTrained {model_name} for {target_type} prediction")
        print(f"Cross-validated RMSE: {rmse_scores.mean():.4f}")
        
        # Save model to disk
        model_path = os.path.join(self.model_dir, f"{model_id}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")
        
        return pipeline, features, rmse_scores
    
    def evaluate_model(self, data, model_id=None, test_size=0.2):
        """
        Evaluate model performance on test set
        
        Args:
            data (DataFrame): Game data
            model_id (str, optional): ID of model to evaluate
            test_size (float): Size of test set
            
        Returns:
            dict: Evaluation metrics
        """
        if model_id is None:
            # Use the most recently trained model
            if not self.models:
                raise ValueError("No models have been trained yet")
            model_id = list(self.models.keys())[-1]
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        pipeline = model_info['pipeline']
        features = model_info['features']
        target = model_info['target']
        
        # Convert team columns to categorical
        data_encoded = data.copy()
        for col in ['home_team', 'away_team']:
            if col in data_encoded.columns:
                data_encoded[col] = data_encoded[col].astype('category').cat.codes
        
        # Prepare data
        X = data_encoded[features]
        y = data_encoded[target]
        
        # Preprocess features
        X_processed = self._preprocess_data(X)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42)
        
        # Train on training set
        pipeline.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"\nEvaluation of {model_id} on test set:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted {target} - {model_id}')
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f"{model_id}_evaluation.png")
        plt.savefig(plot_path)
        plt.close()
        
        # For totals model, evaluate betting performance
        if model_info['target_type'] == 'totals' and 'total_line' in data.columns:
            betting_metrics = self._evaluate_betting_performance(
                y_test, y_pred, data_encoded.loc[X_test.index, 'total_line'],
                data_encoded.loc[X_test.index, 'over'])
            
            metrics.update(betting_metrics)
            
            print("\nBetting Performance:")
            print(f"Accuracy: {betting_metrics['betting_accuracy']:.4f}")
            print(f"ROI: {betting_metrics['betting_roi']:.4f}")
            print(f"Kelly Growth: {betting_metrics['kelly_growth']:.4f}")
        
        return metrics
    
    def _evaluate_betting_performance(self, y_true, y_pred, lines, actual_results):
        """
        Evaluate performance specifically for betting
        
        Args:
            y_true (Series): Actual outcomes
            y_pred (array): Predicted outcomes
            lines (Series): Betting lines
            actual_results (Series): Actual over/under results (1 for over, 0 for under)
            
        Returns:
            dict: Betting performance metrics
        """
        # Determine model's bet decisions
        predictions = []
        
        for pred, line in zip(y_pred, lines):
            if pred > line:
                predictions.append(1)  # Over
            else:
                predictions.append(0)  # Under
        
        predictions = np.array(predictions)
        
        # Calculate betting accuracy
        correct_bets = (predictions == actual_results).sum()
        total_bets = len(predictions)
        accuracy = correct_bets / total_bets
        
        # Calculate ROI (assuming -110 odds, i.e., bet 110 to win 100)
        stake_per_bet = 110
        win_amount = 100
        
        total_staked = total_bets * stake_per_bet
        total_return = correct_bets * (stake_per_bet + win_amount)
        roi = (total_return - total_staked) / total_staked
        
        # Calculate Kelly growth
        # For simplicity, assuming flat 0.524 implied probability for all bets
        p_win = accuracy
        q_lose = 1 - p_win
        b_odds = 0.909  # Decimal odds for -110 (1.909)
        
        kelly_fraction = (p_win * b_odds - q_lose) / b_odds if p_win * b_odds > q_lose else 0
        kelly_growth = (1 + kelly_fraction * 0.91) ** total_bets
        
        return {
            'betting_accuracy': accuracy,
            'betting_roi': roi,
            'kelly_growth': kelly_growth,
            'correct_bets': correct_bets,
            'total_bets': total_bets
        }
    
    def compare_models(self, data, target_types=None, model_types=None):
        """
        Compare models with and without contextual features
        
        Args:
            data (DataFrame): Game data
            target_types (list, optional): Types of predictions to compare
            model_types (list, optional): Types of models to compare
        """
        if target_types is None:
            target_types = ['totals', 'moneyline', 'strikeouts']
        
        if model_types is None:
            model_types = ['gbm']
        
        results = []
        
        for target_type in target_types:
            for model_type in model_types:
                # Train model without contextual features
                print(f"\n--- Training {target_type} model without contextual features ---")
                _, _, basic_scores = self.train_model(
                    data, target_type, include_contextual=False, model_type=model_type)
                
                # Train model with contextual features
                print(f"\n--- Training {target_type} model with contextual features ---")
                _, _, context_scores = self.train_model(
                    data, target_type, include_contextual=True, model_type=model_type)
                
                # Calculate improvement
                basic_rmse = basic_scores.mean()
                context_rmse = context_scores.mean()
                improvement = (basic_rmse - context_rmse) / basic_rmse * 100
                
                results.append({
                    'target_type': target_type,
                    'model_type': model_type,
                    'basic_rmse': basic_rmse,
                    'context_rmse': context_rmse,
                    'improvement': improvement
                })
                
                print(f"\nImprovement with contextual features: {improvement:.2f}%")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot improvements
        plt.figure(figsize=(12, 8))
        sns.barplot(x='target_type', y='improvement', hue='model_type', data=results_df)
        plt.title('Improvement in Prediction Accuracy with Contextual Features')
        plt.xlabel('Prediction Target')
        plt.ylabel('Improvement (%)')
        
        # Save plot
        plot_path = os.path.join(self.model_dir, "contextual_improvement.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nComparison results saved to {plot_path}")
        
        return results_df
    
    def feature_importance(self, model_id=None):
        """
        Analyze feature importance for a given model
        
        Args:
            model_id (str, optional): ID of model to analyze
            
        Returns:
            DataFrame: Feature importance scores
        """
        if model_id is None:
            # Use the most recently trained model
            if not self.models:
                raise ValueError("No models have been trained yet")
            model_id = list(self.models.keys())[-1]
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        pipeline = model_info['pipeline']
        features = model_info['features']
        
        # Extract the model from the pipeline
        model = pipeline.named_steps['model']
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not provide feature importances")
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Feature Importance - {model_id}')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f"{model_id}_feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nFeature importance analysis for {model_id}:")
        print(importance_df.head(10))
        print(f"Feature importance plot saved to {plot_path}")
        
        return importance_df
    
    def make_predictions(self, new_games, model_id=None):
        """
        Make predictions for new games
        
        Args:
            new_games (DataFrame): New game data
            model_id (str, optional): ID of model to use
            
        Returns:
            DataFrame: Game data with predictions
        """
        if model_id is None:
            # Use the most recently trained model
            if not self.models:
                raise ValueError("No models have been trained yet")
            model_id = list(self.models.keys())[-1]
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        pipeline = model_info['pipeline']
        features = model_info['features']
        target = model_info['target']
        target_type = model_info['target_type']
        
        # Convert team columns to categorical
        games_encoded = new_games.copy()
        for col in ['home_team', 'away_team']:
            if col in games_encoded.columns:
                games_encoded[col] = games_encoded[col].astype('category').cat.codes
        
        # Prepare data
        X = games_encoded[features]
        
        # Preprocess features
        X_processed = self._preprocess_data(X)
        
        # Make predictions
        predictions = pipeline.predict(X_processed)
        
        # Add predictions to the data
        new_games[f'predicted_{target}'] = predictions
        
        # For totals, add bet recommendations
        if target_type == 'totals' and 'total_line' in new_games.columns:
            new_games['bet_recommendation'] = new_games.apply(
                lambda row: 'OVER' if row[f'predicted_{target}'] > row['total_line'] 
                          else 'UNDER',
                axis=1
            )
            
            # Calculate edge
            new_games['edge'] = new_games.apply(
                lambda row: abs(row[f'predicted_{target}'] - row['total_line']),
                axis=1
            )
        
        # For moneyline, add probabilities and recommendations
        elif target_type == 'moneyline':
            # Convert predictions to probabilities (0-1)
            predictions_bounded = np.clip(predictions, 0, 1)
            
            # Add predicted win probabilities
            new_games['home_win_probability'] = predictions_bounded
            new_games['away_win_probability'] = 1 - predictions_bounded
            
            # Calculate implied probabilities from odds (if available)
            if 'home_team_odds' in new_games.columns and 'away_team_odds' in new_games.columns:
                new_games['home_implied_prob'] = new_games.apply(
                    lambda row: self._odds_to_probability(row['home_team_odds']),
                    axis=1
                )
                
                new_games['away_implied_prob'] = new_games.apply(
                    lambda row: self._odds_to_probability(row['away_team_odds']),
                    axis=1
                )
                
                # Calculate edge
                new_games['home_edge'] = new_games['home_win_probability'] - new_games['home_implied_prob']
                new_games['away_edge'] = new_games['away_win_probability'] - new_games['away_implied_prob']
                
                # Add bet recommendations
                new_games['bet_recommendation'] = new_games.apply(
                    lambda row: 'HOME' if row['home_edge'] > 0.05 
                              else 'AWAY' if row['away_edge'] > 0.05 
                              else 'PASS',
                    axis=1
                )
        
        print(f"\nMade predictions for {len(new_games)} games using {model_id}")
        
        return new_games
    
    def _odds_to_probability(self, odds):
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
            
    def load_player_data(self):
        """
        Load player data for props predictions
        
        Returns:
            tuple: (player_projections, batter_statcast, pitcher_statcast)
        """
        import os
        import pandas as pd
        print("Loading player data for props predictions...")
        
        # Load player projections
        projection_file = os.path.join(self.data_dir, '..', 'predictions', 'player_projections_2025.csv')
        player_projections = None
        if os.path.exists(projection_file):
            player_projections = pd.read_csv(projection_file)
            print(f"Loaded player projections from {projection_file}")
        else:
            print("No player projections file found.")
        
        # Load Statcast data
        statcast_dir = os.path.join(self.data_dir, '..', 'processed', 'statcast')
        
        batter_statcast = None
        pitcher_statcast = None
        
        batter_file = os.path.join(statcast_dir, 'processed_batter_all_time.csv')
        if os.path.exists(batter_file):
            batter_statcast = pd.read_csv(batter_file)
            print(f"Loaded batter Statcast data from {batter_file}")
        
        pitcher_file = os.path.join(statcast_dir, 'processed_pitcher_all_time.csv')
        if os.path.exists(pitcher_file):
            pitcher_statcast = pd.read_csv(pitcher_file)
            print(f"Loaded pitcher Statcast data from {pitcher_file}")
        
        return player_projections, batter_statcast, pitcher_statcast

    def predict_player_props(self, games_df=None, player_projections=None, batter_statcast=None, pitcher_statcast=None, prop_type=None):
        """
        Predict player props for upcoming games
        
        Args:
            games_df (DataFrame, optional): Game data
            player_projections (DataFrame, optional): Player projections
            batter_statcast (DataFrame, optional): Batter Statcast data
            pitcher_statcast (DataFrame, optional): Pitcher Statcast data
            prop_type (str, optional): Type of prop ('hits', 'hr', 'strikeouts')
            
        Returns:
            DataFrame: Player prop predictions
        """
        from scipy import stats
        import numpy as np
        import os
        import pandas as pd
        
        print("Predicting player props...")
        
        # Load player name mapping if available
        player_map = {}
        mapping_file = os.path.join(self.data_dir, '..', 'predictions', 'improved_player_id_map.csv')
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            player_map = dict(zip(mapping_df['player_id'].astype(str), mapping_df['player_name']))
        
        # If no games provided, create sample games
        if games_df is None:
            games_df = pd.DataFrame({
                'game_id': [1001, 1002, 1003],
                'game_date': ['2025-04-01', '2025-04-01', '2025-04-02'],
                'home_team': ['NYY', 'LAD', 'BOS'],
                'away_team': ['BOS', 'SF', 'NYY'],
                'home_team_id': [147, 119, 111],
                'away_team_id': [111, 137, 147]
            })
        
        # If no player projections, attempt to load from file or create sample
        if player_projections is None:
            projection_file = os.path.join(self.data_dir, '..', 'predictions', 'player_projections_2025.csv')
            if os.path.exists(projection_file):
                player_projections = pd.read_csv(projection_file)
                print(f"Loaded player projections from {projection_file}")
            else:
                print("No player projections file found. Creating sample data.")
                player_projections = pd.DataFrame({
                    'player_id': range(10001, 10021),
                    'player_name': [f"Player {i}" for i in range(1, 21)],
                    'hit_rate': np.random.uniform(0.3, 0.6, 20),
                    'hr_rate': np.random.uniform(0.01, 0.1, 20),
                    'projected_hit_rate_2025': np.random.uniform(0.3, 0.6, 20),
                    'projected_hr_rate_2025': np.random.uniform(0.01, 0.1, 20)
                })
        
        # If no Statcast data provided, try to load or use None
        if batter_statcast is None:
            statcast_dir = os.path.join(self.data_dir, '..', 'processed', 'statcast')
            batter_file = os.path.join(statcast_dir, 'processed_batter_all_time.csv')
            if os.path.exists(batter_file):
                batter_statcast = pd.read_csv(batter_file)
                print(f"Loaded batter Statcast data from {batter_file}")
        if pitcher_statcast is None:
            statcast_dir = os.path.join(self.data_dir, '..', 'processed', 'statcast')
            pitcher_file = os.path.join(statcast_dir, 'processed_pitcher_all_time.csv')
            if os.path.exists(pitcher_file):
                pitcher_statcast = pd.read_csv(pitcher_file)
                print(f"Loaded pitcher Statcast data from {pitcher_file}")
        
        prop_predictions = []
        if prop_type:
            print(f"Generating {prop_type} prop predictions")
        else:
            print("Generating all available prop predictions")
        prop_mappings = {
            'hits': 'projected_hit_rate_2025',
            'hr': 'projected_hr_rate_2025',
            'strikeouts': 'projected_strikeout_rate_2025'
        }
        for _, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            ballpark_run_factor = game.get('ballpark_run_factor', 0)
            ballpark_hr_factor = game.get('ballpark_hr_factor', 0)
            weather_score = game.get('weather_score', 0)
            home_pitcher_id = game.get('home_starting_pitcher_id') or game.get('home_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id') or game.get('away_pitcher_id')
            # Dedicated pitcher strikeouts prediction
            if prop_type == 'strikeouts' or prop_type is None:
                strikeout_props = self.predict_pitcher_strikeouts(
                    game_id, home_team, away_team, home_pitcher_id, away_pitcher_id, pitcher_statcast
                )
                if strikeout_props:
                    prop_predictions.extend(strikeout_props)
                if prop_type == 'strikeouts':
                    continue
            num_players = len(player_projections)
            home_indices = np.random.choice(num_players, 9, replace=False)
            away_indices = np.random.choice(num_players, 9, replace=False)
            for idx in home_indices:
                player = player_projections.iloc[idx]
                player_id = player['player_id']
                player_name = player_map.get(str(player_id), f"Player {player_id}")
                for prop in ['hits', 'hr']:
                    if prop_type and prop != prop_type:
                        continue
                    projection_field = prop_mappings.get(prop)
                    if projection_field and projection_field in player:
                        base_rate = player[projection_field]
                    else:
                        base_rate = player.get('hit_rate', 0.3) if prop == 'hits' else player.get('hr_rate', 0.05)
                    adjusted_rate = base_rate if not np.isnan(base_rate) else 0.25
                    if prop == 'hits':
                        ballpark_factor = ballpark_run_factor if not np.isnan(ballpark_run_factor) else 0
                        adjusted_rate *= (1 + ballpark_factor * 0.2)
                    elif prop == 'hr':
                        ballpark_factor = ballpark_hr_factor if not np.isnan(ballpark_hr_factor) else 0
                        adjusted_rate *= (1 + ballpark_factor * 0.3)
                    weather_factor = weather_score if not np.isnan(weather_score) else 0
                    adjusted_rate *= (1 + weather_factor * 0.1)
                    adjusted_rate = max(0.01, adjusted_rate) if not np.isnan(adjusted_rate) else 0.25
                    at_bats = 4
                    expected_value = round(adjusted_rate * at_bats, 2)
                    if np.isnan(expected_value):
                        expected_value = 0.5
                    if prop == 'hits':
                        line = round(expected_value * 0.8 + 0.5)
                    else:
                        line = 0.5 if expected_value > 0.3 else 0.0
                    prop_predictions.append({
                        'game_id': game_id,
                        'team': home_team,
                        'player_id': player_id,
                        'player_name': player_name,
                        'is_home': True,
                        'prop_type': prop,
                        'base_rate': base_rate,
                        'adjusted_rate': adjusted_rate,
                        'expected_value': expected_value,
                        'line': line,
                        'over_prob': 1 - stats.poisson.cdf(line, expected_value)
                    })
            for idx in away_indices:
                player = player_projections.iloc[idx]
                player_id = player['player_id']
                player_name = player_map.get(str(player_id), f"Player {player_id}")
                for prop in ['hits', 'hr']:
                    if prop_type and prop != prop_type:
                        continue
                    projection_field = prop_mappings.get(prop)
                    if projection_field and projection_field in player:
                        base_rate = player[projection_field]
                    else:
                        base_rate = player.get('hit_rate', 0.3) if prop == 'hits' else player.get('hr_rate', 0.05)
                    adjusted_rate = base_rate if not np.isnan(base_rate) else 0.25
                    if prop == 'hits':
                        ballpark_factor = ballpark_run_factor if not np.isnan(ballpark_run_factor) else 0
                        adjusted_rate *= (1 + ballpark_factor * 0.15)
                    elif prop == 'hr':
                        ballpark_factor = ballpark_hr_factor if not np.isnan(ballpark_hr_factor) else 0
                        adjusted_rate *= (1 + ballpark_factor * 0.25)
                    weather_factor = weather_score if not np.isnan(weather_score) else 0
                    adjusted_rate *= (1 + weather_factor * 0.1)
                    adjusted_rate = max(0.01, adjusted_rate) if not np.isnan(adjusted_rate) else 0.25
                    at_bats = 4
                    expected_value = round(adjusted_rate * at_bats, 2)
                    if np.isnan(expected_value):
                        expected_value = 0.5
                    if prop == 'hits':
                        line = round(expected_value * 0.8 + 0.5)
                    else:
                        line = 0.5 if expected_value > 0.3 else 0.0
                    prop_predictions.append({
                        'game_id': game_id,
                        'team': away_team,
                        'player_id': player_id,
                        'player_name': player_name,
                        'is_home': False,
                        'prop_type': prop,
                        'base_rate': base_rate,
                        'adjusted_rate': adjusted_rate,
                        'expected_value': expected_value,
                        'line': line,
                        'over_prob': 1 - stats.poisson.cdf(line, expected_value)
                    })
        props_df = pd.DataFrame(prop_predictions)
        if 'over_prob' in props_df.columns:
            props_df['edge'] = (props_df['over_prob'] - 0.5) * 2
        if len(props_df) > 0:
            prediction_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(prediction_dir, exist_ok=True)
            output_file = os.path.join(
                prediction_dir, 
                f"player_props_{prop_type}_predictions.csv" if prop_type else "player_props_predictions.csv"
            )
            props_df.to_csv(output_file, index=False)
            print(f"Saved player props predictions to {output_file}")
        return props_df

    def predict_pitcher_strikeouts(self, game_id, home_team, away_team, home_pitcher_id, away_pitcher_id, pitcher_statcast=None):
        """
        Predict strikeout props for pitchers
        
        Args:
            game_id: Game ID
            home_team: Home team code
            away_team: Away team code
            home_pitcher_id: Home starting pitcher ID
            away_pitcher_id: Away starting pitcher ID
            pitcher_statcast: Pitcher Statcast data (optional)
            
        Returns:
            list: Strikeout prop predictions
        """
        from scipy.stats import norm
        import numpy as np
        import os
        import pandas as pd
        # Get pitcher names if available
        player_map = {}
        mapping_file = os.path.join(self.data_dir, '..', 'predictions', 'improved_player_id_map.csv')
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            player_map = dict(zip(mapping_df['player_id'].astype(str), mapping_df['player_name']))
        home_pitcher_name = player_map.get(str(home_pitcher_id), "Home Pitcher")
        away_pitcher_name = player_map.get(str(away_pitcher_id), "Away Pitcher")
        home_k_rate = None
        away_k_rate = None
        if pitcher_statcast is not None:
            if home_pitcher_id:
                home_pitcher = pitcher_statcast[pitcher_statcast['player_id'] == str(home_pitcher_id)]
                if not home_pitcher.empty:
                    if 'whiff_rate' in home_pitcher.columns:
                        home_k_rate = home_pitcher['whiff_rate'].iloc[0] * 25
            if away_pitcher_id:
                away_pitcher = pitcher_statcast[pitcher_statcast['player_id'] == str(away_pitcher_id)]
                if not away_pitcher.empty:
                    if 'whiff_rate' in away_pitcher.columns:
                        away_k_rate = away_pitcher['whiff_rate'].iloc[0] * 25
        if home_k_rate is None or np.isnan(home_k_rate):
            home_k_rate = 6.0
        if away_k_rate is None or np.isnan(away_k_rate):
            away_k_rate = 6.0
        predictions = []
        home_expected = round(home_k_rate, 1)
        home_line = round(home_expected)
        home_over_prob = 1 - norm.cdf(home_line, home_expected, 2.0)
        predictions.append({
            'game_id': game_id,
            'team': home_team,
            'player_id': str(home_pitcher_id) if home_pitcher_id else 'unknown',
            'player_name': home_pitcher_name,
            'is_home': True,
            'prop_type': 'strikeouts',
            'expected_value': home_expected,
            'line': home_line,
            'over_prob': home_over_prob,
            'edge': (home_over_prob - 0.5) * 2
        })
        away_expected = round(away_k_rate, 1)
        away_line = round(away_expected)
        away_over_prob = 1 - norm.cdf(away_line, away_expected, 2.0)
        predictions.append({
            'game_id': game_id,
            'team': away_team,
            'player_id': str(away_pitcher_id) if away_pitcher_id else 'unknown',
            'player_name': away_pitcher_name,
            'is_home': False,
            'prop_type': 'strikeouts',
            'expected_value': away_expected,
            'line': away_line,
            'over_prob': away_over_prob,
            'edge': (away_over_prob - 0.5) * 2
        })
        return predictions
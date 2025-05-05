import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

class MLBPlayerPropsModel:
    """
    A dedicated model for MLB player props predictions
    """
    def __init__(self, data_dir=None):
        """
        Initialize the player props model
        
        Args:
            data_dir (str, optional): Directory containing the data
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, 'data', 'contextual')
        else:
            self.data_dir = data_dir
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize data containers
        self.player_projections = None
        self.batter_statcast = None
        self.pitcher_statcast = None
        self.game_data = None
        self.player_map = self._load_player_mapping()
        
        self.logger.info("MLB Player Props Model initialized")
    
    def _load_player_mapping(self):
        """Load player ID to name mapping"""
        mapping_file = os.path.join(self.data_dir, '..', 'predictions', 'improved_player_id_map.csv')
        
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            player_map = dict(zip(mapping_df['player_id'].astype(str), mapping_df['player_name']))
            self.logger.info(f"Loaded {len(player_map)} player ID mappings")
            return player_map
        else:
            self.logger.warning("No player mapping file found")
            return {}
    
    def load_batter_splits(self):
        """Load batter splits (vs LHP/RHP, home/away) and merge into player projections."""
        splits_file = os.path.join(self.data_dir, '..', 'processed', 'batter_splits.csv')
        if os.path.exists(splits_file):
            splits = pd.read_csv(splits_file)
            # Example: merge vs. LHP splits
            splits_lhp = splits[splits['vs_hand'] == 'L'].rename(
                columns={'avg': 'avg_vs_lhp', 'obp': 'obp_vs_lhp', 'slg': 'slg_vs_lhp'}
            )
            splits_rhp = splits[splits['vs_hand'] == 'R'].rename(
                columns={'avg': 'avg_vs_rhp', 'obp': 'obp_vs_rhp', 'slg': 'slg_vs_rhp'}
            )
            self.player_projections = pd.merge(
                self.player_projections, splits_lhp[['player_id', 'avg_vs_lhp', 'obp_vs_lhp', 'slg_vs_lhp']],
                on='player_id', how='left'
            )
            self.player_projections = pd.merge(
                self.player_projections, splits_rhp[['player_id', 'avg_vs_rhp', 'obp_vs_rhp', 'slg_vs_rhp']],
                on='player_id', how='left'
            )
            self.logger.info("Merged batter splits into player projections")
        else:
            self.logger.warning("No batter splits file found")

    def add_recent_form(self, days=7):
        """Add recent form stats (last N days) to player projections."""
        stats_file = os.path.join(self.data_dir, '..', 'historical', 'player_game_stats.csv')
        if not os.path.exists(stats_file):
            self.logger.warning("No player game stats file found for recent form")
            return
        stats = pd.read_csv(stats_file)
        stats['game_date'] = pd.to_datetime(stats['game_date'])
        cutoff = stats['game_date'].max() - pd.Timedelta(days=days)
        recent = stats[stats['game_date'] >= cutoff]
        recent_form = recent.groupby('player_id').agg({
            'hits': 'sum',
            'at_bats': 'sum',
            'home_runs': 'sum'
        }).reset_index()
        recent_form['recent_avg'] = recent_form['hits'] / recent_form['at_bats']
        recent_form['recent_hr_rate'] = recent_form['home_runs'] / recent_form['at_bats']
        self.player_projections = pd.merge(
            self.player_projections, recent_form[['player_id', 'recent_avg', 'recent_hr_rate']],
            on='player_id', how='left'
        )
        self.logger.info(f"Added recent form (last {days} days) to player projections")

    def load_batter_advanced_stats(self):
        """Load advanced plate discipline and batted ball stats."""
        adv_file = os.path.join(self.data_dir, '..', 'processed', 'batter_advanced_stats.csv')
        if os.path.exists(adv_file):
            adv = pd.read_csv(adv_file)
            self.player_projections = pd.merge(
                self.player_projections, adv, on='player_id', how='left'
            )
            self.logger.info("Merged advanced batter stats into player projections")
        else:
            self.logger.warning("No advanced batter stats file found")

    def load_pitcher_splits(self):
        """Load pitcher splits (vs LHB/RHB, home/away) and merge into pitcher Statcast."""
        splits_file = os.path.join(self.data_dir, '..', 'processed', 'pitcher_splits.csv')
        if os.path.exists(splits_file):
            splits = pd.read_csv(splits_file)
            splits_lhb = splits[splits['vs_hand'] == 'L'].rename(
                columns={'era': 'era_vs_lhb', 'whip': 'whip_vs_lhb', 'k9': 'k9_vs_lhb'}
            )
            splits_rhb = splits[splits['vs_hand'] == 'R'].rename(
                columns={'era': 'era_vs_rhb', 'whip': 'whip_vs_rhb', 'k9': 'k9_vs_rhb'}
            )
            self.pitcher_statcast = pd.merge(
                self.pitcher_statcast, splits_lhb[['player_id', 'era_vs_lhb', 'whip_vs_lhb', 'k9_vs_lhb']],
                on='player_id', how='left'
            )
            self.pitcher_statcast = pd.merge(
                self.pitcher_statcast, splits_rhb[['player_id', 'era_vs_rhb', 'whip_vs_rhb', 'k9_vs_rhb']],
                on='player_id', how='left'
            )
            self.logger.info("Merged pitcher splits into pitcher Statcast")
        else:
            self.logger.warning("No pitcher splits file found")

    def add_pitcher_recent_form(self, starts=5):
        """Add recent form stats (last N starts) to pitcher Statcast."""
        stats_file = os.path.join(self.data_dir, '..', 'historical', 'pitcher_game_stats.csv')
        if not os.path.exists(stats_file):
            self.logger.warning("No pitcher game stats file found for recent form")
            return
        stats = pd.read_csv(stats_file)
        stats['game_date'] = pd.to_datetime(stats['game_date'])
        stats = stats.sort_values(['player_id', 'game_date'], ascending=[True, False])
        recent_form = stats.groupby('player_id').head(starts).groupby('player_id').agg({
            'innings_pitched': 'mean',
            'strikeouts': 'mean',
            'walks': 'mean',
            'earned_runs': 'mean'
        }).reset_index()
        recent_form = recent_form.rename(columns={
            'innings_pitched': 'recent_ip',
            'strikeouts': 'recent_k',
            'walks': 'recent_bb',
            'earned_runs': 'recent_er'
        })
        self.pitcher_statcast = pd.merge(
            self.pitcher_statcast, recent_form, on='player_id', how='left'
        )
        self.logger.info(f"Added recent form (last {starts} starts) to pitcher Statcast")

    def load_pitcher_advanced_stats(self):
        """Load advanced pitch arsenal and statcast metrics."""
        adv_file = os.path.join(self.data_dir, '..', 'processed', 'pitcher_advanced_stats.csv')
        if os.path.exists(adv_file):
            adv = pd.read_csv(adv_file)
            self.pitcher_statcast = pd.merge(
                self.pitcher_statcast, adv, on='player_id', how='left'
            )
            self.logger.info("Merged advanced pitcher stats into pitcher Statcast")
        else:
            self.logger.warning("No advanced pitcher stats file found")

    def load_data(self):
        """Load all required data for player props predictions"""
        self.logger.info("Loading data for player props predictions")
        
        # Load player projections
        projection_file = os.path.join(self.data_dir, '..', 'predictions', 'player_projections_2025.csv')
        if os.path.exists(projection_file):
            self.player_projections = pd.read_csv(projection_file)
            self.logger.info(f"Loaded player projections from {projection_file}")
            # Add advanced features
            self.load_batter_splits()
            self.add_recent_form(days=7)
            self.load_batter_advanced_stats()
        else:
            self.logger.warning("No player projections file found")
        
        # Load Statcast data
        statcast_dir = os.path.join(self.data_dir, '..', 'processed', 'statcast')
        
        batter_file = os.path.join(statcast_dir, 'processed_batter_all_time.csv')
        if os.path.exists(batter_file):
            self.batter_statcast = pd.read_csv(batter_file)
            self.batter_statcast['player_id'] = self.batter_statcast['player_id'].astype(str)
            self.logger.info(f"Loaded batter Statcast data from {batter_file}")
        else:
            self.logger.warning(f"Batter Statcast file not found: {batter_file}")
        
        pitcher_file = os.path.join(statcast_dir, 'processed_pitcher_all_time.csv')
        if os.path.exists(pitcher_file):
            self.pitcher_statcast = pd.read_csv(pitcher_file)
            self.pitcher_statcast['player_id'] = self.pitcher_statcast['player_id'].astype(str)
            self.logger.info(f"Loaded pitcher Statcast data from {pitcher_file}")
            # Add advanced pitcher features
            self.load_pitcher_splits()
            self.add_pitcher_recent_form(starts=5)
            self.load_pitcher_advanced_stats()
        else:
            self.logger.warning(f"Pitcher Statcast file not found: {pitcher_file}")
        
        # Load upcoming games
        # First check for a specific file
        games_file = os.path.join(self.data_dir, 'games_with_all_context.csv')
        if os.path.exists(games_file):
            self.game_data = pd.read_csv(games_file)
            self.logger.info(f"Loaded {len(self.game_data)} games with contextual data")
            # Add advanced game/context features
            self.load_umpire_stats()
            self.load_weather_data()
            self.load_ballpark_factors()
        else:
            # Try to find any games file with contextual data
            games_files = [
                os.path.join(self.data_dir, 'games_with_engineered_features.csv'),
                os.path.join(self.data_dir, 'games_with_weather.csv'),
                os.path.join(self.data_dir, 'games_with_umpires.csv')
            ]
            
            for file in games_files:
                if os.path.exists(file):
                    self.game_data = pd.read_csv(file)
                    self.logger.info(f"Loaded {len(self.game_data)} games from {file}")
                    break
            
            if self.game_data is None:
                self.logger.warning("No game data files found")
        
        # If no data is loaded yet, try player game stats
        if (self.player_projections is None and 
            self.batter_statcast is None and 
            self.pitcher_statcast is None):
            
            self.logger.info("Attempting to load player game stats...")
            self._load_player_game_stats()
    
    def _load_player_game_stats(self):
        """Load player game stats and create projections if needed"""
        years = [2023, 2024]
        player_data = None
        
        # Check analysis directory for player game stats
        for year in years:
            stats_file = os.path.join(self.data_dir, '..', '..', 'analysis', f'player_game_stats_{year}.csv')
            if os.path.exists(stats_file):
                self.logger.info(f"Loading player stats for {year}...")
                year_data = pd.read_csv(stats_file)
                if player_data is None:
                    player_data = year_data
                else:
                    player_data = pd.concat([player_data, year_data])
        
        if player_data is not None:
            self.logger.info(f"Loaded {len(player_data)} player-game records")
            
            # Calculate player-level stats
            player_stats = player_data.groupby('player_id').agg({
                'game_id': 'nunique',
                'at_bats': 'sum',
                'hits': 'sum',
                'home_runs': 'sum',
                'got_hit': 'mean',
                'got_home_run': 'mean'
            }).reset_index()
            
            # Rename columns
            player_stats = player_stats.rename(columns={
                'game_id': 'games',
                'got_hit': 'hit_rate',
                'got_home_run': 'hr_rate'
            })
            
            # Calculate batting average
            player_stats['batting_avg'] = player_stats['hits'] / player_stats['at_bats']
            
            # Add player names if available
            if 'player_name' in player_data.columns:
                name_mapping = player_data.drop_duplicates('player_id')[['player_id', 'player_name']]
                player_stats = pd.merge(player_stats, name_mapping, on='player_id', how='left')
            else:
                player_stats['player_name'] = player_stats['player_id'].apply(
                    lambda x: self.player_map.get(str(x), f"Player {x}")
                )
            
            # Create projected rates with some variance
            np.random.seed(42)
            player_stats['projected_hit_rate_2025'] = player_stats['hit_rate'] * np.random.uniform(0.9, 1.1, len(player_stats))
            player_stats['projected_hr_rate_2025'] = player_stats['hr_rate'] * np.random.uniform(0.85, 1.15, len(player_stats))
            
            # Calculate player consistency (streak analysis)
            if 'game_date' in player_data.columns:
                self._calculate_player_consistency(player_data, player_stats)
            
            # Save projections if they don't exist
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 'player_projections_2025.csv')
            if not os.path.exists(output_file):
                player_stats.to_csv(output_file, index=False)
                self.logger.info(f"Created and saved projections for {len(player_stats)} players")
            
            # Set as player projections
            self.player_projections = player_stats
        else:
            self.logger.warning("No player game stats found")
    
    def _calculate_player_consistency(self, player_data, player_stats):
        """Calculate consistency metrics for players"""
        self.logger.info("Calculating player consistency metrics...")
        
        # Ensure player_id is string type
        player_data['player_id'] = player_data['player_id'].astype(str)
        player_stats['player_id'] = player_stats['player_id'].astype(str)
        
        # Sort player data by date
        player_data['game_date'] = pd.to_datetime(player_data['game_date'])
        player_data = player_data.sort_values(['player_id', 'game_date'])
        
        # Initialize consistency columns
        player_stats['hit_consistency'] = np.nan
        player_stats['hr_consistency'] = np.nan
        player_stats['longest_hit_streak'] = np.nan
        player_stats['longest_hr_streak'] = np.nan
        
        # Process each player with sufficient games
        for player_id, group in tqdm(player_data.groupby('player_id'), desc="Calculating player consistency"):
            if len(group) < 10:
                continue
                
            player_idx = player_stats[player_stats['player_id'] == player_id].index
            if len(player_idx) == 0:
                continue
                
            # Get hit and HR sequences
            hit_sequence = group['got_hit'].values
            hr_sequence = group['got_home_run'].values
            
            # Calculate streaks
            hit_streaks = self._calculate_streaks(hit_sequence)
            hr_streaks = self._calculate_streaks(hr_sequence)
            
            # Calculate rolling standard deviation (lower means more consistent)
            hit_std = self._calculate_rolling_std(hit_sequence, window=10)
            hr_std = self._calculate_rolling_std(hr_sequence, window=10)
            
            # Convert to consistency score (0-1, higher is more consistent)
            # 0.5 is theoretical max std for binary outcome
            hit_consistency = max(0, min(1, 1 - (hit_std / 0.5)))
            hr_consistency = max(0, min(1, 1 - (hr_std / 0.5)))
            
            # Update player stats
            player_stats.loc[player_idx, 'hit_consistency'] = hit_consistency
            player_stats.loc[player_idx, 'hr_consistency'] = hr_consistency
            player_stats.loc[player_idx, 'longest_hit_streak'] = hit_streaks['longest_success']
            player_stats.loc[player_idx, 'longest_hr_streak'] = hr_streaks['longest_success']
    
    def _calculate_streaks(self, sequence):
        """Calculate success and failure streaks in a binary sequence"""
        current_success = 0
        current_failure = 0
        longest_success = 0
        longest_failure = 0
        
        for outcome in sequence:
            if outcome == 1:
                # Success
                current_success += 1
                current_failure = 0
                longest_success = max(longest_success, current_success)
            else:
                # Failure
                current_failure += 1
                current_success = 0
                longest_failure = max(longest_failure, current_failure)
        
        return {
            'longest_success': longest_success,
            'longest_failure': longest_failure
        }
    
    def _calculate_rolling_std(self, sequence, window=10):
        """Calculate the average standard deviation for rolling windows"""
        if len(sequence) <= window:
            return np.std(sequence)
        
        stds = []
        for i in range(len(sequence) - window + 1):
            window_slice = sequence[i:i+window]
            stds.append(np.std(window_slice))
        
        return np.mean(stds)
    
    def _add_batter_contextual_factors(self, player_id, team, is_home, game):
        """
        Add contextual factors for a batter
        
        Args:
            player_id (str): Player ID
            team (str): Team code
            is_home (bool): Whether player is on home team
            game (Series): Game data
            
        Returns:
            dict: Contextual adjustment factors
        """
        # Extract contextual factors from game
        ballpark_run_factor = float(game.get('ballpark_run_factor', 0)) if pd.notna(game.get('ballpark_run_factor')) else 0
        ballpark_hr_factor = float(game.get('ballpark_hr_factor', 0)) if pd.notna(game.get('ballpark_hr_factor')) else 0
        weather_score = float(game.get('weather_score', 0)) if pd.notna(game.get('weather_score')) else 0
        
        # Default factors
        factors = {
            'ballpark_hit_adjust': 0,
            'ballpark_hr_adjust': 0,
            'weather_hit_adjust': 0,
            'weather_hr_adjust': 0,
            'umpire_hit_adjust': 0,
        }
        
        # Ballpark adjustments
        if is_home:
            # Home batters get full ballpark effect
            factors['ballpark_hit_adjust'] = ballpark_run_factor * 0.2
            factors['ballpark_hr_adjust'] = ballpark_hr_factor * 0.3
        else:
            # Away batters get slightly reduced ballpark effect
            factors['ballpark_hit_adjust'] = ballpark_run_factor * 0.15
            factors['ballpark_hr_adjust'] = ballpark_hr_factor * 0.25
        
        # Weather adjustments
        factors['weather_hit_adjust'] = weather_score * 0.1
        factors['weather_hr_adjust'] = weather_score * 0.15
        
        # Umpire adjustments if available
        umpire_runs_boost = float(game.get('umpire_runs_boost', 0)) if pd.notna(game.get('umpire_runs_boost')) else 0
        factors['umpire_hit_adjust'] = umpire_runs_boost * 0.1
        
        # Player-specific contextual adjustments
        if self.batter_statcast is not None:
            player_data = self.batter_statcast[self.batter_statcast['player_id'] == str(player_id)]
            if not player_data.empty:
                # Get player tendencies
                pull_pct = float(player_data['pull_percent'].iloc[0]) if 'pull_percent' in player_data.columns and pd.notna(player_data['pull_percent'].iloc[0]) else 50
                
                # Extreme pull hitters are more affected by ballpark factors
                if pull_pct > 60:
                    factors['ballpark_hr_adjust'] *= 1.2
        
        return factors
    
    def predict_batter_hit_model(self, games_df=None, threshold=1):
        """
        Predict batter hit props
        
        Args:
            games_df (DataFrame): Games to predict for
            threshold (int): Hit threshold (1 for any hit, 2 for multi-hit)
            
        Returns:
            DataFrame: Hit prop predictions
        """
        prop_type = "multi_hit" if threshold > 1 else "hits"
        self.logger.info(f"Predicting {prop_type} props with threshold {threshold}...")
        
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            
            games_df = self.game_data
        
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        
        # Initialize predictions list
        predictions = []
        
        # Process each game
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Assign random hitters if we don't have actual lineups
            # In a real implementation, you would use actual team lineups
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            
            # Home team batters
            home_indices = random_indices[:num_batters]
            
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Get base projection
                if threshold == 1:
                    # Any hit
                    base_field = 'projected_hit_rate_2025' if 'projected_hit_rate_2025' in batter else 'hit_rate'
                    base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                else:
                    # Multiple hits
                    base_field = 'hit_rate'
                    single_hit_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                    
                    # Chance of multi-hit is approximately hit_rate^2 * 2
                    # This approximates the binomial distribution for 2+ hits in ~4 at-bats
                    base_rate = min(0.6, single_hit_rate * single_hit_rate * 2.2)
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, home_team, is_home=True, game=game
                )
                
                adjusted_rate = base_rate
                
                # Apply each adjustment factor
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Apply consistency adjustment
                if 'hit_consistency' in batter and pd.notna(batter['hit_consistency']):
                    consistency = float(batter['hit_consistency'])
                    # More consistent players stay closer to their baseline
                    adjusted_rate = (adjusted_rate * consistency) + (base_rate * (1 - consistency))
                
                # Ensure rate is reasonable
                adjusted_rate = max(0.05, min(0.8, adjusted_rate))
                
                # Convert to expected value based on average at-bats
                at_bats = 4.0  # Average at-bats per game
                
                if threshold == 1:
                    # Probability of at least one hit
                    expected_value = 1 - (1 - adjusted_rate) ** at_bats
                else:
                    # Use the binomial distribution for 2+ hits
                    expected_value = 1 - stats.binom.cdf(1, at_bats, adjusted_rate)
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line - typically 0.5 for any hit, 1.5 for multi-hit
                line = 0.5 if threshold == 1 else 1.5
                
                # Calculate over probability (expected value is already a probability)
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': prop_type,
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
            
            # Away team batters (similar logic to home team)
            away_indices = random_indices[num_batters:]
            
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Get base projection
                if threshold == 1:
                    # Any hit
                    base_field = 'projected_hit_rate_2025' if 'projected_hit_rate_2025' in batter else 'hit_rate'
                    base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                else:
                    # Multiple hits
                    base_field = 'hit_rate'
                    single_hit_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                    base_rate = min(0.6, single_hit_rate * single_hit_rate * 2.2)
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, away_team, is_home=False, game=game
                )
                
                adjusted_rate = base_rate
                
                # Apply each adjustment factor
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Ensure rate is reasonable
                adjusted_rate = max(0.05, min(0.8, adjusted_rate))
                
                # Convert to expected value based on average at-bats
                at_bats = 4.0  # Average at-bats per game
                
                if threshold == 1:
                    # Probability of at least one hit
                    expected_value = 1 - (1 - adjusted_rate) ** at_bats
                else:
                    # Use binomial distribution for 2+ hits
                    expected_value = 1 - stats.binom.cdf(1, at_bats, adjusted_rate)
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line - typically 0.5 for any hit, 1.5 for multi-hit
                line = 0.5 if threshold == 1 else 1.5
                
                # Calculate over probability
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': prop_type,
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"player_props_{prop_type}_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} {prop_type} prop predictions to {output_file}")
        
        return predictions_df

    def predict_batter_home_run_model(self, games_df=None):
        """
        Predict batter home run props
        
        Args:
            games_df (DataFrame): Games to predict for
            
        Returns:
            DataFrame: Home run prop predictions
        """
        self.logger.info("Predicting home run props...")
        
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            
            games_df = self.game_data
        
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        
        # Initialize predictions list
        predictions = []
        
        # Process each game
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Assign random hitters if we don't have actual lineups
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            
            # Home team batters
            home_indices = random_indices[:num_batters]
            
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Get base projection
                base_field = 'projected_hr_rate_2025' if 'projected_hr_rate_2025' in batter else 'hr_rate'
                base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.05
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, home_team, is_home=True, game=game
                )
                
                adjusted_rate = base_rate
                
                # Apply each adjustment factor
                for factor_name, factor_value in contextual_factors.items():
                    if 'hr' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Apply consistency adjustment
                if 'hr_consistency' in batter and pd.notna(batter['hr_consistency']):
                    consistency = float(batter['hr_consistency'])
                    # More consistent players stay closer to their baseline
                    adjusted_rate = (adjusted_rate * consistency) + (base_rate * (1 - consistency))
                
                # Ensure rate is reasonable
                adjusted_rate = max(0.01, min(0.25, adjusted_rate))
                
                # Convert to expected value based on average at-bats
                at_bats = 4.0  # Average at-bats per game
                
                # Probability of at least one HR
                expected_value = 1 - (1 - adjusted_rate) ** at_bats
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line - typically 0.5 for home runs
                line = 0.5
                
                # Calculate over probability
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': 'hr',
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
            
            # Away team batters (similar logic to home team)
            away_indices = random_indices[num_batters:]
            
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Get base projection
                base_field = 'projected_hr_rate_2025' if 'projected_hr_rate_2025' in batter else 'hr_rate'
                base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.05
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, away_team, is_home=False, game=game
                )
                
                adjusted_rate = base_rate
                
                # Apply each adjustment factor
                for factor_name, factor_value in contextual_factors.items():
                    if 'hr' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Ensure rate is reasonable
                adjusted_rate = max(0.01, min(0.25, adjusted_rate))
                
                # Convert to expected value based on average at-bats
                at_bats = 4.0  # Average at-bats per game
                
                # Probability of at least one HR
                expected_value = 1 - (1 - adjusted_rate) ** at_bats
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line - typically 0.5 for home runs
                line = 0.5
                
                # Calculate over probability
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': 'hr',
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"player_props_hr_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} home run prop predictions to {output_file}")
        
        return predictions_df

    def predict_batter_runs_rbi_model(self, games_df=None, prop_type='runs', multi_threshold=False):
        """
        Predict batter runs or RBI props
        
        Args:
            games_df (DataFrame): Games to predict for
            prop_type (str): 'runs' or 'rbi'
            multi_threshold (bool): Whether to use 1.5+ threshold
            
        Returns:
            DataFrame: Runs or RBI prop predictions
        """
        self.logger.info(f"Predicting {prop_type} props (multi_threshold={multi_threshold})...")
        
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            
            games_df = self.game_data
        
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        
        # Initialize predictions list
        predictions = []
        
        # Process each game
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Assign random hitters if we don't have actual lineups
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            
            # Home team batters
            home_indices = random_indices[:num_batters]
            
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Get base projection - use hit rate as proxy, scaled appropriately
                if prop_type == 'runs':
                    # Runs correlate with on-base percentage (hits + walks)
                    base_field = 'hit_rate'
                    base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                    
                    # Scale for runs (typically ~60-70% of hits turn into runs)
                    base_rate = base_rate * 0.65
                    
                    # Use batting position to adjust
                    # Top of order gets more runs, middle more RBIs, bottom fewer of both
                    batting_position = i % 9  # Simulated position in lineup
                    if batting_position < 3:
                        base_rate *= 1.2  # Top of order bonus
                    elif batting_position > 6:
                        base_rate *= 0.8  # Bottom of order penalty
                else:  # RBI
                    # RBIs correlate with power and hit rate
                    hit_rate = float(batter['hit_rate']) if pd.notna(batter['hit_rate']) else 0.3
                    hr_rate = float(batter['hr_rate']) if pd.notna(batter['hr_rate']) else 0.05
                    
                    # Combined metric (weighted toward power)
                    base_rate = (hit_rate * 0.4) + (hr_rate * 3.0)
                    
                    # Use batting position to adjust
                    batting_position = i % 9
                    if batting_position >= 3 and batting_position <= 5:
                        base_rate *= 1.3  # Middle of order bonus for RBIs
                    elif batting_position > 6:
                        base_rate *= 0.7  # Bottom of order penalty
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, home_team, is_home=True, game=game
                )
                
                adjusted_rate = base_rate
                
                # Apply each adjustment factor
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Runs and RBIs are team-dependent
                # Better teams score more runs, creating more opportunities
                team_quality_factor = 0
                if 'home_win_pct' in game and pd.notna(game['home_win_pct']):
                    # Above .500 teams get a bonus, below get penalty
                    team_quality_factor = (float(game['home_win_pct']) - 0.5) * 0.5
                adjusted_rate *= (1 + team_quality_factor)
                
                # Apply recency bias (if available)
                # This would use player's recent performance trend
                recency_adjustment = 0
                
                # Ensure rate is reasonable
                if multi_threshold:
                    # For 1.5+ threshold
                    adjusted_rate = max(0.01, min(0.3, adjusted_rate))
                else:
                    # For 0.5+ threshold
                    adjusted_rate = max(0.05, min(0.6, adjusted_rate))
                
                # Convert to expected value based on game opportunities
                if multi_threshold:
                    # Probability of 2+ runs/RBIs
                    # Use Poisson distribution
                    expected_value = 1 - stats.poisson.cdf(1, adjusted_rate * 5)
                else:
                    # Probability of 1+ runs/RBIs
                    expected_value = 1 - stats.poisson.cdf(0, adjusted_rate * 5)
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line
                line = 1.5 if multi_threshold else 0.5
                
                # Calculate over probability
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': f"{prop_type}{'_multi' if multi_threshold else ''}",
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
            
            # Away team batters (similar logic to home team)
            away_indices = random_indices[num_batters:]
            
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                
                # Similar logic for away batters...
                if prop_type == 'runs':
                    base_field = 'hit_rate'
                    base_rate = float(batter[base_field]) if pd.notna(batter[base_field]) else 0.3
                    base_rate = base_rate * 0.65
                    
                    batting_position = i % 9
                    if batting_position < 3:
                        base_rate *= 1.2
                    elif batting_position > 6:
                        base_rate *= 0.8
                else:  # RBI
                    hit_rate = float(batter['hit_rate']) if pd.notna(batter['hit_rate']) else 0.3
                    hr_rate = float(batter['hr_rate']) if pd.notna(batter['hr_rate']) else 0.05
                    base_rate = (hit_rate * 0.4) + (hr_rate * 3.0)
                    
                    batting_position = i % 9
                    if batting_position >= 3 and batting_position <= 5:
                        base_rate *= 1.3
                    elif batting_position > 6:
                        base_rate *= 0.7
                
                # Apply contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(
                    player_id, away_team, is_home=False, game=game
                )
                
                adjusted_rate = base_rate
                
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        adjusted_rate *= (1 + factor_value)
                
                # Team quality adjustment
                team_quality_factor = 0
                if 'away_win_pct' in game and pd.notna(game['away_win_pct']):
                    team_quality_factor = (float(game['away_win_pct']) - 0.5) * 0.5
                adjusted_rate *= (1 + team_quality_factor)
                
                # Ensure rate is reasonable
                if multi_threshold:
                    adjusted_rate = max(0.01, min(0.3, adjusted_rate))
                else:
                    adjusted_rate = max(0.05, min(0.6, adjusted_rate))
                
                # Convert to expected value
                if multi_threshold:
                    expected_value = 1 - stats.poisson.cdf(1, adjusted_rate * 5)
                else:
                    expected_value = 1 - stats.poisson.cdf(0, adjusted_rate * 5)
                
                # Round expected value
                expected_value = round(expected_value, 3)
                
                # Set line
                line = 1.5 if multi_threshold else 0.5
                
                # Calculate over probability
                over_prob = expected_value
                
                # Add prediction
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': f"{prop_type}{'_multi' if multi_threshold else ''}",
                    'base_rate': base_rate,
                    'adjusted_rate': adjusted_rate,
                    'expected_value': expected_value,
                    'line': line,
                    'over_prob': over_prob,
                    'edge': (over_prob - 0.5) * 2
                })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"player_props_{prop_type}{'_multi' if multi_threshold else ''}_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} {prop_type} prop predictions to {output_file}")
        
        return predictions_df

    def predict_pitcher_strikeouts(self, games_df=None, multi_lines=True):
        """
        Predict pitcher strikeout props
        
        Args:
            games_df (DataFrame): Games to predict for
            multi_lines (bool): Whether to generate multiple lines (5.5, 6.5, etc.)
            
        Returns:
            DataFrame: Strikeout prop predictions
        """
        self.logger.info("Predicting pitcher strikeout props...")
        
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            
            games_df = self.game_data
        
        # Initialize predictions list
        predictions = []
        
        # Process each game
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get pitcher IDs if available
            home_pitcher_id = game.get('home_starting_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id')
            
            # Get pitcher names if available
            home_pitcher_name = self.player_map.get(str(home_pitcher_id), "Home Starting Pitcher")
            away_pitcher_name = self.player_map.get(str(away_pitcher_id), "Away Starting Pitcher")
            
            # Default K rates
            home_k_rate = 6.0  # League average Ks per 9 innings
            away_k_rate = 6.0
            
            # Get K rates from Statcast if available
            if self.pitcher_statcast is not None:
                if home_pitcher_id:
                    home_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(home_pitcher_id)]
                    if not home_pitcher.empty:
                        if 'whiff_rate' in home_pitcher.columns:
                            # Convert whiff rate to strikeouts
                            # Typical conversion: ~25 whiffs per 9 innings = ~8 strikeouts
                            home_k_rate = float(home_pitcher['whiff_rate'].iloc[0]) * 25
                        elif 'strikeout_rate' in home_pitcher.columns:
                            home_k_rate = float(home_pitcher['strikeout_rate'].iloc[0]) * 27  # 27 outs per 9 innings
                
                if away_pitcher_id:
                    away_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(away_pitcher_id)]
                    if not away_pitcher.empty:
                        if 'whiff_rate' in away_pitcher.columns:
                            away_k_rate = float(away_pitcher['whiff_rate'].iloc[0]) * 25
                        elif 'strikeout_rate' in away_pitcher.columns:
                            away_k_rate = float(away_pitcher['strikeout_rate'].iloc[0]) * 27
            
            # Apply contextual adjustments
            
            # Get contextual factors from game
            umpire_strikeout_boost = float(game.get('umpire_strikeout_boost', 0)) if pd.notna(game.get('umpire_strikeout_boost')) else 0
            ballpark_factor = float(game.get('ballpark_run_factor', 0)) if pd.notna(game.get('ballpark_run_factor')) else 0
            weather_score = float(game.get('weather_score', 0)) if pd.notna(game.get('weather_score')) else 0
            
            # Apply adjustments to home pitcher
            home_adjusted_rate = home_k_rate
            
            # Umpire effect on Ks
            home_adjusted_rate *= (1 + umpire_strikeout_boost * 0.1)
            
            # Weather effect (offense-friendly weather usually means fewer Ks)
            home_adjusted_rate *= (1 - weather_score * 0.05)
            
            # Opposing team quality
            away_team_contact_rate = 0.75  # Default contact rate (75%)
            # In a real model, you'd use the opposing team's contact/strikeout rates
            home_adjusted_rate *= (2 - away_team_contact_rate)  # Lower contact rate = more Ks
            
            # Ensure reasonable range
            home_adjusted_rate = max(3.0, min(12.0, home_adjusted_rate))
            
            # Scale to expected Ks (assuming ~6 innings pitched)
            home_expected_ks = home_adjusted_rate * (6.0 / 9.0)  # Convert from per 9 to per 6 innings
            
            # Similar adjustments for away pitcher
            away_adjusted_rate = away_k_rate
            away_adjusted_rate *= (1 + umpire_strikeout_boost * 0.1)
            away_adjusted_rate *= (1 - weather_score * 0.05)
            
            # Home team contact rate
            home_team_contact_rate = 0.75  # Default
            away_adjusted_rate *= (2 - home_team_contact_rate)
            
            # Ensure reasonable range
            away_adjusted_rate = max(3.0, min(12.0, away_adjusted_rate))
            
            # Scale to expected Ks
            away_expected_ks = away_adjusted_rate * (6.0 / 9.0)
            
            # Generate predictions for home pitcher
            if multi_lines:
                # Generate multiple lines
                for line in [4.5, 5.5, 6.5, 7.5, 8.5]:
                    if abs(line - home_expected_ks) <= 2.0:
                        # Only generate lines reasonably close to expected value
                        # Use normal distribution for K props
                        over_prob = 1 - stats.norm.cdf(line, home_expected_ks, 2.0)
                        
                        predictions.append({
                            'game_id': game_id,
                            'team': home_team,
                            'player_id': str(home_pitcher_id) if home_pitcher_id else 'unknown',
                            'player_name': home_pitcher_name,
                            'is_home': True,
                            'prop_type': 'strikeouts',
                            'expected_value': round(home_expected_ks, 1),
                            'line': line,
                            'over_prob': round(over_prob, 3),
                            'edge': round((over_prob - 0.5) * 2, 3)
                        })
            else:
                # Just one standard line
                line = round(home_expected_ks * 0.9 + 0.5)  # Slightly below expected
                over_prob = 1 - stats.norm.cdf(line, home_expected_ks, 2.0)
                
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(home_pitcher_id) if home_pitcher_id else 'unknown',
                    'player_name': home_pitcher_name,
                    'is_home': True,
                    'prop_type': 'strikeouts',
                    'expected_value': round(home_expected_ks, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            
            # Generate predictions for away pitcher (similar logic)
            if multi_lines:
                for line in [4.5, 5.5, 6.5, 7.5, 8.5]:
                    if abs(line - away_expected_ks) <= 2.0:
                        over_prob = 1 - stats.norm.cdf(line, away_expected_ks, 2.0)
                        
                        predictions.append({
                            'game_id': game_id,
                            'team': away_team,
                            'player_id': str(away_pitcher_id) if away_pitcher_id else 'unknown',
                            'player_name': away_pitcher_name,
                            'is_home': False,
                            'prop_type': 'strikeouts',
                            'expected_value': round(away_expected_ks, 1),
                            'line': line,
                            'over_prob': round(over_prob, 3),
                            'edge': round((over_prob - 0.5) * 2, 3)
                        })
            else:
                line = round(away_expected_ks * 0.9 + 0.5)
                over_prob = 1 - stats.norm.cdf(line, away_expected_ks, 2.0)
                
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(away_pitcher_id) if away_pitcher_id else 'unknown',
                    'player_name': away_pitcher_name,
                    'is_home': False,
                    'prop_type': 'strikeouts',
                    'expected_value': round(away_expected_ks, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"player_props_strikeouts_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} strikeout prop predictions to {output_file}")
        
        return predictions_df

    def predict_pitcher_outs_innings(self, games_df=None, prop_type='outs'):
        """
        Predict pitcher outs or innings props
        
        Args:
            games_df (DataFrame): Games to predict for
            prop_type (str): Either 'outs' or 'innings'
            
        Returns:
            DataFrame: Outs/innings prop predictions
        """
        self.logger.info(f"Predicting pitcher {prop_type} props...")
        
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            
            games_df = self.game_data
        
        # Initialize predictions list
        predictions = []
        
        # Process each game
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get pitcher IDs if available
            home_pitcher_id = game.get('home_starting_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id')
            
            # Get pitcher names if available
            home_pitcher_name = self.player_map.get(str(home_pitcher_id), "Home Starting Pitcher")
            away_pitcher_name = self.player_map.get(str(away_pitcher_id), "Away Starting Pitcher")
            
            # Default innings pitched
            home_expected_innings = 5.5  # League average
            away_expected_innings = 5.5
            
            # Get innings pitched from projections if available
            if self.player_projections is not None:
                if home_pitcher_id:
                    home_pitcher = self.player_projections[self.player_projections['player_id'] == str(home_pitcher_id)]
                    if not home_pitcher.empty and 'innings_per_start' in home_pitcher.columns:
                        home_expected_innings = float(home_pitcher['innings_per_start'].iloc[0])
                
                if away_pitcher_id:
                    away_pitcher = self.player_projections[self.player_projections['player_id'] == str(away_pitcher_id)]
                    if not away_pitcher.empty and 'innings_per_start' in away_pitcher.columns:
                        away_expected_innings = float(away_pitcher['innings_per_start'].iloc[0])
            
            # Apply contextual adjustments
            
            # Get contextual factors from game
            ballpark_factor = float(game.get('ballpark_run_factor', 0)) if pd.notna(game.get('ballpark_run_factor')) else 0
            weather_score = float(game.get('weather_score', 0)) if pd.notna(game.get('weather_score')) else 0
            
            # Apply adjustments to home pitcher
            home_adjusted_innings = home_expected_innings
            
            # Weather effect (offense-friendly weather usually means shorter outings)
            home_adjusted_innings *= (1 - weather_score * 0.05)
            
            # Opposing team quality
            away_team_offense_rating = 0.5  # Default (0-1 scale)
            # In a real model, you'd use the opposing team's offense rating
            home_adjusted_innings *= (1 - away_team_offense_rating * 0.1)
            
            # Ensure reasonable range
            home_adjusted_innings = max(4.0, min(7.0, home_adjusted_innings))
            
            # Similar adjustments for away pitcher
            away_adjusted_innings = away_expected_innings
            away_adjusted_innings *= (1 - weather_score * 0.05)
            
            # Home team offense rating
            home_team_offense_rating = 0.5  # Default
            away_adjusted_innings *= (1 - home_team_offense_rating * 0.1)
            
            # Ensure reasonable range
            away_adjusted_innings = max(4.0, min(7.0, away_adjusted_innings))
            
            # Generate predictions for home pitcher
            if prop_type == 'outs':
                # Convert innings to outs
                home_expected_outs = home_adjusted_innings * 3
                line = round(home_expected_outs * 0.9)  # Slightly below expected
                
                # Use normal distribution for outs
                over_prob = 1 - stats.norm.cdf(line, home_expected_outs, 2.0)
                
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(home_pitcher_id) if home_pitcher_id else 'unknown',
                    'player_name': home_pitcher_name,
                    'is_home': True,
                    'prop_type': 'outs',
                    'expected_value': round(home_expected_outs, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            else:  # innings
                line = round(home_adjusted_innings * 0.9, 1)  # Slightly below expected
                
                # Use normal distribution for innings
                over_prob = 1 - stats.norm.cdf(line, home_adjusted_innings, 0.7)
                
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(home_pitcher_id) if home_pitcher_id else 'unknown',
                    'player_name': home_pitcher_name,
                    'is_home': True,
                    'prop_type': 'innings',
                    'expected_value': round(home_adjusted_innings, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            
            # Generate predictions for away pitcher
            if prop_type == 'outs':
                away_expected_outs = away_adjusted_innings * 3
                line = round(away_expected_outs * 0.9)
                
                over_prob = 1 - stats.norm.cdf(line, away_expected_outs, 2.0)
                
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(away_pitcher_id) if away_pitcher_id else 'unknown',
                    'player_name': away_pitcher_name,
                    'is_home': False,
                    'prop_type': 'outs',
                    'expected_value': round(away_expected_outs, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            else:  # innings
                line = round(away_adjusted_innings * 0.9, 1)
                
                over_prob = 1 - stats.norm.cdf(line, away_adjusted_innings, 0.7)
                
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(away_pitcher_id) if away_pitcher_id else 'unknown',
                    'player_name': away_pitcher_name,
                    'is_home': False,
                    'prop_type': 'innings',
                    'expected_value': round(away_adjusted_innings, 1),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"player_props_{prop_type}_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} {prop_type} prop predictions to {output_file}")
        
        return predictions_df

    def get_best_props_by_edge(self, min_edge=0.05, max_edge=0.5, prop_types=None):
        """
        Get the best props by edge
        
        Args:
            min_edge (float): Minimum edge to consider
            max_edge (float): Maximum edge to consider
            prop_types (list): List of prop types to consider (None for all)
            
        Returns:
            DataFrame: Best props by edge
        """
        self.logger.info(f"Getting best props by edge (min: {min_edge}, max: {max_edge})")
        
        # Load all prediction files
        predictions_dir = os.path.join(self.data_dir, '..', 'predictions')
        all_predictions = []
        
        if prop_types is None:
            prop_types = ['hits', 'multi_hits', 'home_runs', 'runs', 'rbi', 'strikeouts', 'outs', 'innings']
        
        for prop_type in prop_types:
            file_path = os.path.join(predictions_dir, f"player_props_{prop_type}_predictions.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['prop_type'] = prop_type
                    all_predictions.append(df)
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {str(e)}")
        
        if not all_predictions:
            self.logger.warning("No prediction files found")
            return pd.DataFrame()
        
        # Combine all predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        # Filter by edge
        filtered_df = combined_df[
            (combined_df['edge'] >= min_edge) & 
            (combined_df['edge'] <= max_edge)
        ]
        
        # Sort by edge
        filtered_df = filtered_df.sort_values('edge', ascending=False)
        
        # Add some additional useful columns
        filtered_df['implied_prob'] = 0.5 - (filtered_df['edge'] / 2)  # Convert edge to implied probability
        filtered_df['value'] = filtered_df['over_prob'] - filtered_df['implied_prob']
        
        # Format the output
        output_cols = [
            'game_id', 'team', 'player_name', 'prop_type', 'line',
            'expected_value', 'over_prob', 'implied_prob', 'edge', 'value'
        ]
        
        filtered_df = filtered_df[output_cols]
        
        # Save to file
        output_file = os.path.join(predictions_dir, "best_props_by_edge.csv")
        filtered_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(filtered_df)} best props to {output_file}")
        
        return filtered_df

    def backtest_props(self, start_date=None, end_date=None, min_edge=0.05, max_edge=0.5):
        """
        Backtest player prop predictions
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            min_edge (float): Minimum edge to consider
            max_edge (float): Maximum edge to consider
            
        Returns:
            DataFrame: Backtest results
        """
        self.logger.info(f"Backtesting props from {start_date} to {end_date}")
        
        # Load historical game data
        historical_dir = os.path.join(self.data_dir, '..', 'historical')
        if not os.path.exists(historical_dir):
            self.logger.error("Historical data directory not found")
            return pd.DataFrame()
        
        # Load player stats
        player_stats_file = os.path.join(historical_dir, 'player_stats.csv')
        if not os.path.exists(player_stats_file):
            self.logger.error("Player stats file not found")
            return pd.DataFrame()
        
        player_stats = pd.read_csv(player_stats_file)
        
        # Convert dates to datetime
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        
        # Filter player stats by date range
        player_stats['game_date'] = pd.to_datetime(player_stats['game_date'])
        if start_date:
            player_stats = player_stats[player_stats['game_date'] >= start_date]
        if end_date:
            player_stats = player_stats[player_stats['game_date'] <= end_date]
        
        # Initialize results
        results = []
        
        # Process each prop type
        prop_types = ['hits', 'multi_hits', 'home_runs', 'runs', 'rbi', 'strikeouts', 'outs', 'innings']
        
        for prop_type in prop_types:
            # Load predictions for this prop type
            pred_file = os.path.join(self.data_dir, '..', 'predictions', f"player_props_{prop_type}_predictions.csv")
            if not os.path.exists(pred_file):
                continue
            
            predictions = pd.read_csv(pred_file)
            
            # Filter by edge
            predictions = predictions[
                (predictions['edge'] >= min_edge) & 
                (predictions['edge'] <= max_edge)
            ]
            
            # Merge with actual results
            if prop_type == 'hits':
                actual_col = 'hits'
            elif prop_type == 'multi_hits':
                actual_col = 'hits'
                predictions['line'] = 1.5  # For multi-hits, we're always betting over 1.5
            elif prop_type == 'home_runs':
                actual_col = 'home_runs'
            elif prop_type == 'runs':
                actual_col = 'runs'
            elif prop_type == 'rbi':
                actual_col = 'rbi'
            elif prop_type == 'strikeouts':
                actual_col = 'strikeouts'
            elif prop_type == 'outs':
                actual_col = 'outs'
            elif prop_type == 'innings':
                actual_col = 'innings_pitched'
            
            # Merge predictions with actual results
            merged = pd.merge(
                predictions,
                player_stats[['game_id', 'player_id', actual_col]],
                on=['game_id', 'player_id'],
                how='inner'
            )
            
            # Calculate results
            for _, row in merged.iterrows():
                actual = row[actual_col]
                line = row['line']
                over_prob = row['over_prob']
                edge = row['edge']
                
                # For multi-hits, we need to check if hits > 1.5
                if prop_type == 'multi_hits':
                    hit = 1 if actual > 1.5 else 0
                else:
                    hit = 1 if actual > line else 0
                
                results.append({
                    'game_id': row['game_id'],
                    'player_name': row['player_name'],
                    'prop_type': prop_type,
                    'line': line,
                    'actual': actual,
                    'over_prob': over_prob,
                    'edge': edge,
                    'hit': hit
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            self.logger.warning("No backtest results found")
            return pd.DataFrame()
        
        # Calculate summary statistics
        summary = results_df.groupby('prop_type').agg({
            'hit': ['count', 'sum', 'mean'],
            'edge': 'mean'
        }).reset_index()
        
        summary.columns = ['prop_type', 'bets', 'wins', 'win_rate', 'avg_edge']
        summary['win_rate'] = summary['win_rate'].round(3)
        summary['avg_edge'] = summary['avg_edge'].round(3)
        
        # Save results
        output_dir = os.path.join(self.data_dir, '..', 'backtest')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "prop_backtest_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Save summary
        summary_file = os.path.join(output_dir, "prop_backtest_summary.csv")
        summary.to_csv(summary_file, index=False)
        
        self.logger.info(f"Backtest complete. Results saved to {results_file} and {summary_file}")
        
        return summary

    def run_daily_workflow(self, date=None):
        """
        Run the daily workflow for player props
        
        Args:
            date (str): Date in YYYY-MM-DD format (None for today)
            
        Returns:
            dict: Results of the daily workflow
        """
        if date is not None and not isinstance(date, str):
            raise ValueError(f"The 'date' argument to run_daily_workflow must be a string in 'YYYY-MM-DD' format or None. Got: {type(date)}")

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"Running daily workflow for {date}")
        
        # Load data
        self.load_data()
        
        # Filter games for the specified date
        if self.game_data is not None:
            self.game_data['game_date'] = pd.to_datetime(self.game_data['game_date'])
            games_df = self.game_data[self.game_data['game_date'].dt.strftime('%Y-%m-%d') == date]
        else:
            self.logger.error("No game data available")
            return {}
        
        if len(games_df) == 0:
            self.logger.warning(f"No games found for {date}")
            return {}
        
        # Initialize results
        results = {}
        
        # Predict batter props
        self.logger.info("Predicting batter props...")
        
        # Hits
        hits_predictions = self.predict_batter_hit_model(games_df, threshold=0.5)
        results['hits'] = len(hits_predictions)
        
        # Multi-hits
        multi_hits_predictions = self.predict_batter_hit_model(games_df, threshold=1.5)
        results['multi_hits'] = len(multi_hits_predictions)
        
        # Home runs
        hr_predictions = self.predict_batter_home_run_model(games_df)
        results['home_runs'] = len(hr_predictions)
        
        # Runs and RBIs
        runs_predictions = self.predict_batter_runs_rbi_model(games_df, prop_type='runs')
        results['runs'] = len(runs_predictions)
        
        rbi_predictions = self.predict_batter_runs_rbi_model(games_df, prop_type='rbi')
        results['rbi'] = len(rbi_predictions)
        
        # Predict pitcher props
        self.logger.info("Predicting pitcher props...")
        
        # Strikeouts
        k_predictions = self.predict_pitcher_strikeouts(games_df)
        results['strikeouts'] = len(k_predictions)
        
        # Outs
        outs_predictions = self.predict_pitcher_outs_innings(games_df, prop_type='outs')
        results['outs'] = len(outs_predictions)
        
        # Innings
        innings_predictions = self.predict_pitcher_outs_innings(games_df, prop_type='innings')
        results['innings'] = len(innings_predictions)
        
        # Get best props by edge
        self.logger.info("Getting best props by edge...")
        best_props = self.get_best_props_by_edge(min_edge=0.05, max_edge=0.5)
        results['best_props'] = len(best_props)
        
        # Save workflow results
        output_dir = os.path.join(self.data_dir, '..', 'workflow')
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, f"daily_workflow_{date}.csv")
        pd.DataFrame([results]).to_csv(results_file, index=False)
        
        self.logger.info(f"Daily workflow complete. Results saved to {results_file}")
        
        return results

    def load_umpire_stats(self):
        """Load umpire stats and merge into game_data."""
        umpire_file = os.path.join(self.data_dir, '..', 'processed', 'umpire_stats.csv')
        if os.path.exists(umpire_file) and self.game_data is not None:
            ump = pd.read_csv(umpire_file)
            # Merge on umpire name or id if available
            if 'umpire_id' in ump.columns and 'umpire_id' in self.game_data.columns:
                self.game_data = pd.merge(self.game_data, ump, on='umpire_id', how='left')
            elif 'umpire_name' in ump.columns and 'umpire_name' in self.game_data.columns:
                self.game_data = pd.merge(self.game_data, ump, on='umpire_name', how='left')
            self.logger.info("Merged umpire stats into game data")
        else:
            self.logger.warning("No umpire stats file found or no game data loaded")

    def load_weather_data(self):
        """Load weather data and merge into game_data."""
        weather_file = os.path.join(self.data_dir, '..', 'processed', 'weather_data.csv')
        if os.path.exists(weather_file) and self.game_data is not None:
            weather = pd.read_csv(weather_file)
            # Merge on game_id and date
            if 'game_id' in weather.columns and 'game_id' in self.game_data.columns:
                self.game_data = pd.merge(self.game_data, weather, on='game_id', how='left')
            self.logger.info("Merged weather data into game data")
        else:
            self.logger.warning("No weather data file found or no game data loaded")

    def load_ballpark_factors(self):
        """Load ballpark factors and merge into game_data."""
        park_file = os.path.join(self.data_dir, '..', 'processed', 'ballpark_factors.csv')
        if os.path.exists(park_file) and self.game_data is not None:
            parks = pd.read_csv(park_file)
            # Merge on home_team or park_id
            if 'home_team' in parks.columns and 'home_team' in self.game_data.columns:
                self.game_data = pd.merge(self.game_data, parks, on='home_team', how='left')
            elif 'park_id' in parks.columns and 'park_id' in self.game_data.columns:
                self.game_data = pd.merge(self.game_data, parks, on='park_id', how='left')
            self.logger.info("Merged ballpark factors into game data")
        else:
            self.logger.warning("No ballpark factors file found or no game data loaded")

    def predict_batter_total_bases_model(self, games_df=None, threshold=1.5):
        """
        Predict batter total bases props
        Args:
            games_df (DataFrame): Games to predict for
            threshold (float): Total bases threshold (e.g., 1.5, 2.5)
        Returns:
            DataFrame: Total bases prop predictions
        """
        self.logger.info(f"Predicting total bases props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            # Home team batters
            home_indices = random_indices[:num_batters]
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                # Estimate 1B, 2B, 3B, HR rates
                hit_rate = float(batter['hit_rate']) if 'hit_rate' in batter and pd.notna(batter['hit_rate']) else 0.3
                hr_rate = float(batter['hr_rate']) if 'hr_rate' in batter and pd.notna(batter['hr_rate']) else 0.05
                # Use advanced stats if available
                double_rate = float(batter['double_rate']) if 'double_rate' in batter and pd.notna(batter['double_rate']) else 0.05
                triple_rate = float(batter['triple_rate']) if 'triple_rate' in batter and pd.notna(batter['triple_rate']) else 0.005
                single_rate = max(0.0, hit_rate - double_rate - triple_rate - hr_rate)
                # Contextual adjustments
                contextual_factors = self._add_batter_contextual_factors(player_id, home_team, is_home=True, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        single_rate *= (1 + factor_value)
                        double_rate *= (1 + factor_value)
                        triple_rate *= (1 + factor_value)
                    if 'hr' in factor_name:
                        hr_rate *= (1 + factor_value)
                # Calculate expected TB for 4 AB
                at_bats = 4.0
                exp_tb = (single_rate * 1 + double_rate * 2 + triple_rate * 3 + hr_rate * 4) * at_bats
                # Probability of exceeding threshold (Poisson approx)
                over_prob = 1 - stats.poisson.cdf(threshold, exp_tb)
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': 'total_bases',
                    'expected_value': round(exp_tb, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            # Away team batters
            away_indices = random_indices[num_batters:]
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                hit_rate = float(batter['hit_rate']) if 'hit_rate' in batter and pd.notna(batter['hit_rate']) else 0.3
                hr_rate = float(batter['hr_rate']) if 'hr_rate' in batter and pd.notna(batter['hr_rate']) else 0.05
                double_rate = float(batter['double_rate']) if 'double_rate' in batter and pd.notna(batter['double_rate']) else 0.05
                triple_rate = float(batter['triple_rate']) if 'triple_rate' in batter and pd.notna(batter['triple_rate']) else 0.005
                single_rate = max(0.0, hit_rate - double_rate - triple_rate - hr_rate)
                contextual_factors = self._add_batter_contextual_factors(player_id, away_team, is_home=False, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        single_rate *= (1 + factor_value)
                        double_rate *= (1 + factor_value)
                        triple_rate *= (1 + factor_value)
                    if 'hr' in factor_name:
                        hr_rate *= (1 + factor_value)
                at_bats = 4.0
                exp_tb = (single_rate * 1 + double_rate * 2 + triple_rate * 3 + hr_rate * 4) * at_bats
                over_prob = 1 - stats.poisson.cdf(threshold, exp_tb)
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': 'total_bases',
                    'expected_value': round(exp_tb, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_total_bases_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} total bases prop predictions to {output_file}")
        return predictions_df

    def predict_batter_walks_model(self, games_df=None, threshold=0.5):
        """
        Predict batter walks props (over/under 0.5)
        Args:
            games_df (DataFrame): Games to predict for
            threshold (float): Walks threshold (default 0.5)
        Returns:
            DataFrame: Walks prop predictions
        """
        self.logger.info(f"Predicting walks props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            # Home team batters
            home_indices = random_indices[:num_batters]
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                # Use walk rate if available
                bb_rate = float(batter['bb_rate']) if 'bb_rate' in batter and pd.notna(batter['bb_rate']) else 0.08
                # Contextual adjustments (e.g., opposing pitcher control, umpire zone)
                contextual_factors = self._add_batter_contextual_factors(player_id, home_team, is_home=True, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        bb_rate *= (1 + factor_value * 0.5)  # Less sensitive than hits
                # Estimate walks in 4 PA
                pa = 4.0
                exp_walks = bb_rate * pa
                # Probability of at least 1 walk (binomial)
                over_prob = 1 - stats.binom.cdf(0, int(pa), bb_rate)
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': 'walks',
                    'expected_value': round(exp_walks, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            # Away team batters
            away_indices = random_indices[num_batters:]
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                bb_rate = float(batter['bb_rate']) if 'bb_rate' in batter and pd.notna(batter['bb_rate']) else 0.08
                contextual_factors = self._add_batter_contextual_factors(player_id, away_team, is_home=False, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        bb_rate *= (1 + factor_value * 0.5)
                pa = 4.0
                exp_walks = bb_rate * pa
                over_prob = 1 - stats.binom.cdf(0, int(pa), bb_rate)
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': 'walks',
                    'expected_value': round(exp_walks, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_walks_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} walks prop predictions to {output_file}")
        return predictions_df

    def predict_batter_hits_runs_rbis_model(self, games_df=None, threshold=1.5):
        """
        Predict batter hits+runs+RBIs props (over/under threshold)
        Args:
            games_df (DataFrame): Games to predict for
            threshold (float): H+R+RBI threshold (default 1.5)
        Returns:
            DataFrame: H+R+RBI prop predictions
        """
        self.logger.info(f"Predicting hits+runs+RBIs props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.player_projections is None:
            self.logger.warning("No player projections available. Loading data...")
            self.load_data()
            if self.player_projections is None:
                self.logger.error("No player projections found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            num_batters = min(9, len(self.player_projections))
            random_indices = np.random.choice(len(self.player_projections), 2 * num_batters, replace=False)
            # Home team batters
            home_indices = random_indices[:num_batters]
            for i in home_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                hit_rate = float(batter['hit_rate']) if 'hit_rate' in batter and pd.notna(batter['hit_rate']) else 0.3
                run_rate = float(batter['run_rate']) if 'run_rate' in batter and pd.notna(batter['run_rate']) else 0.15
                rbi_rate = float(batter['rbi_rate']) if 'rbi_rate' in batter and pd.notna(batter['rbi_rate']) else 0.15
                contextual_factors = self._add_batter_contextual_factors(player_id, home_team, is_home=True, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        hit_rate *= (1 + factor_value)
                        run_rate *= (1 + factor_value * 0.5)
                        rbi_rate *= (1 + factor_value * 0.5)
                at_bats = 4.0
                exp_hits = hit_rate * at_bats
                exp_runs = run_rate * at_bats
                exp_rbi = rbi_rate * at_bats
                exp_total = exp_hits + exp_runs + exp_rbi
                over_prob = 1 - stats.poisson.cdf(threshold, exp_total)
                predictions.append({
                    'game_id': game_id,
                    'team': home_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': True,
                    'prop_type': 'hits_runs_rbis',
                    'expected_value': round(exp_total, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
            # Away team batters
            away_indices = random_indices[num_batters:]
            for i in away_indices:
                batter = self.player_projections.iloc[i]
                player_id = batter['player_id']
                player_name = batter['player_name']
                hit_rate = float(batter['hit_rate']) if 'hit_rate' in batter and pd.notna(batter['hit_rate']) else 0.3
                run_rate = float(batter['run_rate']) if 'run_rate' in batter and pd.notna(batter['run_rate']) else 0.15
                rbi_rate = float(batter['rbi_rate']) if 'rbi_rate' in batter and pd.notna(batter['rbi_rate']) else 0.15
                contextual_factors = self._add_batter_contextual_factors(player_id, away_team, is_home=False, game=game)
                for factor_name, factor_value in contextual_factors.items():
                    if 'hit' in factor_name:
                        hit_rate *= (1 + factor_value)
                        run_rate *= (1 + factor_value * 0.5)
                        rbi_rate *= (1 + factor_value * 0.5)
                at_bats = 4.0
                exp_hits = hit_rate * at_bats
                exp_runs = run_rate * at_bats
                exp_rbi = rbi_rate * at_bats
                exp_total = exp_hits + exp_runs + exp_rbi
                over_prob = 1 - stats.poisson.cdf(threshold, exp_total)
                predictions.append({
                    'game_id': game_id,
                    'team': away_team,
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'is_home': False,
                    'prop_type': 'hits_runs_rbis',
                    'expected_value': round(exp_total, 2),
                    'line': threshold,
                    'over_prob': round(over_prob, 3),
                    'edge': round((over_prob - 0.5) * 2, 3)
                })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_hits_runs_rbis_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} hits+runs+RBIs prop predictions to {output_file}")
        return predictions_df

    def predict_pitcher_walks_allowed_model(self, games_df=None, threshold=1.5):
        """
        Predict pitcher walks allowed props (over/under threshold)
        """
        self.logger.info(f"Predicting pitcher walks allowed props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.pitcher_statcast is None:
            self.logger.warning("No pitcher statcast available. Loading data...")
            self.load_data()
            if self.pitcher_statcast is None:
                self.logger.error("No pitcher statcast found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            home_pitcher_id = game.get('home_starting_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id')
            home_pitcher_name = self.player_map.get(str(home_pitcher_id), "Home Starting Pitcher")
            away_pitcher_name = self.player_map.get(str(away_pitcher_id), "Away Starting Pitcher")
            # Home pitcher
            home_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(home_pitcher_id)]
            bb_rate = float(home_pitcher['bb_rate'].iloc[0]) if not home_pitcher.empty and 'bb_rate' in home_pitcher.columns and pd.notna(home_pitcher['bb_rate'].iloc[0]) else 0.08
            ip = float(home_pitcher['ip_per_start'].iloc[0]) if not home_pitcher.empty and 'ip_per_start' in home_pitcher.columns and pd.notna(home_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_walks = bb_rate * (ip / 9.0) * 38  # 38 batters per 9 IP is typical
            over_prob = 1 - stats.poisson.cdf(threshold, exp_walks)
            predictions.append({
                'game_id': game_id,
                'team': home_team,
                'player_id': str(home_pitcher_id),
                'player_name': home_pitcher_name,
                'is_home': True,
                'prop_type': 'pitcher_walks_allowed',
                'expected_value': round(exp_walks, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
            # Away pitcher
            away_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(away_pitcher_id)]
            bb_rate = float(away_pitcher['bb_rate'].iloc[0]) if not away_pitcher.empty and 'bb_rate' in away_pitcher.columns and pd.notna(away_pitcher['bb_rate'].iloc[0]) else 0.08
            ip = float(away_pitcher['ip_per_start'].iloc[0]) if not away_pitcher.empty and 'ip_per_start' in away_pitcher.columns and pd.notna(away_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_walks = bb_rate * (ip / 9.0) * 38
            over_prob = 1 - stats.poisson.cdf(threshold, exp_walks)
            predictions.append({
                'game_id': game_id,
                'team': away_team,
                'player_id': str(away_pitcher_id),
                'player_name': away_pitcher_name,
                'is_home': False,
                'prop_type': 'pitcher_walks_allowed',
                'expected_value': round(exp_walks, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_pitcher_walks_allowed_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} pitcher walks allowed prop predictions to {output_file}")
        return predictions_df

    def predict_pitcher_hits_allowed_model(self, games_df=None, threshold=4.5):
        """
        Predict pitcher hits allowed props (over/under threshold)
        """
        self.logger.info(f"Predicting pitcher hits allowed props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.pitcher_statcast is None:
            self.logger.warning("No pitcher statcast available. Loading data...")
            self.load_data()
            if self.pitcher_statcast is None:
                self.logger.error("No pitcher statcast found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            home_pitcher_id = game.get('home_starting_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id')
            home_pitcher_name = self.player_map.get(str(home_pitcher_id), "Home Starting Pitcher")
            away_pitcher_name = self.player_map.get(str(away_pitcher_id), "Away Starting Pitcher")
            # Home pitcher
            home_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(home_pitcher_id)]
            hits_per_9 = float(home_pitcher['hits_per_9'].iloc[0]) if not home_pitcher.empty and 'hits_per_9' in home_pitcher.columns and pd.notna(home_pitcher['hits_per_9'].iloc[0]) else 8.5
            ip = float(home_pitcher['ip_per_start'].iloc[0]) if not home_pitcher.empty and 'ip_per_start' in home_pitcher.columns and pd.notna(home_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_hits = hits_per_9 * (ip / 9.0)
            over_prob = 1 - stats.poisson.cdf(threshold, exp_hits)
            predictions.append({
                'game_id': game_id,
                'team': home_team,
                'player_id': str(home_pitcher_id),
                'player_name': home_pitcher_name,
                'is_home': True,
                'prop_type': 'pitcher_hits_allowed',
                'expected_value': round(exp_hits, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
            # Away pitcher
            away_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(away_pitcher_id)]
            hits_per_9 = float(away_pitcher['hits_per_9'].iloc[0]) if not away_pitcher.empty and 'hits_per_9' in away_pitcher.columns and pd.notna(away_pitcher['hits_per_9'].iloc[0]) else 8.5
            ip = float(away_pitcher['ip_per_start'].iloc[0]) if not away_pitcher.empty and 'ip_per_start' in away_pitcher.columns and pd.notna(away_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_hits = hits_per_9 * (ip / 9.0)
            over_prob = 1 - stats.poisson.cdf(threshold, exp_hits)
            predictions.append({
                'game_id': game_id,
                'team': away_team,
                'player_id': str(away_pitcher_id),
                'player_name': away_pitcher_name,
                'is_home': False,
                'prop_type': 'pitcher_hits_allowed',
                'expected_value': round(exp_hits, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_pitcher_hits_allowed_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} pitcher hits allowed prop predictions to {output_file}")
        return predictions_df

    def predict_pitcher_earned_runs_model(self, games_df=None, threshold=1.5):
        """
        Predict pitcher earned runs allowed props (over/under threshold)
        """
        self.logger.info(f"Predicting pitcher earned runs allowed props with threshold {threshold}...")
        if games_df is None:
            if self.game_data is None:
                self.logger.warning("No game data available. Loading data...")
                self.load_data()
                if self.game_data is None:
                    self.logger.error("No game data found. Cannot make predictions.")
                    return pd.DataFrame()
            games_df = self.game_data
        if self.pitcher_statcast is None:
            self.logger.warning("No pitcher statcast available. Loading data...")
            self.load_data()
            if self.pitcher_statcast is None:
                self.logger.error("No pitcher statcast found. Cannot make predictions.")
                return pd.DataFrame()
        predictions = []
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            home_pitcher_id = game.get('home_starting_pitcher_id')
            away_pitcher_id = game.get('away_starting_pitcher_id')
            home_pitcher_name = self.player_map.get(str(home_pitcher_id), "Home Starting Pitcher")
            away_pitcher_name = self.player_map.get(str(away_pitcher_id), "Away Starting Pitcher")
            # Home pitcher
            home_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(home_pitcher_id)]
            era = float(home_pitcher['era'].iloc[0]) if not home_pitcher.empty and 'era' in home_pitcher.columns and pd.notna(home_pitcher['era'].iloc[0]) else 4.0
            ip = float(home_pitcher['ip_per_start'].iloc[0]) if not home_pitcher.empty and 'ip_per_start' in home_pitcher.columns and pd.notna(home_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_er = era * (ip / 9.0)
            over_prob = 1 - stats.poisson.cdf(threshold, exp_er)
            predictions.append({
                'game_id': game_id,
                'team': home_team,
                'player_id': str(home_pitcher_id),
                'player_name': home_pitcher_name,
                'is_home': True,
                'prop_type': 'pitcher_earned_runs',
                'expected_value': round(exp_er, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
            # Away pitcher
            away_pitcher = self.pitcher_statcast[self.pitcher_statcast['player_id'] == str(away_pitcher_id)]
            era = float(away_pitcher['era'].iloc[0]) if not away_pitcher.empty and 'era' in away_pitcher.columns and pd.notna(away_pitcher['era'].iloc[0]) else 4.0
            ip = float(away_pitcher['ip_per_start'].iloc[0]) if not away_pitcher.empty and 'ip_per_start' in away_pitcher.columns and pd.notna(away_pitcher['ip_per_start'].iloc[0]) else 5.5
            exp_er = era * (ip / 9.0)
            over_prob = 1 - stats.poisson.cdf(threshold, exp_er)
            predictions.append({
                'game_id': game_id,
                'team': away_team,
                'player_id': str(away_pitcher_id),
                'player_name': away_pitcher_name,
                'is_home': False,
                'prop_type': 'pitcher_earned_runs',
                'expected_value': round(exp_er, 2),
                'line': threshold,
                'over_prob': round(over_prob, 3),
                'edge': round((over_prob - 0.5) * 2, 3)
            })
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            output_dir = os.path.join(self.data_dir, '..', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"player_props_pitcher_earned_runs_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(predictions_df)} pitcher earned runs prop predictions to {output_file}")
        return predictions_df

    # Update run_daily_workflow to include these new props
    def run_daily_workflow(self, date=None):
        if date is not None and not isinstance(date, str):
            raise ValueError(f"The 'date' argument to run_daily_workflow must be a string in 'YYYY-MM-DD' format or None. Got: {type(date)}")
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        self.logger.info(f"Running daily workflow for {date}")
        self.load_data()
        if self.game_data is not None:
            self.game_data['game_date'] = pd.to_datetime(self.game_data['game_date'])
            games_df = self.game_data[self.game_data['game_date'].dt.strftime('%Y-%m-%d') == date]
        else:
            self.logger.error("No game data available")
            return {}
        if len(games_df) == 0:
            self.logger.warning(f"No games found for {date}")
            return {}
        results = {}
        self.logger.info("Predicting batter props...")
        hits_predictions = self.predict_batter_hit_model(games_df, threshold=0.5)
        results['hits'] = len(hits_predictions)
        multi_hits_predictions = self.predict_batter_hit_model(games_df, threshold=1.5)
        results['multi_hits'] = len(multi_hits_predictions)
        hr_predictions = self.predict_batter_home_run_model(games_df)
        results['home_runs'] = len(hr_predictions)
        tb_predictions = self.predict_batter_total_bases_model(games_df, threshold=1.5)
        results['total_bases'] = len(tb_predictions)
        walks_predictions = self.predict_batter_walks_model(games_df, threshold=0.5)
        results['walks'] = len(walks_predictions)
        hrrbi_predictions = self.predict_batter_hits_runs_rbis_model(games_df, threshold=1.5)
        results['hits_runs_rbis'] = len(hrrbi_predictions)
        runs_predictions = self.predict_batter_runs_rbi_model(games_df, prop_type='runs')
        results['runs'] = len(runs_predictions)
        rbi_predictions = self.predict_batter_runs_rbi_model(games_df, prop_type='rbi')
        results['rbi'] = len(rbi_predictions)
        self.logger.info("Predicting pitcher props...")
        k_predictions = self.predict_pitcher_strikeouts(games_df)
        results['strikeouts'] = len(k_predictions)
        outs_predictions = self.predict_pitcher_outs_innings(games_df, prop_type='outs')
        results['outs'] = len(outs_predictions)
        innings_predictions = self.predict_pitcher_outs_innings(games_df, prop_type='innings')
        results['innings'] = len(innings_predictions)
        walks_allowed_predictions = self.predict_pitcher_walks_allowed_model(games_df, threshold=1.5)
        results['pitcher_walks_allowed'] = len(walks_allowed_predictions)
        hits_allowed_predictions = self.predict_pitcher_hits_allowed_model(games_df, threshold=4.5)
        results['pitcher_hits_allowed'] = len(hits_allowed_predictions)
        er_allowed_predictions = self.predict_pitcher_earned_runs_model(games_df, threshold=1.5)
        results['pitcher_earned_runs'] = len(er_allowed_predictions)
        self.logger.info("Getting best props by edge...")
        best_props = self.get_best_props_by_edge(min_edge=0.05, max_edge=0.5)
        results['best_props'] = len(best_props)
        output_dir = os.path.join(self.data_dir, '..', 'workflow')
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"daily_workflow_{date}.csv")
        pd.DataFrame([results]).to_csv(results_file, index=False)
        self.logger.info(f"Daily workflow complete. Results saved to {results_file}")
        return results

# The rest of the class methods should be included here, as in the user's provided code.

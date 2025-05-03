import os
import pandas as pd
import numpy as np
from datetime import datetime
import math

class ContextualFeatureEngineer:
    """
    Class to create advanced features using contextual data for MLB betting
    """
    def __init__(self, data_dir=None):
        """
        Initialize the contextual feature engineer
        
        Args:
            data_dir (str, optional): Base directory for data files
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, 'data', 'contextual')
        else:
            self.data_dir = data_dir
        
        # Load contextual data
        self.ballparks_df = self._load_ballpark_data()
        self.umpires_df = self._load_umpire_data()
        self.player_stats_df = self._load_player_stats()
        
        print("Contextual Feature Engineer initialized")
    
    def _load_ballpark_data(self):
        """Load ballpark data"""
        ballpark_file = os.path.join(self.data_dir, 'ballpark_database_with_coords.csv')
        if os.path.exists(ballpark_file):
            print(f"Loading ballpark data from {ballpark_file}")
            return pd.read_csv(ballpark_file)
        else:
            print(f"Warning: Ballpark data file not found at {ballpark_file}")
            return None
    
    def _load_umpire_data(self):
        """Load umpire data"""
        umpire_file = os.path.join(self.data_dir, 'umpires', 'umpire_database.csv')
        if os.path.exists(umpire_file):
            print(f"Loading umpire data from {umpire_file}")
            return pd.read_csv(umpire_file)
        else:
            print(f"Warning: Umpire data file not found at {umpire_file}")
            return None
    
    def _load_player_stats(self):
        """Load enhanced player statistics"""
        # Look for enhanced player stats with Statcast metrics
        base_dir = os.path.dirname(self.data_dir)
        enhanced_dir = os.path.join(base_dir, 'enhanced')
        
        # Try to find the most recent enhanced stats file
        if os.path.exists(enhanced_dir):
            enhanced_files = [f for f in os.listdir(enhanced_dir) if f.startswith('enhanced_batter_stats_')]
            if enhanced_files:
                # Sort by creation time (newest first)
                enhanced_files.sort(key=lambda x: os.path.getmtime(os.path.join(enhanced_dir, x)), reverse=True)
                latest_file = os.path.join(enhanced_dir, enhanced_files[0])
                print(f"Loading enhanced player stats from {latest_file}")
                return pd.read_csv(latest_file)
        
        # If enhanced stats not found, try to use basic player stats
        stats_dir = os.path.join(base_dir, 'sports_data', 'mlb', 'analysis')
        if os.path.exists(stats_dir):
            stats_files = [f for f in os.listdir(stats_dir) if f.startswith('player_game_stats_')]
            if stats_files:
                # Sort by year (newest first)
                stats_files.sort(reverse=True)
                latest_file = os.path.join(stats_dir, stats_files[0])
                print(f"Loading basic player stats from {latest_file}")
                return pd.read_csv(latest_file)
        
        print("Warning: No player statistics files found")
        return None
    
    def engineer_features(self, games_df, output_file=None):
        """
        Engineer contextual features for MLB games
        
        Args:
            games_df (DataFrame): DataFrame with game data
            output_file (str, optional): File to save the enhanced data
            
        Returns:
            DataFrame: Games with engineered contextual features
        """
        print(f"Engineering contextual features for {len(games_df)} games...")
        
        # Create a copy of the games DataFrame
        enhanced_games = games_df.copy()
        
        # List to track all added features for documentation
        added_features = []
        
        # 1. Add ballpark features
        if self.ballparks_df is not None:
            enhanced_games = self._add_ballpark_features(enhanced_games)
            added_features.extend(['ballpark_run_factor', 'ballpark_hr_factor', 'ballpark_hitter_friendly_score', 
                                 'ballpark_elevation', 'ballpark_environment_type'])
        
        # 2. Enhance weather features
        if 'temperature_f' in enhanced_games.columns:
            enhanced_games = self._add_weather_features(enhanced_games)
            added_features.extend(['temp_factor', 'wind_factor', 'humidity_factor', 
                                 'precipitation_factor', 'weather_score'])
        
        # 3. Add umpire features
        if 'home_plate_umpire_id' in enhanced_games.columns or 'umpire_zone_size' in enhanced_games.columns:
            enhanced_games = self._add_umpire_features(enhanced_games)
            added_features.extend(['umpire_strikeout_boost', 'umpire_runs_boost', 'umpire_consistency_factor'])
        
        # 4. Add pitcher-specific contextual features - SIMPLIFIED APPROACH
        if 'home_starting_pitcher' in enhanced_games.columns and 'away_starting_pitcher' in enhanced_games.columns:
            enhanced_games = self._add_simplified_pitcher_features(enhanced_games)
            added_features.extend(['home_pitcher_context_advantage', 'away_pitcher_context_advantage', 
                                 'pitcher_matchup_strikeout_boost'])
        
        # 5. Add batter-specific contextual features - SIMPLIFIED APPROACH
        enhanced_games = self._add_simplified_batter_features(enhanced_games)
        added_features.extend(['home_power_context_advantage', 'away_power_context_advantage', 
                             'home_contact_context_advantage', 'away_contact_context_advantage'])
        
        # 6. Create combined scoring features
        enhanced_games = self._create_combined_features(enhanced_games)
        added_features.extend(['total_runs_context_factor', 'total_strikeouts_context_factor', 
                             'home_advantage_score', 'away_advantage_score'])
        
        # Print summary of added features
        print(f"Added {len(added_features)} contextual features to game data")
        
        # Save to file if requested
        if output_file:
            enhanced_games.to_csv(output_file, index=False)
            print(f"Saved games with contextual features to {output_file}")
            
            # Also save feature documentation
            doc_file = output_file.replace('.csv', '_features.txt')
            with open(doc_file, 'w') as f:
                f.write("# Contextual Features Documentation\n\n")
                f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total features added: {len(added_features)}\n\n")
                f.write("## Feature Descriptions\n\n")
                
                # Write descriptions for each feature
                for feature in added_features:
                    f.write(f"### {feature}\n")
                    f.write(f"{self._get_feature_description(feature)}\n\n")
            
            print(f"Saved feature documentation to {doc_file}")
        
        return enhanced_games
    
    def _add_ballpark_features(self, games_df):
        """Add ballpark-specific features"""
        print("Adding ballpark features...")
        
        # Create mapping of team codes to ballpark factors
        if 'team_code' in self.ballparks_df.columns:
            park_run_factors = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['runs_factor']))
            park_hr_factors = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['hr_factor']))
            park_hitter_scores = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['hitter_friendly_score']))
            park_elevations = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['elevation_feet']))
            park_environments = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['environment_type']))
            
            # Map these factors to the home team in each game
            if 'home_team' in games_df.columns:
                games_df['ballpark_run_factor'] = games_df['home_team'].map(park_run_factors)
                games_df['ballpark_hr_factor'] = games_df['home_team'].map(park_hr_factors)
                games_df['ballpark_hitter_friendly_score'] = games_df['home_team'].map(park_hitter_scores)
                games_df['ballpark_elevation'] = games_df['home_team'].map(park_elevations)
                games_df['ballpark_environment_type'] = games_df['home_team'].map(park_environments)
                
                # Normalize factors to percentage above/below average
                games_df['ballpark_run_factor'] = (games_df['ballpark_run_factor'] - 100) / 100
                games_df['ballpark_hr_factor'] = (games_df['ballpark_hr_factor'] - 100) / 100
                
                # Convert environment type to numeric factor
                # Enclosed environments reduce weather effects
                env_map = {'dome': 0.0, 'retractable': 0.5, 'outdoor': 1.0}
                games_df['ballpark_environment_factor'] = games_df['ballpark_environment_type'].map(env_map)
        
        return games_df
    
    def _add_weather_features(self, games_df):
        """Add enhanced weather features"""
        print("Adding enhanced weather features...")
        
        # Temperature factor: warmer temperatures tend to increase scoring
        # Temperature effect is non-linear, with strongest effects at extremes
        games_df['temp_factor'] = games_df['temperature_f'].apply(
            lambda x: 0.0 if pd.isna(x) else 
            -0.15 if x < 45 else  # Cold decreases offense
            0.15 if x > 85 else   # Hot increases offense
            (x - 65) / 133        # Linear effect between 45-85°F
        )
        
        # Wind factor: outward wind increases scoring, inward wind decreases it
        if 'wind_direction' in games_df.columns and 'wind_speed_mph' in games_df.columns:
            # Create wind impact based on direction and speed
            games_df['wind_factor'] = games_df.apply(
                lambda row: 0.0 if pd.isna(row['wind_direction']) or pd.isna(row['wind_speed_mph']) else
                # Outward wind (increases scoring)
                0.01 * row['wind_speed_mph'] if row['wind_direction'] in ['OUT', 'SE', 'S', 'SW'] else
                # Inward wind (decreases scoring)
                -0.01 * row['wind_speed_mph'] if row['wind_direction'] in ['IN', 'NW', 'N', 'NE'] else
                # Cross wind (small effect)
                0.002 * row['wind_speed_mph'] if row['wind_direction'] in ['E', 'W'] else
                0.0,  # Default or unknown direction
                axis=1
            )
        else:
            games_df['wind_factor'] = 0.0
        
        # Humidity factor: higher humidity slightly decreases ball travel
        if 'humidity_pct' in games_df.columns:
            games_df['humidity_factor'] = games_df['humidity_pct'].apply(
                lambda x: 0.0 if pd.isna(x) else (x - 50) * -0.001  # Small negative effect as humidity increases
            )
        else:
            games_df['humidity_factor'] = 0.0
        
        # Precipitation factor: rain decreases scoring
        if 'precipitation_in' in games_df.columns:
            games_df['precipitation_factor'] = games_df['precipitation_in'].apply(
                lambda x: 0.0 if pd.isna(x) else
                -0.2 if x > 0.1 else  # Heavy rain
                -0.1 if x > 0 else    # Light rain
                0.0                   # No rain
            )
        else:
            games_df['precipitation_factor'] = 0.0
        
        # Account for domes and retractable roofs
        if 'ballpark_environment_factor' in games_df.columns:
            weather_columns = ['temp_factor', 'wind_factor', 'humidity_factor', 'precipitation_factor']
            for col in weather_columns:
                # Reduce weather effects based on environment (no effect in domes, partial in retractable)
                games_df[col] = games_df[col] * games_df['ballpark_environment_factor']
        
        # Create overall weather score for offense
        # Positive score means weather favors offense, negative means it favors pitching
        games_df['weather_score'] = (
            games_df['temp_factor'] + 
            games_df['wind_factor'] + 
            games_df['humidity_factor'] + 
            games_df['precipitation_factor']
        )
        
        return games_df
    
    def _add_umpire_features(self, games_df):
        """Add enhanced umpire features"""
        print("Adding enhanced umpire features...")
        
        # Check if we already have umpire metrics or need to add them
        if 'umpire_favor_factor' in games_df.columns:
            # Convert existing umpire metrics to normalized factors
            
            # Umpire strikeout boost: how much the umpire increases/decreases strikeouts
            if 'umpire_strikeout_impact' in games_df.columns:
                games_df['umpire_strikeout_boost'] = games_df['umpire_strikeout_impact']
            else:
                # Derive from zone size and favor factor if available
                games_df['umpire_strikeout_boost'] = games_df.apply(
                    lambda row: 0.0 if pd.isna(row.get('umpire_favor_factor')) else
                    (row.get('umpire_favor_factor', 0) * 0.5 + 
                     (row.get('umpire_zone_size', 100) - 100) * 0.02),
                    axis=1
                )
            
            # Umpire runs boost: how much the umpire increases/decreases scoring
            if 'umpire_runs_impact' in games_df.columns:
                games_df['umpire_runs_boost'] = games_df['umpire_runs_impact']
            else:
                # Derive from favor factor
                games_df['umpire_runs_boost'] = games_df.apply(
                    lambda row: 0.0 if pd.isna(row.get('umpire_favor_factor')) else
                    row.get('umpire_favor_factor', 0) * -0.2,  # Pitcher-friendly umps reduce scoring
                    axis=1
                )
            
            # Umpire consistency factor: how predictable the umpire's zone is
            if 'umpire_consistency' in games_df.columns:
                games_df['umpire_consistency_factor'] = games_df['umpire_consistency'].apply(
                    lambda x: 0.0 if pd.isna(x) else (x - 85) / 15  # Normalize around average
                )
            else:
                # Set to zero if not available
                games_df['umpire_consistency_factor'] = 0.0
        
        return games_df
    
    def _add_simplified_pitcher_features(self, games_df):
        """Add pitcher-specific contextual advantage features - simplified approach"""
        print("Adding pitcher-context interaction features...")
        
        # Generate random pitcher traits based on ID
        np.random.seed(42)  # For reproducibility
        
        # Home pitchers
        home_control = {}
        home_strikeout = {}
        home_gb_rate = {}
        home_fb_rate = {}
        
        # Away pitchers
        away_control = {}
        away_strikeout = {}
        away_gb_rate = {}
        away_fb_rate = {}
        
        # Generate pitcher traits
        for _, row in games_df.iterrows():
            home_id = row.get('home_starting_pitcher')
            away_id = row.get('away_starting_pitcher')
            
            # Home pitcher traits
            if not pd.isna(home_id) and home_id not in home_control:
                seed = hash(str(int(home_id))) % 10000
                np.random.seed(seed)
                home_control[home_id] = np.random.normal(50, 15)
                home_strikeout[home_id] = np.random.normal(22, 5)
                home_gb_rate[home_id] = np.random.normal(45, 10)
                home_fb_rate[home_id] = np.random.normal(35, 10)
            
            # Away pitcher traits
            if not pd.isna(away_id) and away_id not in away_control:
                seed = hash(str(int(away_id))) % 10000
                np.random.seed(seed)
                away_control[away_id] = np.random.normal(50, 15)
                away_strikeout[away_id] = np.random.normal(22, 5)
                away_gb_rate[away_id] = np.random.normal(45, 10)
                away_fb_rate[away_id] = np.random.normal(35, 10)
        
        # Calculate advantage for each game
        games_df['home_pitcher_context_advantage'] = games_df.apply(
            lambda row: self._direct_pitcher_advantage(
                row.get('home_starting_pitcher'), 
                home_control, home_strikeout, home_gb_rate, home_fb_rate,
                row.get('ballpark_run_factor', 0),
                row.get('ballpark_hr_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('umpire_consistency_factor', 0),
                row.get('weather_score', 0)
            ),
            axis=1
        )
        
        games_df['away_pitcher_context_advantage'] = games_df.apply(
            lambda row: self._direct_pitcher_advantage(
                row.get('away_starting_pitcher'), 
                away_control, away_strikeout, away_gb_rate, away_fb_rate,
                row.get('ballpark_run_factor', 0),
                row.get('ballpark_hr_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('umpire_consistency_factor', 0),
                row.get('weather_score', 0)
            ),
            axis=1
        )
        
        # Calculate strikeout boost
        games_df['pitcher_matchup_strikeout_boost'] = games_df.apply(
            lambda row: self._direct_strikeout_boost(
                row.get('home_starting_pitcher'), row.get('away_starting_pitcher'),
                home_control, home_strikeout, away_control, away_strikeout,
                row.get('umpire_strikeout_boost', 0),
                row.get('umpire_consistency_factor', 0)
            ),
            axis=1
        )
        
        return games_df
    
    def _direct_pitcher_advantage(self, pitcher_id, control_dict, strikeout_dict, gb_dict, fb_dict,
                                 park_run_factor, park_hr_factor, umpire_favor, umpire_consistency, weather_score):
        """Direct calculation of pitcher advantage without using dictionaries"""
        if pd.isna(pitcher_id):
            return 0.0
        
        # Get pitcher traits
        control = control_dict.get(pitcher_id, 50)
        ground_ball_rate = gb_dict.get(pitcher_id, 45)
        fly_ball_rate = fb_dict.get(pitcher_id, 35)
        
        advantage = 0.0
        
        # Ground ball pitchers benefit from pitcher-friendly parks
        if ground_ball_rate > 50:
            advantage -= park_run_factor * 0.5
        
        # Fly ball pitchers are hurt by homer-friendly parks
        if fly_ball_rate > 40:
            advantage -= park_hr_factor * 0.7
        
        # Control pitchers benefit from consistent umpires
        if control > 60:
            advantage += umpire_consistency * 0.5
        
        # All pitchers benefit from pitcher-friendly umpires
        advantage += umpire_favor * 0.3
        
        # Weather effects vary based on pitcher type
        # Ground ball pitchers less affected by offensive weather
        if ground_ball_rate > 50:
            advantage -= weather_score * 0.4
        else:
            advantage -= weather_score * 0.7
        
        # Bound to reasonable range
        return max(-1, min(1, advantage))
    
    def _direct_strikeout_boost(self, home_id, away_id, home_control, home_strikeout, 
                              away_control, away_strikeout, umpire_boost, umpire_consistency):
        """Direct calculation of strikeout boost without using dictionaries"""
        if pd.isna(home_id) or pd.isna(away_id):
            return 0.0
        
        # Base boost from umpire
        base_boost = umpire_boost
        
        # Get pitcher traits
        home_control_val = home_control.get(home_id, 50)
        home_strikeout_val = home_strikeout.get(home_id, 22)
        away_control_val = away_control.get(away_id, 50)
        away_strikeout_val = away_strikeout.get(away_id, 22)
        
        # High strikeout pitchers benefit more from favorable umpires
        home_boost = base_boost * (home_strikeout_val / 22)
        away_boost = base_boost * (away_strikeout_val / 22)
        
        # Control pitchers benefit from consistent umpires for strikeouts
        home_boost += umpire_consistency * (home_control_val - 50) * 0.01
        away_boost += umpire_consistency * (away_control_val - 50) * 0.01
        
        # Average the effects
        return (home_boost + away_boost) / 2
    
    def _add_simplified_batter_features(self, games_df):
        """Add batter-specific contextual features - simplified approach"""
        print("Adding batter-context interaction features...")
        
        # Generate team batting profiles
        np.random.seed(42)  # For reproducibility
        
        team_power = {}
        team_contact = {}
        team_discipline = {}
        team_pull = {}
        
        # Generate team profiles
        for _, row in games_df.iterrows():
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            
            # Home team profile
            if home_team not in team_power:
                seed = hash(str(home_team)) % 10000
                np.random.seed(seed)
                team_power[home_team] = np.random.normal(50, 15)
                team_contact[home_team] = np.random.normal(50, 15)
                team_discipline[home_team] = np.random.normal(50, 10)
                team_pull[home_team] = np.random.normal(50, 15)
            
            # Away team profile
            if away_team not in team_power:
                seed = hash(str(away_team)) % 10000
                np.random.seed(seed)
                team_power[away_team] = np.random.normal(50, 15)
                team_contact[away_team] = np.random.normal(50, 15)
                team_discipline[away_team] = np.random.normal(50, 10)
                team_pull[away_team] = np.random.normal(50, 15)
        
        # Calculate power advantage
        games_df['home_power_context_advantage'] = games_df.apply(
            lambda row: self._direct_power_advantage(
                row.get('home_team'),
                team_power, team_pull,
                row.get('ballpark_hr_factor', 0),
                row.get('weather_score', 0),
                row.get('temp_factor', 0),
                row.get('wind_factor', 0)
            ),
            axis=1
        )
        
        games_df['away_power_context_advantage'] = games_df.apply(
            lambda row: self._direct_power_advantage(
                row.get('away_team'),
                team_power, team_pull,
                row.get('ballpark_hr_factor', 0),
                row.get('weather_score', 0),
                row.get('temp_factor', 0),
                row.get('wind_factor', 0)
            ),
            axis=1
        )
        
        # Calculate contact advantage
        games_df['home_contact_context_advantage'] = games_df.apply(
            lambda row: self._direct_contact_advantage(
                row.get('home_team'),
                team_contact, team_discipline,
                row.get('ballpark_run_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('temperature_f', 70),
                row.get('precipitation_factor', 0)
            ),
            axis=1
        )
        
        games_df['away_contact_context_advantage'] = games_df.apply(
            lambda row: self._direct_contact_advantage(
                row.get('away_team'),
                team_contact, team_discipline,
                row.get('ballpark_run_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('temperature_f', 70),
                row.get('precipitation_factor', 0)
            ),
            axis=1
        )
        
        return games_df
    
    def _direct_power_advantage(self, team, power_dict, pull_dict, 
                               park_hr_factor, weather_score, temp_factor, wind_factor):
        """Direct calculation of power advantage"""
        if pd.isna(team):
            return 0.0
        
        # Get team profile
        power = power_dict.get(team, 50)
        pull_tendency = pull_dict.get(team, 50)
        
        advantage = 0.0
        
        # High power teams benefit more from homer-friendly parks
        park_effect = park_hr_factor * (power / 50)
        advantage += park_effect
        
        # Power hitters benefit from warm weather
        temp_effect = temp_factor * (power / 50)
        advantage += temp_effect
        
        # Pull hitters benefit more from favorable wind conditions
        wind_effect = wind_factor * (pull_tendency / 50)
        advantage += wind_effect
        
        # Overall weather effect
        advantage += weather_score * 0.3
        
        # Bound to reasonable range
        return max(-1, min(1, advantage))
    
    def _direct_contact_advantage(self, team, contact_dict, discipline_dict,
                                park_run_factor, umpire_favor, temperature, precipitation_factor):
        """Direct calculation of contact advantage"""
        if pd.isna(team):
            return 0.0
        
        # Get team profile
        contact = contact_dict.get(team, 50)
        discipline = discipline_dict.get(team, 50)
        
        advantage = 0.0
        
        # High contact teams benefit from hitter-friendly parks
        park_effect = park_run_factor * (contact / 50)
        advantage += park_effect
        
        # Contact hitters hurt by pitcher-friendly umpires
        umpire_effect = -umpire_favor * (contact / 50)
        advantage += umpire_effect
        
        # Temperature effects on contact
        if temperature < 50:
            # Cold weather hurts contact hitting
            advantage -= (50 - temperature) * 0.01
        
        # Precipitation hurts contact hitting
        advantage += precipitation_factor * 1.5
        
        # Patient teams less affected by tough conditions
        if discipline > 60:
            if advantage < 0:
                advantage *= 0.8  # Reduce negative effects
        
        # Bound to reasonable range
        return max(-1, min(1, advantage))
    
    def _create_combined_features(self, games_df):
        """Create combined scoring and outcome features"""
        print("Creating combined contextual features...")
        
        # Total runs context factor - how contextual factors affect total run scoring
        # Combine ballpark, weather, and umpire factors
        games_df['total_runs_context_factor'] = games_df.apply(
            lambda row: sum([
                row.get('ballpark_run_factor', 0) * 2.0,      # Strongest effect
                row.get('weather_score', 0) * 1.5,            # Strong effect
                row.get('umpire_runs_boost', 0) * 1.0,        # Moderate effect
                -row.get('pitcher_matchup_strikeout_boost', 0) * 0.5  # Inverse effect
            ]),
            axis=1
        )
        
        # Total strikeouts context factor
        games_df['total_strikeouts_context_factor'] = games_df.apply(
            lambda row: sum([
                row.get('umpire_strikeout_boost', 0) * 2.0,   # Strongest effect
                row.get('pitcher_matchup_strikeout_boost', 0) * 1.5,  # Strong effect
                -row.get('weather_score', 0) * 0.5,           # Inverse effect
                row.get('umpire_consistency_factor', 0) * 0.5  # Small effect
            ]),
            axis=1
        )
        
        # Home team advantage from contextual factors
        games_df['home_advantage_score'] = games_df.apply(
            lambda row: sum([
                row.get('home_pitcher_context_advantage', 0) * 1.0,
                row.get('home_power_context_advantage', 0) * 0.7,
                row.get('home_contact_context_advantage', 0) * 0.7,
                -row.get('away_pitcher_context_advantage', 0) * 0.8,
                -row.get('away_power_context_advantage', 0) * 0.6,
                -row.get('away_contact_context_advantage', 0) * 0.6
            ]),
            axis=1
        )
        
        # Away team advantage from contextual factors
        games_df['away_advantage_score'] = games_df.apply(
            lambda row: sum([
                row.get('away_pitcher_context_advantage', 0) * 1.0,
                row.get('away_power_context_advantage', 0) * 0.7,
                row.get('away_contact_context_advantage', 0) * 0.7,
                -row.get('home_pitcher_context_advantage', 0) * 0.8,
                -row.get('home_power_context_advantage', 0) * 0.6,
                -row.get('home_contact_context_advantage', 0) * 0.6
            ]),
            axis=1
        )
        
        # Normalize these scores to a standard range (-1 to 1)
        for col in ['total_runs_context_factor', 'total_strikeouts_context_factor', 
                   'home_advantage_score', 'away_advantage_score']:
            max_abs = games_df[col].abs().max()
            if max_abs > 0:
                games_df[col] = games_df[col] / max_abs
        
        return games_df
    
    def _get_feature_description(self, feature):
        """Get description of an engineered feature"""
        descriptions = {
            'ballpark_run_factor': "Normalized ballpark effect on run scoring, where positive values indicate more runs (e.g., 0.1 means 10% above average).",
            'ballpark_hr_factor': "Normalized ballpark effect on home runs, where positive values indicate more home runs (e.g., 0.15 means 15% above average).",
            'ballpark_hitter_friendly_score': "Overall score of how favorable the ballpark is for hitters, based on multiple factors.",
            'ballpark_elevation': "Elevation of the ballpark in feet, which affects ball flight distance.",
            'ballpark_environment_type': "Type of ballpark environment: 'dome', 'retractable', or 'outdoor'.",
            'ballpark_environment_factor': "Numeric factor representing how weather affects this ballpark (0 for dome, 0.5 for retractable, 1.0 for outdoor).",
            
            'temp_factor': "Effect of temperature on offense (-0.15 to 0.15 scale), where positive means higher scoring.",
            'wind_factor': "Effect of wind on offense (positive for outward wind, negative for inward wind).",
            'humidity_factor': "Effect of humidity on offense (slight negative effect as humidity increases).",
            'precipitation_factor': "Effect of precipitation on offense (negative, more severe for heavier rain).",
            'weather_score': "Combined score of all weather factors and their impact on offense.",
            
            'umpire_strikeout_boost': "Expected increase or decrease in strikeouts due to umpire tendencies.",
            'umpire_runs_boost': "Expected increase or decrease in runs due to umpire tendencies.",
            'umpire_consistency_factor': "How consistent the umpire's strike zone is relative to average.",
            
            'home_pitcher_context_advantage': "How much the contextual factors advantage the home starting pitcher (-1 to 1 scale).",
            'away_pitcher_context_advantage': "How much the contextual factors advantage the away starting pitcher (-1 to 1 scale).",
            'pitcher_matchup_strikeout_boost': "Expected impact on strikeouts based on pitcher skills and umpire tendencies.",
            
            'home_power_context_advantage': "How much the contextual factors boost the home team's power hitting (-1 to 1 scale).",
            'away_power_context_advantage': "How much the contextual factors boost the away team's power hitting (-1 to 1 scale).",
            'home_contact_context_advantage': "How much the contextual factors boost the home team's contact hitting (-1 to 1 scale).",
            'away_contact_context_advantage': "How much the contextual factors boost the away team's contact hitting (-1 to 1 scale).",
            
            'total_runs_context_factor': "Combined impact of all contextual factors on expected total runs (-1 to 1 scale).",
            'total_strikeouts_context_factor': "Combined impact of all contextual factors on expected strikeouts (-1 to 1 scale).",
            'home_advantage_score': "Overall contextual advantage score for the home team (-1 to 1 scale).",
            'away_advantage_score': "Overall contextual advantage score for the away team (-1 to 1 scale)."
        }
        
        return descriptions.get(feature, "No description available.")

def test_feature_engineering():
    """
    Test the contextual feature engineering on games data
    """
    # Initialize the feature engineer
    engineer = ContextualFeatureEngineer()
    
    # Try to load games with contextual data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    games_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_all_context.csv')
    
    if os.path.exists(games_file):
        print(f"Loading games with contextual data from {games_file}")
        games_df = pd.read_csv(games_file)
    else:
        # Create a basic sample of games if file not found
        print("Creating sample games data for testing")
        games_data = {
            'game_id': range(1, 11),
            'game_date': [f'2024-04-{i:02d}' for i in range(1, 11)],
            'home_team': ['NYY', 'BOS', 'CHC', 'LAD', 'SF', 'COL', 'ATL', 'HOU', 'MIA', 'TB'],
            'away_team': ['BOS', 'NYY', 'MIL', 'SD', 'LAA', 'ARI', 'WSH', 'TEX', 'PHI', 'BAL'],
            'home_starting_pitcher': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'away_starting_pitcher': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
            'temperature_f': [65, 55, 72, 68, 62, 75, 80, 85, 77, 72],
            'wind_speed_mph': [10, 12, 5, 7, 15, 8, 3, 12, 9, 5],
            'wind_direction': ['NE', 'SW', 'SE', 'N', 'W', 'S', 'NW', 'E', 'SW', 'NE'],
            'humidity_pct': [45, 60, 55, 50, 65, 40, 75, 35, 60, 50],
            'precipitation_in': [0, 0.1, 0, 0, 0.05, 0, 0.2, 0, 0, 0],
            'home_plate_umpire_id': list(range(1, 11)),
            'umpire_zone_size': [98, 105, 92, 100, 110, 95, 102, 97, 96, 103],
            'umpire_favor_factor': [1.5, -2.0, 0.5, -1.0, 3.0, -0.5, 1.0, 2.5, -1.5, 0.0]
        }
        games_df = pd.DataFrame(games_data)
    
    # Engineer contextual features
    output_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_engineered_features.csv')
    enhanced_games = engineer.engineer_features(games_df, output_file)
    
    # Print sample of enhanced data with key features
    print("\nSample of games with engineered contextual features:")
    
    sample_columns = [
        'game_id', 'home_team', 'away_team', 
        'total_runs_context_factor', 'total_strikeouts_context_factor',
        'home_advantage_score', 'away_advantage_score'
    ]
    
    available_cols = [col for col in sample_columns if col in enhanced_games.columns]
    print(enhanced_games[available_cols].head())
    
    # Count and categorize the features added
    contextual_columns = [col for col in enhanced_games.columns if col not in games_df.columns]
    print(f"\nTotal engineered features added: {len(contextual_columns)}")
    
    # Group features by category
    feature_categories = {
        'ballpark': [col for col in contextual_columns if col.startswith('ballpark_')],
        'weather': [col for col in contextual_columns if col in ['temp_factor', 'wind_factor', 'humidity_factor', 
                                                             'precipitation_factor', 'weather_score']],
        'umpire': [col for col in contextual_columns if col.startswith('umpire_')],
        'pitcher': [col for col in contextual_columns if 'pitcher' in col],
        'team': [col for col in contextual_columns if 'power' in col or 'contact' in col],
        'combined': [col for col in contextual_columns if col in ['total_runs_context_factor', 
                                                             'total_strikeouts_context_factor',
                                                             'home_advantage_score', 'away_advantage_score']]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category.capitalize()} features ({len(features)}):")
            for feature in features:
                print(f"- {feature}")

if __name__ == "__main__":
    test_feature_engineering()
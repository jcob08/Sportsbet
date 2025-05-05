import os
import pandas as pd
import numpy as np
from datetime import datetime
import math
import requests

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
        """Load enhanced player statistics from processed advanced stats file"""
        # Use the new processed advanced stats file
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
        # Find the most recent advanced_player_stats file
        files = [f for f in os.listdir(processed_dir) if f.startswith('advanced_player_stats_') and f.endswith('.csv')]
        if files:
            files.sort(reverse=True)
            latest_file = os.path.join(processed_dir, files[0])
            print(f"Loading advanced player stats from {latest_file}")
            return pd.read_csv(latest_file)
        print("Warning: No advanced player statistics files found")
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
                                 'ballpark_elevation', 'ballpark_environment_type', 'ballpark_dimensions',
                                 'ballpark_orientation', 'ballpark_roof_type', 'day_night_factor',
                                 'ballpark_wind_factor'])
        
        # 2. Enhance weather features
        if 'temperature_f' in enhanced_games.columns:
            enhanced_games = self._add_weather_features(enhanced_games)
            added_features.extend(['temp_factor', 'wind_factor', 'humidity_factor', 
                                 'precipitation_factor', 'weather_score', 'temp_humidity_interaction',
                                 'wind_speed_squared', 'precipitation_probability_factor',
                                 'temperature_change_factor'])
        
        # 3. Add umpire features
        if 'home_plate_umpire_id' in enhanced_games.columns or 'umpire_zone_size' in enhanced_games.columns:
            enhanced_games = self._add_umpire_features(enhanced_games)
            added_features.extend(['umpire_strikeout_boost', 'umpire_runs_boost', 'umpire_consistency_factor',
                                 'umpire_fatigue_factor', 'umpire_home_team_interaction',
                                 'umpire_away_team_interaction', 'umpire_home_pitcher_interaction',
                                 'umpire_away_pitcher_interaction', 'umpire_impact_score'])
        
        # 4. Add pitcher-specific contextual features using advanced stats
        if 'home_starting_pitcher' in enhanced_games.columns and 'away_starting_pitcher' in enhanced_games.columns:
            enhanced_games = self._add_simplified_pitcher_features(enhanced_games)
            added_features.extend(['home_pitcher_context_advantage', 'away_pitcher_context_advantage', 
                                 'pitcher_matchup_strikeout_boost', 'home_pitcher_fatigue',
                                 'away_pitcher_fatigue'])
        
        # 5. Add batter-specific contextual features using advanced stats
        enhanced_games = self._add_simplified_batter_features(enhanced_games)
        added_features.extend(['home_power_context_advantage', 'away_power_context_advantage', 
                             'home_contact_context_advantage', 'away_contact_context_advantage',
                             'home_lineup_advantage', 'away_lineup_advantage'])
        
        # 6. Create combined scoring features
        enhanced_games = self._create_combined_features(enhanced_games)
        added_features.extend(['total_runs_context_factor', 'total_strikeouts_context_factor', 
                             'home_advantage_score', 'away_advantage_score', 'pitcher_matchup_advantage',
                             'power_matchup_advantage', 'contact_matchup_advantage', 'game_context_score'])
        
        # Print summary of added features
        print(f"Added {len(added_features)} contextual features to game data")
        
        # Validate features
        validation_results = self.validate_features(enhanced_games)
        
        # Print validation results
        print("\nFeature Validation Results:")
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"- {warning}")
        
        if validation_results['missing_values']:
            print("\nFeatures with missing values:")
            for feature, stats in validation_results['missing_values'].items():
                print(f"- {feature}: {stats['count']} missing values ({stats['percentage']:.2f}%)")
        
        if validation_results['value_ranges']:
            print("\nFeatures with values outside expected range:")
            for feature, stats in validation_results['value_ranges'].items():
                print(f"- {feature}: min={stats['min']:.2f}, max={stats['max']:.2f} (expected {stats['expected_range']})")
        
        if validation_results['correlations']:
            print("\nTop correlations with target variables:")
            for target, correlations in validation_results['correlations'].items():
                print(f"\n{target}:")
                for feature, corr in correlations.items():
                    print(f"- {feature}: {corr:.3f}")
        
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
                
                # Write validation results
                f.write("## Validation Results\n\n")
                if validation_results['warnings']:
                    f.write("### Warnings\n\n")
                    for warning in validation_results['warnings']:
                        f.write(f"- {warning}\n")
                
                if validation_results['missing_values']:
                    f.write("\n### Missing Values\n\n")
                    for feature, stats in validation_results['missing_values'].items():
                        f.write(f"- {feature}: {stats['count']} missing values ({stats['percentage']:.2f}%)\n")
                
                if validation_results['value_ranges']:
                    f.write("\n### Value Ranges\n\n")
                    for feature, stats in validation_results['value_ranges'].items():
                        f.write(f"- {feature}: min={stats['min']:.2f}, max={stats['max']:.2f} (expected {stats['expected_range']})\n")
                
                if validation_results['correlations']:
                    f.write("\n### Correlations\n\n")
                    for target, correlations in validation_results['correlations'].items():
                        f.write(f"#### {target}\n\n")
                        for feature, corr in correlations.items():
                            f.write(f"- {feature}: {corr:.3f}\n")
            
            print(f"Saved feature documentation to {doc_file}")
        
        return enhanced_games
    
    def _add_ballpark_features(self, games_df):
        """Add ballpark-specific features"""
        print("Adding ballpark features...")
        # Create mapping of team codes to ballpark factors
        if 'team_code' in self.ballparks_df.columns:
            # Basic ballpark factors
            park_run_factors = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['runs_factor']))
            park_hr_factors = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['hr_factor']))
            park_hitter_scores = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['hitter_friendly_score']))
            park_elevations = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['elevation_feet']))
            park_environments = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['environment_type']))
            # Enhanced ballpark factors (check for existence)
            park_dimensions = None
            park_orientation = None
            park_roof_type = None
            if 'dimensions' in self.ballparks_df.columns:
                park_dimensions = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['dimensions']))
            if 'orientation' in self.ballparks_df.columns:
                park_orientation = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['orientation']))
            if 'roof_type' in self.ballparks_df.columns:
                park_roof_type = dict(zip(self.ballparks_df['team_code'], self.ballparks_df['roof_type']))
            # Map these factors to the home team in each game
            if 'home_team' in games_df.columns:
                # Basic factors
                games_df['ballpark_run_factor'] = games_df['home_team'].map(park_run_factors)
                games_df['ballpark_hr_factor'] = games_df['home_team'].map(park_hr_factors)
                games_df['ballpark_hitter_friendly_score'] = games_df['home_team'].map(park_hitter_scores)
                games_df['ballpark_elevation'] = games_df['home_team'].map(park_elevations)
                games_df['ballpark_environment_type'] = games_df['home_team'].map(park_environments)
                # Enhanced factors (only if available)
                if park_dimensions is not None:
                    games_df['ballpark_dimensions'] = games_df['home_team'].map(park_dimensions)
                if park_orientation is not None:
                    games_df['ballpark_orientation'] = games_df['home_team'].map(park_orientation)
                if park_roof_type is not None:
                    games_df['ballpark_roof_type'] = games_df['home_team'].map(park_roof_type)
                # Normalize factors to percentage above/below average
                games_df['ballpark_run_factor'] = (games_df['ballpark_run_factor'] - 100) / 100
                games_df['ballpark_hr_factor'] = (games_df['ballpark_hr_factor'] - 100) / 100
                # Convert environment type to numeric factor
                env_map = {'dome': 0.0, 'retractable': 0.5, 'outdoor': 1.0}
                games_df['ballpark_environment_factor'] = games_df['ballpark_environment_type'].map(env_map)
                # Add day/night game effect based on ballpark
                games_df['day_night_factor'] = games_df.apply(
                    lambda row: 0.1 if row.get('is_day_game', False) and row['ballpark_environment_type'] == 'outdoor' else
                    0.05 if row.get('is_day_game', False) and row['ballpark_environment_type'] == 'retractable' else
                    0.0,
                    axis=1
                )
                # Add ballpark-specific wind effects
                if 'wind_direction' in games_df.columns and park_orientation is not None:
                    games_df['ballpark_wind_factor'] = games_df.apply(
                        lambda row: self._calculate_ballpark_wind_effect(
                            row['wind_direction'],
                            row['wind_speed_mph'],
                            row['ballpark_orientation'] if 'ballpark_orientation' in games_df.columns else None
                        ),
                        axis=1
                    )
        return games_df
    
    def _calculate_ballpark_wind_effect(self, wind_direction, wind_speed, ballpark_orientation):
        """Calculate the effect of wind on the ballpark based on its orientation"""
        if pd.isna(wind_direction) or pd.isna(wind_speed) or pd.isna(ballpark_orientation):
            return 0.0
            
        # Convert wind direction to degrees (0-360)
        wind_degrees = self._direction_to_degrees(wind_direction)
        
        # Calculate relative angle between wind and ballpark orientation
        relative_angle = abs(wind_degrees - ballpark_orientation) % 360
        if relative_angle > 180:
            relative_angle = 360 - relative_angle
            
        # Calculate wind effect based on relative angle
        # 0 degrees = wind blowing straight out
        # 180 degrees = wind blowing straight in
        wind_effect = math.cos(math.radians(relative_angle))
        
        # Scale by wind speed
        return wind_effect * (wind_speed / 10.0)
    
    def _direction_to_degrees(self, direction):
        """Convert wind direction to degrees"""
        direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        return direction_map.get(direction, 0)
    
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
        
        # Enhanced humidity effects
        if 'humidity_pct' in games_df.columns:
            # Humidity effect is stronger at higher temperatures
            games_df['humidity_factor'] = games_df.apply(
                lambda row: 0.0 if pd.isna(row['humidity_pct']) or pd.isna(row['temperature_f']) else
                (row['humidity_pct'] - 50) * -0.001 * (1 + (row['temperature_f'] - 70) / 50),  # Stronger effect at higher temps
                axis=1
            )
            
            # Add temperature-humidity interaction
            games_df['temp_humidity_interaction'] = games_df.apply(
                lambda row: 0.0 if pd.isna(row['humidity_pct']) or pd.isna(row['temperature_f']) else
                (row['temperature_f'] - 70) * (row['humidity_pct'] - 50) * 0.0001,  # Small interaction effect
                axis=1
            )
        else:
            games_df['humidity_factor'] = 0.0
            games_df['temp_humidity_interaction'] = 0.0
        
        # Enhanced wind effects
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
            
            # Add wind speed squared effect (non-linear impact)
            games_df['wind_speed_squared'] = games_df['wind_speed_mph'].apply(
                lambda x: 0.0 if pd.isna(x) else (x ** 2) * 0.0001
            )
        else:
            games_df['wind_factor'] = 0.0
            games_df['wind_speed_squared'] = 0.0
        
        # Enhanced precipitation effects
        if 'precipitation_in' in games_df.columns:
            # Non-linear precipitation effect
            games_df['precipitation_factor'] = games_df['precipitation_in'].apply(
                lambda x: 0.0 if pd.isna(x) else
                -0.3 if x > 0.2 else  # Heavy rain
                -0.2 if x > 0.1 else  # Moderate rain
                -0.1 if x > 0 else    # Light rain
                0.0                   # No rain
            )
            
            # Add precipitation probability effect
            if 'precipitation_probability' in games_df.columns:
                games_df['precipitation_probability_factor'] = games_df['precipitation_probability'].apply(
                    lambda x: 0.0 if pd.isna(x) else
                    -0.1 if x > 0.7 else  # High probability
                    -0.05 if x > 0.3 else  # Medium probability
                    0.0                   # Low probability
                )
            else:
                games_df['precipitation_probability_factor'] = 0.0
        else:
            games_df['precipitation_factor'] = 0.0
            games_df['precipitation_probability_factor'] = 0.0
        
        # Create overall weather score with enhanced factors
        weather_components = [
            'temp_factor',
            'humidity_factor',
            'temp_humidity_interaction',
            'wind_factor',
            'wind_speed_squared',
            'precipitation_factor',
            'precipitation_probability_factor'
        ]
        
        # Calculate weighted weather score
        weights = {
            'temp_factor': 0.3,
            'humidity_factor': 0.15,
            'temp_humidity_interaction': 0.1,
            'wind_factor': 0.2,
            'wind_speed_squared': 0.05,
            'precipitation_factor': 0.15,
            'precipitation_probability_factor': 0.05
        }
        
        games_df['weather_score'] = sum(
            games_df[col] * weights[col] for col in weather_components if col in games_df.columns
        )
        
        # Add weather change effect if available
        if 'temperature_change' in games_df.columns:
            games_df['temperature_change_factor'] = games_df['temperature_change'].apply(
                lambda x: 0.0 if pd.isna(x) else
                -0.1 if x < -10 else  # Large temperature drop
                -0.05 if x < -5 else  # Moderate temperature drop
                0.05 if x > 5 else    # Moderate temperature rise
                0.1 if x > 10 else    # Large temperature rise
                0.0                   # Small change
            )
            games_df['weather_score'] += games_df['temperature_change_factor']
        
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
            
            # Add umpire fatigue factor
            if 'umpire_games_worked' in games_df.columns:
                games_df['umpire_fatigue_factor'] = games_df['umpire_games_worked'].apply(
                    lambda x: 0.0 if pd.isna(x) else
                    -0.1 if x > 5 else  # High fatigue
                    -0.05 if x > 3 else  # Moderate fatigue
                    0.0                 # Low fatigue
                )
            else:
                games_df['umpire_fatigue_factor'] = 0.0
            
            # Add umpire-team interaction effects
            if 'home_team' in games_df.columns and 'umpire_id' in games_df.columns:
                # Create umpire-team interaction features
                games_df['umpire_home_team_interaction'] = games_df.apply(
                    lambda row: self._get_umpire_team_interaction(
                        row['umpire_id'],
                        row['home_team']
                    ),
                    axis=1
                )
                
                games_df['umpire_away_team_interaction'] = games_df.apply(
                    lambda row: self._get_umpire_team_interaction(
                        row['umpire_id'],
                        row['away_team']
                    ),
                    axis=1
                )
            else:
                games_df['umpire_home_team_interaction'] = 0.0
                games_df['umpire_away_team_interaction'] = 0.0
            
            # Add umpire-pitcher interaction effects
            if 'home_starting_pitcher' in games_df.columns and 'away_starting_pitcher' in games_df.columns and 'umpire_id' in games_df.columns:
                games_df['umpire_home_pitcher_interaction'] = games_df.apply(
                    lambda row: self._get_umpire_pitcher_interaction(
                        row['umpire_id'],
                        row['home_starting_pitcher']
                    ),
                    axis=1
                )
                games_df['umpire_away_pitcher_interaction'] = games_df.apply(
                    lambda row: self._get_umpire_pitcher_interaction(
                        row['umpire_id'],
                        row['away_starting_pitcher']
                    ),
                    axis=1
                )
            else:
                games_df['umpire_home_pitcher_interaction'] = 0.0
                games_df['umpire_away_pitcher_interaction'] = 0.0
            
            # Create combined umpire impact score
            umpire_components = [
                'umpire_strikeout_boost',
                'umpire_runs_boost',
                'umpire_consistency_factor',
                'umpire_fatigue_factor',
                'umpire_home_team_interaction',
                'umpire_away_team_interaction',
                'umpire_home_pitcher_interaction',
                'umpire_away_pitcher_interaction'
            ]
            
            # Calculate weighted umpire impact
            weights = {
                'umpire_strikeout_boost': 0.25,
                'umpire_runs_boost': 0.25,
                'umpire_consistency_factor': 0.15,
                'umpire_fatigue_factor': 0.1,
                'umpire_home_team_interaction': 0.1,
                'umpire_away_team_interaction': 0.1,
                'umpire_home_pitcher_interaction': 0.05,
                'umpire_away_pitcher_interaction': 0.05
            }
            
            games_df['umpire_impact_score'] = sum(
                games_df[col] * weights[col] for col in umpire_components if col in games_df.columns
            )
        
        return games_df
    
    def _get_umpire_team_interaction(self, umpire_id, team):
        """Get historical interaction between umpire and team"""
        # This would be implemented with historical data
        # For now, return a placeholder value
        return 0.0
    
    def _get_umpire_pitcher_interaction(self, umpire_id, pitcher_id):
        """Get historical interaction between umpire and pitcher"""
        # This would be implemented with historical data
        # For now, return a placeholder value
        return 0.0
    
    def _add_simplified_pitcher_features(self, games_df):
        """Add pitcher-specific contextual advantage features using advanced stats"""
        print("Adding pitcher-context interaction features (using advanced stats)...")
        
        # Build pitcher stats lookup
        pitcher_stats = {}
        if self.player_stats_df is not None:
            for _, row in self.player_stats_df.iterrows():
                if 'player_id' in row and 'xERA' in row:
                    pitcher_stats[row['player_id']] = row
        
        def get_pitcher_stat(pid, stat):
            if pd.isna(pid):
                return None
            row = pitcher_stats.get(pid)
            if row is not None and stat in row:
                return row[stat]
            return None
        
        def pitcher_advantage(pid, park_run_factor, park_hr_factor, umpire_favor, umpire_consistency, weather_score):
            if pd.isna(pid):
                return 0.0
            # Use advanced stats
            xERA = get_pitcher_stat(pid, 'xERA') or 4.5
            spin_rate = get_pitcher_stat(pid, 'spin_rate') or 2200
            whiff_rate = get_pitcher_stat(pid, 'whiff_rate') or 0.25
            gb_rate = get_pitcher_stat(pid, 'gb_rate') or 0.45
            fb_rate = get_pitcher_stat(pid, 'fb_rate') or 0.35
            pfx_x = get_pitcher_stat(pid, 'pfx_x') or 0.0
            pfx_z = get_pitcher_stat(pid, 'pfx_z') or 0.0
            
            # Enhanced pitcher metrics
            rest_days = get_pitcher_stat(pid, 'rest_days') or 4
            recent_workload = get_pitcher_stat(pid, 'recent_workload') or 0.0
            pitch_mix_change = get_pitcher_stat(pid, 'pitch_mix_change') or 0.0
            velocity_trend = get_pitcher_stat(pid, 'velocity_trend') or 0.0
            
            # Calculate advantage
            advantage = 0.0
            
            # Base performance metrics
            advantage -= (xERA - 4.0) * 0.3  # Lower xERA is better
            advantage += (spin_rate - 2200) / 10000
            advantage += (whiff_rate - 0.25) * 1.5
            advantage += (gb_rate - 0.45) * 0.5
            advantage -= (fb_rate - 0.35) * 0.5
            advantage += (pfx_x + pfx_z) * 0.01
            
            # Enhanced metrics
            # Rest days effect (optimal at 4-5 days)
            if rest_days < 3:
                advantage -= 0.2  # Too little rest
            elif rest_days > 6:
                advantage -= 0.1  # Too much rest
            
            # Recent workload effect
            advantage -= recent_workload * 0.1
            
            # Pitch mix change effect
            advantage += pitch_mix_change * 0.05  # Small positive effect for changing pitch mix
            
            # Velocity trend effect
            advantage += velocity_trend * 0.1  # Positive for increasing velocity
            
            # Contextual effects
            advantage -= park_run_factor * 0.5
            advantage -= park_hr_factor * 0.7
            advantage += umpire_favor * 0.2
            advantage += umpire_consistency * 0.1
            advantage += weather_score * 0.1
            
            return max(-1, min(1, advantage))
        
        def strikeout_boost(pid, umpire_boost, umpire_consistency):
            if pd.isna(pid):
                return 0.0
            whiff_rate = get_pitcher_stat(pid, 'whiff_rate') or 0.25
            spin_rate = get_pitcher_stat(pid, 'spin_rate') or 2200
            velocity = get_pitcher_stat(pid, 'velocity') or 92.0
            
            # Calculate strikeout boost
            boost = 0.0
            boost += (whiff_rate - 0.25) * 2.0
            boost += (spin_rate - 2200) / 5000
            boost += (velocity - 92.0) * 0.05
            boost += umpire_boost * 1.5
            boost += umpire_consistency * 0.5
            
            return boost
        
        # Add pitcher features
        games_df['home_pitcher_context_advantage'] = games_df.apply(
            lambda row: pitcher_advantage(
                row.get('home_starting_pitcher'),
                row.get('ballpark_run_factor', 0),
                row.get('ballpark_hr_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('umpire_consistency_factor', 0),
                row.get('weather_score', 0)
            ),
            axis=1
        )
        
        games_df['away_pitcher_context_advantage'] = games_df.apply(
            lambda row: pitcher_advantage(
                row.get('away_starting_pitcher'),
                row.get('ballpark_run_factor', 0),
                row.get('ballpark_hr_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('umpire_consistency_factor', 0),
                row.get('weather_score', 0)
            ),
            axis=1
        )
        
        # Add strikeout boost
        games_df['pitcher_matchup_strikeout_boost'] = games_df.apply(
            lambda row: strikeout_boost(
                row.get('home_starting_pitcher'),
                row.get('umpire_strikeout_boost', 0),
                row.get('umpire_consistency_factor', 0)
            ) + strikeout_boost(
                row.get('away_starting_pitcher'),
                row.get('umpire_strikeout_boost', 0),
                row.get('umpire_consistency_factor', 0)
            ),
            axis=1
        )
        
        # Add pitcher fatigue factor
        if 'pitcher_rest_days' in games_df.columns:
            games_df['home_pitcher_fatigue'] = games_df['pitcher_rest_days'].apply(
                lambda x: 0.0 if pd.isna(x) else
                -0.2 if x < 3 else  # Too little rest
                -0.1 if x < 4 else  # Slightly less than optimal
                0.0 if x <= 5 else  # Optimal rest
                -0.05 if x <= 6 else  # Slightly more than optimal
                -0.1  # Too much rest
            )
            
            games_df['away_pitcher_fatigue'] = games_df['pitcher_rest_days'].apply(
                lambda x: 0.0 if pd.isna(x) else
                -0.2 if x < 3 else  # Too little rest
                -0.1 if x < 4 else  # Slightly less than optimal
                0.0 if x <= 5 else  # Optimal rest
                -0.05 if x <= 6 else  # Slightly more than optimal
                -0.1  # Too much rest
            )
        else:
            games_df['home_pitcher_fatigue'] = 0.0
            games_df['away_pitcher_fatigue'] = 0.0
        
        return games_df
    
    def _add_simplified_batter_features(self, games_df):
        """Add batter-specific contextual features using advanced stats"""
        print("Adding batter-context interaction features (using advanced stats)...")
        
        # Aggregate batter stats by team
        if self.player_stats_df is not None:
            # Assume player_stats_df has 'team' and advanced stat columns
            team_batters = self.player_stats_df.groupby('team').agg({
                'barrel_rate': 'mean',
                'hard_hit_rate': 'mean',
                'pull_rate': 'mean',
                'gb_rate': 'mean',
                'fb_rate': 'mean',
                'avg_distance': 'mean',
                'xwOBA': 'mean',
                'xSLG': 'mean',
                'sweet_spot_rate': 'mean',
                'contact_rate': 'mean',
                'exit_velocity': 'mean',
                'launch_angle': 'mean',
                'walk_rate': 'mean',
                'strikeout_rate': 'mean',
                'batted_ball_events': 'sum',
                'recent_form': 'mean',
                'streak': 'mean'
            }).to_dict(orient='index')
        else:
            team_batters = {}
            
        def get_team_stat(team, stat):
            if team in team_batters and stat in team_batters[team]:
                return team_batters[team][stat]
            return 0.0
            
        def power_advantage(team, park_hr_factor, weather_score, temp_factor, wind_factor):
            barrel = get_team_stat(team, 'barrel_rate')
            hard_hit = get_team_stat(team, 'hard_hit_rate')
            pull = get_team_stat(team, 'pull_rate')
            xSLG = get_team_stat(team, 'xSLG')
            ev = get_team_stat(team, 'exit_velocity')
            la = get_team_stat(team, 'launch_angle')
            recent_form = get_team_stat(team, 'recent_form')
            streak = get_team_stat(team, 'streak')
            
            # Calculate power advantage
            advantage = 0.0
            
            # Base power metrics
            advantage += (barrel - 0.07) * 2.0
            advantage += (hard_hit - 0.35) * 1.5
            advantage += (pull - 0.4) * 1.0
            advantage += (xSLG - 0.400) * 2.0
            advantage += (ev - 88.0) * 0.05
            advantage += (la - 12.0) * 0.02
            
            # Recent form and streak effects
            advantage += recent_form * 0.2
            advantage += streak * 0.1
            
            # Contextual effects
            advantage += park_hr_factor * 1.0
            advantage += weather_score * 0.5
            advantage += temp_factor * 0.5
            advantage += wind_factor * 0.3
            
            return max(-1, min(1, advantage))
            
        def contact_advantage(team, park_run_factor, umpire_favor, temperature, precipitation_factor):
            sweet_spot = get_team_stat(team, 'sweet_spot_rate')
            contact = get_team_stat(team, 'contact_rate')
            xwOBA = get_team_stat(team, 'xwOBA')
            walk_rate = get_team_stat(team, 'walk_rate')
            strikeout_rate = get_team_stat(team, 'strikeout_rate')
            recent_form = get_team_stat(team, 'recent_form')
            streak = get_team_stat(team, 'streak')
            
            # Calculate contact advantage
            advantage = 0.0
            
            # Base contact metrics
            advantage += (sweet_spot - 0.32) * 2.0
            advantage += (contact - 0.75) * 1.5
            advantage += (xwOBA - 0.320) * 2.0
            advantage += (walk_rate - 0.08) * 1.0
            advantage -= (strikeout_rate - 0.22) * 1.0
            
            # Recent form and streak effects
            advantage += recent_form * 0.2
            advantage += streak * 0.1
            
            # Contextual effects
            advantage += park_run_factor * 1.0
            advantage += -umpire_favor * 0.5
            if temperature < 50:
                advantage -= (50 - temperature) * 0.01
            advantage += precipitation_factor * 1.0
            
            return max(-1, min(1, advantage))
            
        def lineup_advantage(team, park_factors, weather_factors, umpire_factors):
            # Calculate lineup advantage based on team batting order
            # This would be implemented with actual lineup data
            # For now, return a placeholder value
            return 0.0
            
        # Add power and contact features
        games_df['home_power_context_advantage'] = games_df.apply(
            lambda row: power_advantage(
                row.get('home_team'),
                row.get('ballpark_hr_factor', 0),
                row.get('weather_score', 0),
                row.get('temp_factor', 0),
                row.get('wind_factor', 0)
            ),
            axis=1
        )
        
        games_df['away_power_context_advantage'] = games_df.apply(
            lambda row: power_advantage(
                row.get('away_team'),
                row.get('ballpark_hr_factor', 0),
                row.get('weather_score', 0),
                row.get('temp_factor', 0),
                row.get('wind_factor', 0)
            ),
            axis=1
        )
        
        games_df['home_contact_context_advantage'] = games_df.apply(
            lambda row: contact_advantage(
                row.get('home_team'),
                row.get('ballpark_run_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('temperature_f', 70),
                row.get('precipitation_factor', 0)
            ),
            axis=1
        )
        
        games_df['away_contact_context_advantage'] = games_df.apply(
            lambda row: contact_advantage(
                row.get('away_team'),
                row.get('ballpark_run_factor', 0),
                row.get('umpire_favor_factor', 0),
                row.get('temperature_f', 70),
                row.get('precipitation_factor', 0)
            ),
            axis=1
        )
        
        # Add lineup advantage features
        games_df['home_lineup_advantage'] = games_df.apply(
            lambda row: lineup_advantage(
                row.get('home_team'),
                {
                    'run_factor': row.get('ballpark_run_factor', 0),
                    'hr_factor': row.get('ballpark_hr_factor', 0)
                },
                {
                    'score': row.get('weather_score', 0),
                    'temp': row.get('temp_factor', 0),
                    'wind': row.get('wind_factor', 0)
                },
                {
                    'favor': row.get('umpire_favor_factor', 0),
                    'consistency': row.get('umpire_consistency_factor', 0)
                }
            ),
            axis=1
        )
        
        games_df['away_lineup_advantage'] = games_df.apply(
            lambda row: lineup_advantage(
                row.get('away_team'),
                {
                    'run_factor': row.get('ballpark_run_factor', 0),
                    'hr_factor': row.get('ballpark_hr_factor', 0)
                },
                {
                    'score': row.get('weather_score', 0),
                    'temp': row.get('temp_factor', 0),
                    'wind': row.get('wind_factor', 0)
                },
                {
                    'favor': row.get('umpire_favor_factor', 0),
                    'consistency': row.get('umpire_consistency_factor', 0)
                }
            ),
            axis=1
        )
        
        return games_df
    
    def _create_combined_features(self, games_df):
        """Create combined scoring and outcome features"""
        print("Creating combined contextual features...")
        
        # Total runs context factor - how contextual factors affect total run scoring
        # Combine ballpark, weather, and umpire factors with interactions
        games_df['total_runs_context_factor'] = games_df.apply(
            lambda row: sum([
                # Base factors
                row.get('ballpark_run_factor', 0) * 2.0,      # Strongest effect
                row.get('weather_score', 0) * 1.5,            # Strong effect
                row.get('umpire_runs_boost', 0) * 1.0,        # Moderate effect
                -row.get('pitcher_matchup_strikeout_boost', 0) * 0.5,  # Inverse effect
                
                # Interaction effects
                row.get('ballpark_run_factor', 0) * row.get('weather_score', 0) * 0.5,  # Park-weather interaction
                row.get('ballpark_run_factor', 0) * row.get('umpire_runs_boost', 0) * 0.3,  # Park-umpire interaction
                row.get('weather_score', 0) * row.get('umpire_runs_boost', 0) * 0.2,  # Weather-umpire interaction
                
                # Enhanced effects
                row.get('day_night_factor', 0) * 0.5,  # Day/night effect
                row.get('ballpark_wind_factor', 0) * 0.8,  # Ballpark-specific wind
                row.get('temp_humidity_interaction', 0) * 0.3,  # Temperature-humidity interaction
                row.get('wind_speed_squared', 0) * 0.2,  # Non-linear wind effect
                row.get('precipitation_probability_factor', 0) * 0.4,  # Rain probability effect
                row.get('temperature_change_factor', 0) * 0.3  # Temperature change effect
            ]),
            axis=1
        )
        
        # Total strikeouts context factor with enhanced interactions
        games_df['total_strikeouts_context_factor'] = games_df.apply(
            lambda row: sum([
                # Base factors
                row.get('umpire_strikeout_boost', 0) * 2.0,   # Strongest effect
                row.get('pitcher_matchup_strikeout_boost', 0) * 1.5,  # Strong effect
                -row.get('weather_score', 0) * 0.5,           # Inverse effect
                row.get('umpire_consistency_factor', 0) * 0.5,  # Small effect
                
                # Interaction effects
                row.get('umpire_strikeout_boost', 0) * row.get('pitcher_matchup_strikeout_boost', 0) * 0.3,  # Umpire-pitcher interaction
                row.get('umpire_strikeout_boost', 0) * -row.get('weather_score', 0) * 0.2,  # Umpire-weather interaction
                row.get('pitcher_matchup_strikeout_boost', 0) * -row.get('weather_score', 0) * 0.2,  # Pitcher-weather interaction
                
                # Enhanced effects
                row.get('umpire_fatigue_factor', 0) * 0.3,  # Umpire fatigue
                row.get('home_pitcher_fatigue', 0) * 0.2,  # Home pitcher fatigue
                row.get('away_pitcher_fatigue', 0) * 0.2,  # Away pitcher fatigue
                row.get('umpire_home_pitcher_interaction', 0) * 0.3,  # Umpire-pitcher interaction
                row.get('umpire_away_pitcher_interaction', 0) * 0.3  # Umpire-pitcher interaction
            ]),
            axis=1
        )
        
        # Home team advantage with enhanced contextual factors
        games_df['home_advantage_score'] = games_df.apply(
            lambda row: sum([
                # Base advantages
                row.get('home_pitcher_context_advantage', 0) * 1.0,
                row.get('home_power_context_advantage', 0) * 0.7,
                row.get('home_contact_context_advantage', 0) * 0.7,
                -row.get('away_pitcher_context_advantage', 0) * 0.8,
                -row.get('away_power_context_advantage', 0) * 0.6,
                -row.get('away_contact_context_advantage', 0) * 0.6,
                
                # Ballpark advantages
                row.get('ballpark_run_factor', 0) * 0.5,  # Home park run factor
                row.get('ballpark_hr_factor', 0) * 0.4,  # Home park HR factor
                row.get('ballpark_hitter_friendly_score', 0) * 0.3,  # Overall park favorability
                
                # Weather advantages
                row.get('weather_score', 0) * 0.4,  # Weather conditions
                row.get('day_night_factor', 0) * 0.3,  # Day/night effect
                row.get('ballpark_wind_factor', 0) * 0.3,  # Park-specific wind
                
                # Umpire advantages
                row.get('umpire_impact_score', 0) * 0.3,  # Overall umpire impact
                row.get('umpire_home_team_interaction', 0) * 0.2,  # Umpire-team interaction
                -row.get('umpire_away_team_interaction', 0) * 0.2,  # Umpire-team interaction
                
                # Lineup advantages
                row.get('home_lineup_advantage', 0) * 0.4,  # Lineup strength
                -row.get('away_lineup_advantage', 0) * 0.3  # Opponent lineup strength
            ]),
            axis=1
        )
        
        # Away team advantage with enhanced contextual factors
        games_df['away_advantage_score'] = games_df.apply(
            lambda row: sum([
                # Base advantages
                row.get('away_pitcher_context_advantage', 0) * 1.0,
                row.get('away_power_context_advantage', 0) * 0.7,
                row.get('away_contact_context_advantage', 0) * 0.7,
                -row.get('home_pitcher_context_advantage', 0) * 0.8,
                -row.get('home_power_context_advantage', 0) * 0.6,
                -row.get('home_contact_context_advantage', 0) * 0.6,
                
                # Ballpark advantages (inverse of home)
                -row.get('ballpark_run_factor', 0) * 0.5,  # Away park run factor
                -row.get('ballpark_hr_factor', 0) * 0.4,  # Away park HR factor
                -row.get('ballpark_hitter_friendly_score', 0) * 0.3,  # Overall park favorability
                
                # Weather advantages
                row.get('weather_score', 0) * 0.4,  # Weather conditions
                -row.get('day_night_factor', 0) * 0.3,  # Day/night effect
                -row.get('ballpark_wind_factor', 0) * 0.3,  # Park-specific wind
                
                # Umpire advantages
                row.get('umpire_impact_score', 0) * 0.3,  # Overall umpire impact
                row.get('umpire_away_team_interaction', 0) * 0.2,  # Umpire-team interaction
                -row.get('umpire_home_team_interaction', 0) * 0.2,  # Umpire-team interaction
                
                # Lineup advantages
                row.get('away_lineup_advantage', 0) * 0.4,  # Lineup strength
                -row.get('home_lineup_advantage', 0) * 0.3  # Opponent lineup strength
            ]),
            axis=1
        )
        
        # Create matchup-specific features
        games_df['pitcher_matchup_advantage'] = games_df.apply(
            lambda row: row.get('home_pitcher_context_advantage', 0) - row.get('away_pitcher_context_advantage', 0),
            axis=1
        )
        
        games_df['power_matchup_advantage'] = games_df.apply(
            lambda row: row.get('home_power_context_advantage', 0) - row.get('away_power_context_advantage', 0),
            axis=1
        )
        
        games_df['contact_matchup_advantage'] = games_df.apply(
            lambda row: row.get('home_contact_context_advantage', 0) - row.get('away_contact_context_advantage', 0),
            axis=1
        )
        
        # Create overall game context score
        games_df['game_context_score'] = games_df.apply(
            lambda row: sum([
                row.get('total_runs_context_factor', 0) * 0.4,  # Run scoring context
                row.get('total_strikeouts_context_factor', 0) * 0.3,  # Strikeout context
                row.get('home_advantage_score', 0) * 0.2,  # Home advantage
                row.get('away_advantage_score', 0) * 0.1  # Away advantage
            ]),
            axis=1
        )
        
        # Normalize these scores to a standard range (-1 to 1)
        for col in ['total_runs_context_factor', 'total_strikeouts_context_factor', 
                   'home_advantage_score', 'away_advantage_score', 'game_context_score',
                   'pitcher_matchup_advantage', 'power_matchup_advantage', 'contact_matchup_advantage']:
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
            'ballpark_dimensions': "Physical dimensions of the ballpark affecting play.",
            'ballpark_orientation': "Orientation of the ballpark affecting wind patterns.",
            'ballpark_roof_type': "Type of roof affecting weather conditions.",
            'day_night_factor': "Effect of day/night game on scoring (positive for day games in outdoor parks).",
            'ballpark_wind_factor': "Effect of wind on the ballpark based on its orientation and wind conditions.",
            
            'temp_factor': "Effect of temperature on offense (-0.15 to 0.15 scale), where positive means higher scoring.",
            'wind_factor': "Effect of wind on offense (positive for outward wind, negative for inward wind).",
            'humidity_factor': "Effect of humidity on offense (slight negative effect as humidity increases).",
            'precipitation_factor': "Effect of precipitation on offense (negative, more severe for heavier rain).",
            'weather_score': "Combined score of all weather factors and their impact on offense.",
            'temp_humidity_interaction': "Interaction effect between temperature and humidity on offense.",
            'wind_speed_squared': "Non-linear effect of wind speed on offense.",
            'precipitation_probability_factor': "Effect of rain probability on offense.",
            'temperature_change_factor': "Effect of temperature changes during the game on offense.",
            
            'umpire_strikeout_boost': "Expected increase or decrease in strikeouts due to umpire tendencies.",
            'umpire_runs_boost': "Expected increase or decrease in runs due to umpire tendencies.",
            'umpire_consistency_factor': "How consistent the umpire's strike zone is relative to average.",
            'umpire_fatigue_factor': "Effect of umpire fatigue on game calling.",
            'umpire_home_team_interaction': "Historical interaction between umpire and home team.",
            'umpire_away_team_interaction': "Historical interaction between umpire and away team.",
            'umpire_home_pitcher_interaction': "Historical interaction between umpire and home pitcher.",
            'umpire_away_pitcher_interaction': "Historical interaction between umpire and away pitcher.",
            'umpire_impact_score': "Overall impact of umpire on the game.",
            
            'home_pitcher_context_advantage': "How much the contextual factors advantage the home starting pitcher (-1 to 1 scale).",
            'away_pitcher_context_advantage': "How much the contextual factors advantage the away starting pitcher (-1 to 1 scale).",
            'pitcher_matchup_strikeout_boost': "Expected impact on strikeouts based on pitcher skills and umpire tendencies.",
            'home_pitcher_fatigue': "Effect of home pitcher fatigue on performance.",
            'away_pitcher_fatigue': "Effect of away pitcher fatigue on performance.",
            
            'home_power_context_advantage': "How much the contextual factors boost the home team's power hitting (-1 to 1 scale).",
            'away_power_context_advantage': "How much the contextual factors boost the away team's power hitting (-1 to 1 scale).",
            'home_contact_context_advantage': "How much the contextual factors boost the home team's contact hitting (-1 to 1 scale).",
            'away_contact_context_advantage': "How much the contextual factors boost the away team's contact hitting (-1 to 1 scale).",
            'home_lineup_advantage': "Advantage of home team's lineup in current context.",
            'away_lineup_advantage': "Advantage of away team's lineup in current context.",
            
            'total_runs_context_factor': "Combined impact of all contextual factors on expected total runs (-1 to 1 scale).",
            'total_strikeouts_context_factor': "Combined impact of all contextual factors on expected strikeouts (-1 to 1 scale).",
            'home_advantage_score': "Overall contextual advantage score for the home team (-1 to 1 scale).",
            'away_advantage_score': "Overall contextual advantage score for the away team (-1 to 1 scale).",
            'pitcher_matchup_advantage': "Relative advantage of home pitcher over away pitcher (-1 to 1 scale).",
            'power_matchup_advantage': "Relative advantage of home team's power hitting over away team (-1 to 1 scale).",
            'contact_matchup_advantage': "Relative advantage of home team's contact hitting over away team (-1 to 1 scale).",
            'game_context_score': "Overall game context score combining all factors (-1 to 1 scale)."
        }
        
        return descriptions.get(feature, "No description available.")

    def validate_features(self, games_df):
        """
        Validate the engineered features for quality and consistency
        
        Args:
            games_df (DataFrame): DataFrame with engineered features
            
        Returns:
            dict: Dictionary containing validation results
        """
        validation_results = {
            'missing_values': {},
            'value_ranges': {},
            'correlations': {},
            'warnings': []
        }
        
        # Check for missing values
        for col in games_df.columns:
            missing_count = games_df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(games_df)) * 100
                }
        
        # Validate value ranges
        for col in games_df.columns:
            if col.endswith(('factor', 'score', 'advantage')):
                min_val = games_df[col].min()
                max_val = games_df[col].max()
                if min_val < -1 or max_val > 1:
                    validation_results['value_ranges'][col] = {
                        'min': min_val,
                        'max': max_val,
                        'expected_range': '[-1, 1]'
                    }
        
        # Check correlations with target variables (numeric only)
        target_cols = ['total_runs', 'total_strikeouts', 'home_team_wins']
        numeric_df = games_df.select_dtypes(include=[np.number])
        for target in target_cols:
            if target in numeric_df.columns:
                correlations = numeric_df.corr()[target].sort_values(ascending=False)
                validation_results['correlations'][target] = correlations.head(10).to_dict()
        
        # Generate warnings
        if validation_results['missing_values']:
            validation_results['warnings'].append(
                f"Found {len(validation_results['missing_values'])} features with missing values"
            )
        
        if validation_results['value_ranges']:
            validation_results['warnings'].append(
                f"Found {len(validation_results['value_ranges'])} features with values outside expected range"
            )
        
        return validation_results

    def fetch_and_save_mlb_odds(self, output_csv_path=None, api_key='58b5378fd3d8706963b692f7741eaaac'):
        """
        Fetch current MLB moneyline odds using The Odds API and save as CSV.
        Args:
            output_csv_path (str, optional): Path to save the CSV. If None, uses today's date.
            api_key (str): The Odds API key.
        Returns:
            DataFrame: Odds data (pivoted with home/away odds as columns)
        """
        SPORT = 'baseball_mlb'
        REGION = 'us'
        MARKETS = 'h2h'
        url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
        params = {
            'apiKey': api_key,
            'regions': REGION,
            'markets': MARKETS,
            'oddsFormat': 'american'
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch odds: {response.status_code} {response.text}")
            return None
        odds_json = response.json()
        # Debug: print the first 2 items of the raw API response
        print("\n--- DEBUG: Raw Odds API Response (first 2 items) ---")
        for i, item in enumerate(odds_json[:2]):
            print(f"Item {i}: {item}\n")
        odds_data = []
        for game in odds_json:
            # Use home_team and away_team directly
            if 'home_team' not in game or 'away_team' not in game:
                print(f"Skipping game with missing 'home_team' or 'away_team': {game.get('id', 'unknown')}")
                continue
            game_id = game['id']
            commence_time = game.get('commence_time', None)
            home_team = game['home_team']
            away_team = game['away_team']
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'game_id': game_id,
                                'commence_time': commence_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker': bookmaker.get('title', ''),
                                'team': outcome.get('name', ''),
                                'odds': outcome.get('price', None)
                            })
        odds_df = pd.DataFrame(odds_data)
        if odds_df.empty:
            print("No odds data found.")
            return None
        # Pivot to have home/away odds in columns
        pivot_df = odds_df.pivot_table(
            index=['game_id', 'commence_time', 'home_team', 'away_team', 'bookmaker'],
            columns='team',
            values='odds'
        ).reset_index()
        # Save to CSV
        if output_csv_path is None:
            today = datetime.now().strftime('%Y-%m-%d')
            output_csv_path = f"mlb_moneyline_odds_{today}.csv"
        pivot_df.to_csv(output_csv_path, index=False)
        print(f"Saved MLB moneyline odds to {output_csv_path}")
        print(pivot_df.head())
        return pivot_df

    def merge_odds_and_engineer_edge(self, features_df, odds_df, date_col=None):
        """
        Merge odds with contextual features, calculate implied probabilities, and compute edge features.
        Args:
            features_df (pd.DataFrame): DataFrame with contextual features (must have home_team, away_team)
            odds_df (pd.DataFrame): DataFrame from fetch_and_save_mlb_odds (must have home_team, away_team, home/away odds)
            date_col (str, optional): Name of date column to join on (if available)
        Returns:
            pd.DataFrame: Merged DataFrame with implied probabilities and edge features
        """
        # Print columns and head for debugging
        print("\n--- DEBUG: features_df columns before merge ---")
        print(features_df.columns.tolist())
        print(features_df.head())
        print("\n--- DEBUG: odds_df columns before merge ---")
        print(odds_df.columns.tolist())
        print(odds_df.head())
        # Check for required columns
        for col in ['home_team', 'away_team']:
            if col not in features_df.columns:
                print(f"ERROR: '{col}' not found in features_df. Cannot merge.")
                return features_df
            if col not in odds_df.columns:
                print(f"ERROR: '{col}' not found in odds_df. Cannot merge.")
                return features_df
        # Prepare for merge
        merge_cols = ['home_team', 'away_team']
        if date_col and date_col in features_df.columns and date_col in odds_df.columns:
            merge_cols.append(date_col)
        merged = pd.merge(features_df, odds_df, on=merge_cols, how='left', suffixes=('', '_odds'))
        # Attempt to match home/away team to odds columns
        def get_odds(row, team_col):
            team_name = row[team_col]
            # Try exact match
            if team_name in odds_df.columns:
                return row.get(team_name, np.nan)
            # Try case-insensitive match
            for col in odds_df.columns:
                if isinstance(col, str) and col.lower() == str(team_name).lower():
                    return row.get(col, np.nan)
            # Try partial match (for abbreviations)
            for col in odds_df.columns:
                if isinstance(col, str) and (str(team_name) in col or col in str(team_name)):
                    return row.get(col, np.nan)
            print(f"Warning: Could not find odds column for team '{team_name}' in row {row.name}")
            return np.nan
        merged['home_odds'] = merged.apply(lambda row: get_odds(row, 'home_team'), axis=1)
        merged['away_odds'] = merged.apply(lambda row: get_odds(row, 'away_team'), axis=1)
        # Calculate implied probabilities from American odds
        def american_to_prob(odds):
            if pd.isna(odds):
                return np.nan
            try:
                odds = float(odds)
            except Exception:
                return np.nan
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        merged['implied_home_prob'] = merged['home_odds'].apply(american_to_prob)
        merged['implied_away_prob'] = merged['away_odds'].apply(american_to_prob)
        # Calculate edge (model probability minus implied probability)
        if 'model_home_win_prob' in merged.columns:
            merged['home_edge'] = merged['model_home_win_prob'] - merged['implied_home_prob']
        else:
            merged['home_edge'] = np.nan
        if 'model_away_win_prob' in merged.columns:
            merged['away_edge'] = merged['model_away_win_prob'] - merged['implied_away_prob']
        else:
            merged['away_edge'] = np.nan
        print("Merged odds and engineered implied probabilities and edge features.")
        print(merged[[
            'home_team', 'away_team', 'home_odds', 'away_odds', 'implied_home_prob', 'implied_away_prob', 'home_edge', 'away_edge'
        ]].head())
        return merged

    def simulate_betting(self, merged_df, edge_threshold=0.05, bet_amount=100, odds_col='home'):
        """
        Simulate flat betting on games where model edge exceeds threshold.
        Bets on home team if home_edge > threshold.
        Args:
            merged_df (pd.DataFrame): DataFrame with odds, edge, and actual outcome ('home_team_wins')
            edge_threshold (float): Minimum edge required to place a bet
            bet_amount (float): Amount to bet per game
            odds_col (str): Column name for home odds (default 'home')
        Returns:
            pd.DataFrame: Bets placed with results
        """
        bets = merged_df[merged_df['home_edge'] > edge_threshold].copy()
        if bets.empty:
            print("No value bets found with the given edge threshold.")
            return bets
        # Calculate payout for American odds
        def payout(odds, win):
            if pd.isna(odds):
                return 0.0
            odds = float(odds)
            if win:
                if odds > 0:
                    return bet_amount * odds / 100
                else:
                    return bet_amount * 100 / abs(odds)
            else:
                return -bet_amount
        bets['bet_result'] = bets.apply(
            lambda row: payout(row[odds_col], row['home_team_wins']) if odds_col in row and 'home_team_wins' in row else 0,
            axis=1
        )
        total_bets = len(bets)
        total_profit = bets['bet_result'].sum()
        roi = total_profit / (total_bets * bet_amount) if total_bets > 0 else 0
        print(f"Simulated {total_bets} bets. Total profit: ${total_profit:.2f}, ROI: {roi:.2%}")
        return bets

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

def load_outcomes_data():
    """Load the outcomes data to get the correct game IDs"""
    outcomes_path = "sports_data/mlb/contextual/game_outcomes_2024.csv"
    outcomes_df = pd.read_csv(outcomes_path)
    print(f"\nLoaded {len(outcomes_df)} games from outcomes data")
    print("\nColumns in outcomes data:")
    print(outcomes_df.columns.tolist())
    return outcomes_df

def engineer_contextual_features():
    """Engineer contextual features for each game"""
    # Load outcomes data for game IDs
    outcomes_df = load_outcomes_data()
    
    # Create features dataframe with correct game IDs
    features_df = pd.DataFrame()
    features_df["game_id"] = outcomes_df["game_id"].astype(int)  # Ensure integer type
    features_df["game_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")  # Placeholder
    features_df["home_team"] = outcomes_df["home_team"]  # Updated column name
    features_df["away_team"] = outcomes_df["away_team"]  # Updated column name
    
    # Generate sample contextual features (replace with actual data when available)
    n_games = len(features_df)
    print(f"\nGenerating features for {n_games} games")
    
    # Weather features
    features_df["temperature_f"] = np.random.uniform(50, 85, n_games)
    features_df["humidity_pct"] = np.random.uniform(30, 90, n_games)
    features_df["wind_speed_mph"] = np.random.uniform(0, 15, n_games)
    features_df["wind_direction"] = np.random.choice(["N", "S", "E", "W", "NE", "NW", "SE", "SW"], n_games)
    features_df["precipitation_in"] = np.random.uniform(0, 0.5, n_games)
    features_df["weather_condition"] = np.random.choice(["Clear", "Partly Cloudy", "Cloudy", "Light Rain"], n_games)
    
    # Umpire features
    features_df["umpire_zone_size"] = np.random.uniform(95, 105, n_games)
    features_df["umpire_consistency"] = np.random.uniform(70, 95, n_games)
    features_df["umpire_favor_factor"] = np.random.normal(0, 2, n_games)
    features_df["umpire_runs_impact"] = np.random.normal(0, 0.5, n_games)
    features_df["umpire_strikeout_impact"] = np.random.normal(0, 1, n_games)
    
    # Ballpark features
    features_df["ballpark_run_factor"] = np.random.uniform(-0.1, 0.1, n_games)
    features_df["ballpark_hr_factor"] = np.random.uniform(-0.15, 0.15, n_games)
    features_df["ballpark_hitter_friendly_score"] = np.random.uniform(90, 110, n_games)
    
    # Derived features
    features_df["temp_factor"] = (features_df["temperature_f"] - 70) / 20
    features_df["wind_factor"] = features_df["wind_speed_mph"] / 10
    features_df["humidity_factor"] = (features_df["humidity_pct"] - 60) / 30
    features_df["precipitation_factor"] = -features_df["precipitation_in"]
    
    # Context scores
    features_df["weather_score"] = features_df[["temp_factor", "wind_factor", "humidity_factor", "precipitation_factor"]].mean(axis=1)
    features_df["umpire_strikeout_boost"] = features_df["umpire_strikeout_impact"]
    features_df["umpire_runs_boost"] = features_df["umpire_runs_impact"]
    features_df["umpire_consistency_factor"] = (features_df["umpire_consistency"] - 80) / 15
    
    # Team advantages
    features_df["home_power_context_advantage"] = -1  # Placeholder
    features_df["away_power_context_advantage"] = -1  # Placeholder
    features_df["home_contact_context_advantage"] = -1  # Placeholder
    features_df["away_contact_context_advantage"] = -1  # Placeholder
    
    # Overall factors
    features_df["total_runs_context_factor"] = features_df[["weather_score", "umpire_runs_boost", "ballpark_run_factor"]].mean(axis=1)
    features_df["total_strikeouts_context_factor"] = features_df[["umpire_strikeout_boost", "wind_factor"]].mean(axis=1)
    features_df["home_advantage_score"] = -1  # Placeholder
    features_df["away_advantage_score"] = -1  # Placeholder
    
    # Verify game IDs match
    print("\nSample of generated game IDs:")
    print(features_df["game_id"].head())
    
    # Save features
    features_df.to_csv("data/contextual/games_with_engineered_features.csv", index=False)
    print(f"\nGenerated and saved features for {n_games} games")
    return features_df

if __name__ == "__main__":
    # MLB team abbreviation to full name mapping
    TEAM_ABBR_TO_NAME = {
        'NYY': 'New York Yankees',
        'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs',
        'LAD': 'Los Angeles Dodgers',
        'SF': 'San Francisco Giants',
        'COL': 'Colorado Rockies',
        'ATL': 'Atlanta Braves',
        'HOU': 'Houston Astros',
        'MIA': 'Miami Marlins',
        'TB': 'Tampa Bay Rays',
        'MIL': 'Milwaukee Brewers',
        'SD': 'San Diego Padres',
        'LAA': 'Los Angeles Angels',
        'ARI': 'Arizona Diamondbacks',
        'AZ': 'Arizona Diamondbacks',  # Alternative abbreviation
        'WSH': 'Washington Nationals',
        'TEX': 'Texas Rangers',
        'PHI': 'Philadelphia Phillies',
        'BAL': 'Baltimore Orioles',
        'CLE': 'Cleveland Guardians',
        'MIN': 'Minnesota Twins',
        'DET': 'Detroit Tigers',
        'KC': 'Kansas City Royals',
        'OAK': 'Oakland Athletics',
        'SEA': 'Seattle Mariners',
        'TOR': 'Toronto Blue Jays',
        'CWS': 'Chicago White Sox',
        'STL': 'St. Louis Cardinals',
        'CIN': 'Cincinnati Reds',
        'PIT': 'Pittsburgh Pirates',
        'NYM': 'New York Mets'
    }

    # Add reverse mapping for flexibility
    TEAM_NAME_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}

    engineer = ContextualFeatureEngineer()

    # Load real games data
    real_games_path = 'data/contextual/merged_game_data_2024.csv'
    print(f"Loading real games data from {real_games_path}")
    games_df = pd.read_csv(real_games_path)

    # Print initial column names for debugging
    print("\nInitial columns:", games_df.columns.tolist())

    # Robust team name mapping (abbreviation to full name, or clean up)
    if 'home_team_x' in games_df.columns:
        games_df['home_team'] = games_df['home_team_x'].apply(
            lambda x: TEAM_ABBR_TO_NAME.get(x, x) if isinstance(x, str) else x
        )
    if 'away_team_x' in games_df.columns:
        games_df['away_team'] = games_df['away_team_x'].apply(
            lambda x: TEAM_ABBR_TO_NAME.get(x, x) if isinstance(x, str) else x
        )

    # Remove duplicate team columns to avoid merge errors
    for col in ['home_team_x', 'home_team_y', 'away_team_x', 'away_team_y']:
        if col in games_df.columns:
            games_df = games_df.drop(columns=[col])

    # Print sample of team names for verification
    print("\nSample of mapped team names:")
    print(games_df[['home_team', 'away_team']].head())

    # Add placeholder model predictions if not present
    if 'model_home_win_prob' not in games_df.columns:
        print("Adding placeholder model_home_win_prob (0.5)")
        games_df['model_home_win_prob'] = 0.5
    if 'model_away_win_prob' not in games_df.columns:
        print("Adding placeholder model_away_win_prob (0.5)")
        games_df['model_away_win_prob'] = 0.5
    if 'home_team_wins' not in games_df.columns:
        print("Adding placeholder home_team_wins (0)")
        games_df['home_team_wins'] = 0

    # Engineer contextual features
    print("\n--- Engineering Features ---")
    features_df = engineer.engineer_features(games_df)

    # Standardize team columns for merging
    print("\n--- Standardizing Team Columns ---")
    if 'home_team_x' in features_df.columns:
        features_df = features_df.rename(columns={'home_team_x': 'home_team'})
    if 'away_team_x' in features_df.columns:
        features_df = features_df.rename(columns={'away_team_x': 'away_team'})
    # Drop duplicate columns if present
    for col in ['home_team_y', 'away_team_y']:
        if col in features_df.columns:
            features_df = features_df.drop(columns=[col])
    print("Team columns after standardization:", features_df[['home_team', 'away_team']].head())

    # Fetch current MLB odds
    print("\n--- Fetching MLB Odds ---")
    odds_df = engineer.fetch_and_save_mlb_odds()

    if odds_df is None:
        print("\nNo odds data available. Skipping merging and betting simulation.")
    else:
        # Merge odds and engineer implied probabilities and edge
        print("\n--- Merging Odds and Calculating Edge ---")
        merged_df = engineer.merge_odds_and_engineer_edge(features_df, odds_df)

        # Simulate betting
        print("\n--- Simulating Betting ---")
        bets = engineer.simulate_betting(merged_df)

        # Print sample of bets
        if bets is not None and not bets.empty:
            print("\nSample of bets placed:")
            print(bets[['home_team', 'away_team', 'home_edge', 'home_odds', 'implied_home_prob', 'model_home_win_prob', 'home_team_wins', 'bet_result']].head())
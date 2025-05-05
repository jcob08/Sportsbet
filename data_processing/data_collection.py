import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from dotenv import load_dotenv
import sqlite3
import json
import numpy as np
from bs4 import BeautifulSoup
import re
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Base class for data collection"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to CSV file"""
        filepath = self.data_dir / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")

class MLBStatsCollector(DataCollector):
    """Collects MLB player statistics including advanced metrics"""
    
    # Define valid ranges for advanced metrics
    ADVANCED_METRICS_RANGES = {
        'xwOBA': (0.1, 0.6),           # Expected weighted on-base average
        'xBA': (0.1, 0.4),             # Expected batting average
        'xSLG': (0.1, 0.8),            # Expected slugging percentage
        'xERA': (1.0, 10.0),           # Expected ERA
        'xwOBAcon': (0.1, 0.8),        # Expected wOBA on contact
        'barrel_rate': (0.0, 0.3),     # Barrel rate
        'exit_velocity': (60, 120),     # Exit velocity (mph)
        'launch_angle': (-20, 60),      # Launch angle (degrees)
        'sweet_spot_rate': (0.0, 0.5),  # Sweet spot rate
        'hard_hit_rate': (0.0, 0.6),    # Hard hit rate
        'spin_rate': (1000, 3000),      # Spin rate (rpm)
        'extension': (4.0, 8.0),        # Extension (feet)
        'release_speed': (60, 105),     # Release speed (mph)
        # New: Pitch movement
        'pfx_x': (-25, 25),            # Horizontal movement (inches)
        'pfx_z': (-10, 20),            # Vertical movement (inches)
        'release_side': (-3, 3),       # Release side (feet)
        'release_height': (4, 7),      # Release height (feet)
        'whiff_rate': (0.0, 0.7),      # Whiff rate
        'chase_rate': (0.0, 0.7),      # Chase rate
        'csw_rate': (0.0, 0.6),        # Called Strikes + Whiffs %
        # New: Batted ball profiles
        'gb_rate': (0.0, 0.7),         # Ground ball rate
        'fb_rate': (0.0, 0.7),         # Fly ball rate
        'ld_rate': (0.0, 0.5),         # Line drive rate
        'popup_rate': (0.0, 0.2),      # Pop-up rate
        'pull_rate': (0.0, 0.7),       # Pull %
        'oppo_rate': (0.0, 0.7),       # Oppo %
        'cent_rate': (0.0, 0.7),       # Center %
        'avg_distance': (100, 500),    # Average distance (ft)
        'hr_fb_rate': (0.0, 0.5),      # HR/FB
        'spray_angle': (-45, 45),      # Spray angle (degrees)
    }
    
    # Define required advanced metrics
    REQUIRED_ADVANCED_METRICS = [
        'xwOBA', 'xBA', 'xSLG', 'xERA', 'xwOBAcon', 'barrel_rate', 'exit_velocity',
        'launch_angle', 'sweet_spot_rate', 'hard_hit_rate', 'spin_rate', 'extension',
        'release_speed', 'pfx_x', 'pfx_z', 'release_side', 'release_height',
        'whiff_rate', 'chase_rate', 'csw_rate', 'gb_rate', 'fb_rate', 'ld_rate',
        'popup_rate', 'pull_rate', 'oppo_rate', 'cent_rate', 'avg_distance',
        'hr_fb_rate', 'spray_angle'
    ]
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.statcast_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        
        # Initialize SQLite cache
        self.cache_db = self.data_dir / "advanced_stats_cache.db"
        self._init_cache_db()
        
    def _init_cache_db(self):
        """Initialize the SQLite cache database"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS advanced_stats_cache (
                    player_id TEXT,
                    game_date TEXT,
                    stat_type TEXT,
                    stats_data TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_id, game_date, stat_type)
                )
            """)
            
    def _validate_advanced_metrics(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate advanced metrics against expected ranges"""
        errors = []
        
        # Check for required metrics
        missing_metrics = [metric for metric in self.REQUIRED_ADVANCED_METRICS 
                         if metric not in data]
        if missing_metrics:
            errors.append(f"Missing required metrics: {', '.join(missing_metrics)}")
        
        # Validate numeric fields
        for metric, (min_val, max_val) in self.ADVANCED_METRICS_RANGES.items():
            if metric in data and data[metric] is not None:
                try:
                    value = float(data[metric])
                    if not min_val <= value <= max_val:
                        errors.append(f"{metric} value {value} outside valid range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    errors.append(f"Invalid {metric} value: {data[metric]}")
                    
        return len(errors) == 0, errors
        
    def _get_advanced_stats_from_cache(self, player_id: str, date: str, stat_type: str) -> Optional[Dict]:
        """Get advanced stats from cache with validation"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT stats_data FROM advanced_stats_cache WHERE player_id = ? AND game_date = ? AND stat_type = ?",
                (player_id, date, stat_type)
            )
            result = cursor.fetchone()
            if result:
                data = json.loads(result[0])
                is_valid, errors = self._validate_advanced_metrics(data)
                if is_valid:
                    return data
                else:
                    logger.warning(f"Invalid cached data for player {player_id} on {date}: {', '.join(errors)}")
                    # Remove invalid data from cache
                    conn.execute(
                        "DELETE FROM advanced_stats_cache WHERE player_id = ? AND game_date = ? AND stat_type = ?",
                        (player_id, date, stat_type)
                    )
            return None
            
    def _save_advanced_stats_to_cache(self, player_id: str, date: str, stat_type: str, stats_data: Dict):
        """Save advanced stats to cache with validation"""
        is_valid, errors = self._validate_advanced_metrics(stats_data)
        if not is_valid:
            logger.error(f"Invalid advanced stats for player {player_id} on {date}: {', '.join(errors)}")
            return
            
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO advanced_stats_cache (player_id, game_date, stat_type, stats_data)
                VALUES (?, ?, ?, ?)
                """,
                (player_id, date, stat_type, json.dumps(stats_data))
            )
            
    def _get_statcast_data(self, player_id: str, start_date: str, end_date: str, player_type: str) -> pd.DataFrame:
        """Get Statcast data for a player"""
        params = {
            "all": "true",
            "player_type": player_type,
            "player_id": player_id,
            "start_date": start_date,
            "end_date": end_date,
            "type": "details"
        }
        
        try:
            response = requests.get(self.statcast_url, params=params)
            response.raise_for_status()
            return pd.read_csv(pd.StringIO(response.text))
        except Exception as e:
            logger.error(f"Error fetching Statcast data for player {player_id}: {e}")
            return pd.DataFrame()
            
    def _calculate_advanced_metrics(self, statcast_df: pd.DataFrame, player_type: str) -> Dict:
        """Calculate advanced metrics from Statcast data, including pitch movement and batted ball profiles"""
        if statcast_df.empty:
            return {}
        metrics = {}
        if player_type == "batter":
            # Existing advanced metrics
            metrics['xwOBA'] = statcast_df['estimated_woba_using_speedangle'].mean()
            metrics['xBA'] = statcast_df['estimated_ba_using_speedangle'].mean()
            metrics['xSLG'] = statcast_df['estimated_slg_using_speedangle'].mean()
            metrics['xwOBAcon'] = statcast_df['estimated_woba_using_speedangle'].mean()
            metrics['barrel_rate'] = (statcast_df['barrel'] == 1).mean()
            metrics['exit_velocity'] = statcast_df['launch_speed'].mean()
            metrics['launch_angle'] = statcast_df['launch_angle'].mean()
            metrics['sweet_spot_rate'] = ((statcast_df['launch_angle'] >= 8) & (statcast_df['launch_angle'] <= 32)).mean()
            metrics['hard_hit_rate'] = (statcast_df['launch_speed'] >= 95).mean()
            # Batted ball profiles
            metrics['gb_rate'] = (statcast_df['bb_type'] == 'ground_ball').mean() if 'bb_type' in statcast_df else None
            metrics['fb_rate'] = (statcast_df['bb_type'] == 'fly_ball').mean() if 'bb_type' in statcast_df else None
            metrics['ld_rate'] = (statcast_df['bb_type'] == 'line_drive').mean() if 'bb_type' in statcast_df else None
            metrics['popup_rate'] = (statcast_df['bb_type'] == 'popup').mean() if 'bb_type' in statcast_df else None
            metrics['pull_rate'] = (statcast_df['hit_direction'] == 'pull').mean() if 'hit_direction' in statcast_df else None
            metrics['oppo_rate'] = (statcast_df['hit_direction'] == 'opposite').mean() if 'hit_direction' in statcast_df else None
            metrics['cent_rate'] = (statcast_df['hit_direction'] == 'center').mean() if 'hit_direction' in statcast_df else None
            metrics['avg_distance'] = statcast_df['hit_distance'].mean() if 'hit_distance' in statcast_df else None
            metrics['hr_fb_rate'] = (statcast_df['events'] == 'home_run').sum() / max((statcast_df['bb_type'] == 'fly_ball').sum(), 1) if 'bb_type' in statcast_df and 'events' in statcast_df else None
            metrics['spray_angle'] = statcast_df['spray_angle'].mean() if 'spray_angle' in statcast_df else None
        else:  # pitcher
            metrics['xERA'] = statcast_df['estimated_era_using_speedangle'].mean() if 'estimated_era_using_speedangle' in statcast_df else None
            metrics['spin_rate'] = statcast_df['release_spin_rate'].mean() if 'release_spin_rate' in statcast_df else None
            metrics['extension'] = statcast_df['release_extension'].mean() if 'release_extension' in statcast_df else None
            metrics['release_speed'] = statcast_df['release_speed'].mean() if 'release_speed' in statcast_df else None
            # Pitch movement
            metrics['pfx_x'] = statcast_df['pfx_x'].mean() if 'pfx_x' in statcast_df else None
            metrics['pfx_z'] = statcast_df['pfx_z'].mean() if 'pfx_z' in statcast_df else None
            metrics['release_side'] = statcast_df['release_pos_x'].mean() if 'release_pos_x' in statcast_df else None
            metrics['release_height'] = statcast_df['release_pos_z'].mean() if 'release_pos_z' in statcast_df else None
            # Plate discipline
            metrics['whiff_rate'] = (statcast_df['description'] == 'swinging_strike').mean() if 'description' in statcast_df else None
            metrics['chase_rate'] = (statcast_df['plate_x'].abs() > 0.83).mean() if 'plate_x' in statcast_df else None
            metrics['csw_rate'] = ((statcast_df['description'].isin(['called_strike', 'swinging_strike'])).mean()) if 'description' in statcast_df else None
        return metrics
        
    def get_player_stats(self, 
                        start_date: str, 
                        end_date: str,
                        stat_type: str = "hitting") -> pd.DataFrame:
        """Get player statistics including advanced metrics"""
        # Get traditional stats
        url = f"{self.base_url}/stats"
        params = {
            "sportId": 1,  # MLB
            "group": stat_type,
            "startDate": start_date,
            "endDate": end_date,
            "stats": "season"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            traditional_stats = pd.DataFrame(data["stats"][0]["splits"])
            
            # Get advanced stats for each player
            advanced_stats_list = []
            for _, player in traditional_stats.iterrows():
                player_id = player['player_id']
                
                # Check cache first
                cached_stats = self._get_advanced_stats_from_cache(player_id, start_date, stat_type)
                if cached_stats:
                    advanced_stats_list.append(cached_stats)
                    continue
                    
                # Get Statcast data
                statcast_data = self._get_statcast_data(player_id, start_date, end_date, stat_type)
                if not statcast_data.empty:
                    # Calculate advanced metrics
                    advanced_stats = self._calculate_advanced_metrics(statcast_data, stat_type)
                    if advanced_stats:
                        # Save to cache
                        self._save_advanced_stats_to_cache(player_id, start_date, stat_type, advanced_stats)
                        advanced_stats['player_id'] = player_id
                        advanced_stats_list.append(advanced_stats)
                        
            # Combine traditional and advanced stats
            if advanced_stats_list:
                advanced_stats_df = pd.DataFrame(advanced_stats_list)
                return pd.merge(traditional_stats, advanced_stats_df, on='player_id', how='left')
            else:
                return traditional_stats
                
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return pd.DataFrame()

class StatcastCollector(DataCollector):
    """Collects Statcast data"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        
    def get_statcast_data(self,
                         start_date: str,
                         end_date: str,
                         player_type: str = "batter") -> pd.DataFrame:
        """Get Statcast data for a date range"""
        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfBBT": "",
            "hfPR": "",
            "hfZ": "",
            "stadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfGT": "",
            "hfC": "",
            "hfSea": "",
            "hfSit": "",
            "player_type": player_type,
            "hfOuts": "",
            "hfOpponent": "",
            "hfInn": "",
            "hfTeam": "",
            "hfRO": "",
            "home_road": "",
            "hfFlag": "",
            "hfPull": "",
            "metric_1": "",
            "hfInn": "",
            "min_pitches": 0,
            "min_results": 0,
            "group_by": "name",
            "sort_col": "pitches",
            "player_event_sort": "h_launch_speed",
            "sort_order": "desc",
            "min_abs": 0,
            "type": "details",
            "start_date": start_date,
            "end_date": end_date
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return pd.read_csv(pd.StringIO(response.text))
        except Exception as e:
            logger.error(f"Error fetching Statcast data: {e}")
            return pd.DataFrame()

class WeatherCollector(DataCollector):
    """Collects weather data using OpenWeatherMap API with persistent caching"""
    
    # Define valid ranges for weather metrics
    VALID_RANGES = {
        'temperature': (-50, 120),  # Fahrenheit
        'humidity': (0, 100),       # Percentage
        'wind_speed': (0, 200),     # mph
        'wind_direction': (0, 360), # degrees
        'precipitation': (0, 10),   # inches per hour
        'cloud_cover': (0, 100),    # Percentage
        'pressure': (800, 1100),    # hPa
        'visibility': (0, 10),      # miles
    }
    
    # Define required fields
    REQUIRED_FIELDS = [
        'temperature',
        'humidity',
        'wind_speed',
        'wind_direction',
        'precipitation',
        'cloud_cover',
        'pressure',
        'visibility',
        'weather_condition'
    ]
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not found in environment variables")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Initialize SQLite cache
        self.cache_db = self.data_dir / "weather_cache.db"
        self._init_cache_db()
        
    def _init_cache_db(self):
        """Initialize the SQLite cache database"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_cache (
                    lat REAL,
                    lon REAL,
                    date TEXT,
                    weather_data TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (lat, lon, date)
                )
            """)
            
    def _validate_weather_data(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate weather data against expected ranges and required fields"""
        errors = []
        
        # Check for required fields
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate numeric fields
        for field, (min_val, max_val) in self.VALID_RANGES.items():
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if not min_val <= value <= max_val:
                        errors.append(f"{field} value {value} outside valid range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    errors.append(f"Invalid {field} value: {data[field]}")
        
        # Validate weather condition
        if 'weather_condition' in data and not isinstance(data['weather_condition'], str):
            errors.append("Invalid weather condition format")
            
        return len(errors) == 0, errors
        
    def _get_from_cache(self, lat: float, lon: float, date: str) -> Optional[Dict]:
        """Get weather data from cache with validation"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT weather_data FROM weather_cache WHERE lat = ? AND lon = ? AND date = ?",
                (lat, lon, date)
            )
            result = cursor.fetchone()
            if result:
                data = json.loads(result[0])
                is_valid, errors = self._validate_weather_data(data)
                if is_valid:
                    return data
                else:
                    logger.warning(f"Invalid cached data for {lat}, {lon} on {date}: {', '.join(errors)}")
                    # Remove invalid data from cache
                    conn.execute(
                        "DELETE FROM weather_cache WHERE lat = ? AND lon = ? AND date = ?",
                        (lat, lon, date)
                    )
            return None
            
    def _save_to_cache(self, lat: float, lon: float, date: str, weather_data: Dict):
        """Save weather data to cache with validation"""
        is_valid, errors = self._validate_weather_data(weather_data)
        if not is_valid:
            logger.error(f"Invalid weather data for {lat}, {lon} on {date}: {', '.join(errors)}")
            return
            
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO weather_cache (lat, lon, date, weather_data)
                VALUES (?, ?, ?, ?)
                """,
                (lat, lon, date, json.dumps(weather_data))
            )
            
    def _get_historical_weather(self, lat: float, lon: float, date: str) -> Dict:
        """Get historical weather data for a specific location and date"""
        # Check cache first
        cached_data = self._get_from_cache(lat, lon, date)
        if cached_data:
            logger.info(f"Using cached weather data for {lat}, {lon} on {date}")
            return cached_data
            
        # Convert date to timestamp
        target_date = datetime.strptime(date, "%Y-%m-%d")
        target_timestamp = int(target_date.timestamp())
        
        # Get historical data
        url = f"{self.base_url}/onecall/timemachine"
        params = {
            "lat": lat,
            "lon": lon,
            "dt": target_timestamp,
            "appid": self.api_key,
            "units": "imperial"  # Use Fahrenheit for temperature
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract and validate weather data
            weather_data = {
                "temperature": data["current"]["temp"],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_speed"],
                "wind_direction": data["current"]["wind_deg"],
                "precipitation": data["current"].get("rain", {}).get("1h", 0),
                "cloud_cover": data["current"]["clouds"],
                "pressure": data["current"]["pressure"],
                "visibility": data["current"]["visibility"] / 1609.34,  # Convert meters to miles
                "weather_condition": data["current"]["weather"][0]["main"]
            }
            
            # Validate and save to cache
            is_valid, errors = self._validate_weather_data(weather_data)
            if is_valid:
                self._save_to_cache(lat, lon, date, weather_data)
                logger.info(f"Cached weather data for {lat}, {lon} on {date}")
            else:
                logger.error(f"Invalid weather data from API for {lat}, {lon} on {date}: {', '.join(errors)}")
                
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {
                "temperature": None,
                "humidity": None,
                "wind_speed": None,
                "wind_direction": None,
                "precipitation": None,
                "cloud_cover": None,
                "pressure": None,
                "visibility": None,
                "weather_condition": None
            }
            
    def get_weather_data(self,
                        date: str,
                        lat: float,
                        lon: float) -> Dict:
        """Get weather data for a specific location and date"""
        return self._get_historical_weather(lat, lon, date)
    
    def get_weather_for_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Get weather data for a list of games"""
        if games_df.empty:
            return pd.DataFrame()
            
        weather_data = []
        total_games = len(games_df)
        validation_errors = []
        
        for idx, game in games_df.iterrows():
            logger.info(f"Processing weather data for game {idx + 1}/{total_games}")
            weather = self.get_weather_data(
                game['game_date'],
                game['venue_latitude'],
                game['venue_longitude']
            )
            
            # Validate the data
            is_valid, errors = self._validate_weather_data(weather)
            if not is_valid:
                validation_errors.append(f"Game {game['game_id']}: {', '.join(errors)}")
                continue
                
            weather['game_id'] = game['game_id']
            weather_data.append(weather)
            
            # Add a small delay to respect API rate limits
            time.sleep(0.1)
            
        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} validation errors:\n" + "\n".join(validation_errors))
            
        return pd.DataFrame(weather_data)

class UmpireCollector(DataCollector):
    """Collects umpire data from Baseball Reference with historical analysis and anomaly detection"""
    
    # Define valid ranges for umpire metrics
    VALID_RANGES = {
        'strike_zone_size': (0.5, 1.5),      # Relative to average
        'home_team_bias': (-0.2, 0.2),       # Runs per game
        'strikeout_rate': (0.1, 0.4),        # Strikeouts per plate appearance
        'walk_rate': (0.05, 0.2),            # Walks per plate appearance
        'total_pitches': (200, 400),         # Pitches per game
        'consistency_score': (0.5, 1.0),     # Strike zone consistency
    }
    
    # Define required fields
    REQUIRED_FIELDS = [
        'umpire_id',
        'name',
        'strike_zone_size',
        'home_team_bias',
        'strikeout_rate',
        'walk_rate',
        'total_pitches',
        'consistency_score',
        'last_updated'
    ]
    
    # Define historical analysis periods
    HISTORICAL_PERIODS = {
        'last_7_days': 7,
        'last_30_days': 30,
        'last_90_days': 90,
        'season': 180
    }
    
    # Define anomaly detection parameters
    ANOMALY_THRESHOLD = 0.95  # 95th percentile for statistical anomalies
    ISOLATION_FOREST_CONTAMINATION = 0.05  # Expected proportion of anomalies
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.base_url = "https://www.baseball-reference.com/umpires"
        
        # Initialize SQLite cache
        self.cache_db = self.data_dir / "umpire_cache.db"
        self._init_cache_db()
        
        # Initialize anomaly detection models
        self.anomaly_models = {}
        self.scalers = {}
        
    def _init_cache_db(self):
        """Initialize the SQLite cache database"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS umpire_cache (
                    umpire_id TEXT,
                    game_date TEXT,
                    umpire_data TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (umpire_id, game_date)
                )
            """)
            
    def _get_historical_data(self, umpire_id: str, days_back: int) -> pd.DataFrame:
        """Get historical umpire data for a specific period"""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        with sqlite3.connect(self.cache_db) as conn:
            query = """
                SELECT umpire_data, game_date
                FROM umpire_cache
                WHERE umpire_id = ? AND game_date >= ?
                ORDER BY game_date DESC
            """
            cursor = conn.execute(query, (umpire_id, cutoff_date))
            results = cursor.fetchall()
            
        if not results:
            return pd.DataFrame()
            
        data = []
        for result in results:
            umpire_data = json.loads(result[0])
            umpire_data['game_date'] = result[1]
            data.append(umpire_data)
            
        return pd.DataFrame(data)
        
    def _analyze_trends(self, df: pd.DataFrame, metric: str) -> Dict:
        """Analyze trends for a specific metric"""
        if df.empty or len(df) < 3:
            return {
                'current_value': None,
                'trend': None,
                'slope': None,
                'p_value': None,
                'std_dev': None,
                'mean': None
            }
            
        values = df[metric].values
        dates = pd.to_datetime(df['game_date']).values
        x = np.array([(d - dates[0]).days for d in dates])
        
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Calculate current value and standard deviation
        current_value = values[-1]
        std_dev = np.std(values)
        mean = np.mean(values)
        
        # Determine trend direction
        if p_value < 0.05:  # Statistically significant trend
            trend = "increasing" if slope > 0 else "decreasing"
        else:
            trend = "stable"
            
        return {
            'current_value': current_value,
            'trend': trend,
            'slope': slope,
            'p_value': p_value,
            'std_dev': std_dev,
            'mean': mean
        }
        
    def _detect_statistical_anomalies(self, values: np.ndarray) -> Tuple[List[int], Dict]:
        """Detect statistical anomalies using z-scores and percentiles"""
        if len(values) < 3:
            return [], {}
            
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(values))
        
        # Calculate percentiles
        percentiles = np.array([stats.percentileofscore(values, x) for x in values])
        
        # Identify anomalies
        anomaly_indices = np.where(
            (z_scores > 2.5) |  # More than 2.5 standard deviations
            (percentiles < 2.5) |  # Below 2.5th percentile
            (percentiles > 97.5)  # Above 97.5th percentile
        )[0].tolist()
        
        # Calculate anomaly metrics
        metrics = {
            'z_scores': z_scores.tolist(),
            'percentiles': percentiles.tolist(),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        
        return anomaly_indices, metrics
        
    def _detect_behavioral_anomalies(self, df: pd.DataFrame) -> Tuple[List[int], Dict]:
        """Detect behavioral anomalies using Isolation Forest"""
        if len(df) < 10:  # Need enough data for meaningful detection
            return [], {}
            
        # Select features for anomaly detection
        features = ['strike_zone_size', 'strikeout_rate', 'walk_rate', 'consistency_score']
        feature_data = df[features].dropna()
        
        if len(feature_data) < 10:
            return [], {}
            
        # Scale the data
        if 'behavioral' not in self.scalers:
            self.scalers['behavioral'] = StandardScaler()
            scaled_data = self.scalers['behavioral'].fit_transform(feature_data)
        else:
            scaled_data = self.scalers['behavioral'].transform(feature_data)
            
        # Train or use existing model
        if 'behavioral' not in self.anomaly_models:
            self.anomaly_models['behavioral'] = IsolationForest(
                contamination=self.ISOLATION_FOREST_CONTAMINATION,
                random_state=42
            )
            anomaly_scores = self.anomaly_models['behavioral'].fit_predict(scaled_data)
        else:
            anomaly_scores = self.anomaly_models['behavioral'].predict(scaled_data)
            
        # Identify anomalies (scores of -1)
        anomaly_indices = np.where(anomaly_scores == -1)[0].tolist()
        
        # Calculate anomaly metrics
        metrics = {
            'anomaly_scores': anomaly_scores.tolist(),
            'feature_importance': dict(zip(features, self.anomaly_models['behavioral'].feature_importances_))
        }
        
        return anomaly_indices, metrics
        
    def _detect_fatigue_anomalies(self, df: pd.DataFrame) -> Tuple[List[int], Dict]:
        """Detect anomalies related to umpire fatigue"""
        if len(df) < 5:  # Need enough consecutive games
            return [], {}
            
        # Calculate metrics related to fatigue
        df['games_in_last_5_days'] = df['game_date'].rolling(5).count()
        df['avg_pitches_per_game'] = df['total_pitches'].rolling(5).mean()
        df['consistency_change'] = df['consistency_score'].diff()
        
        # Identify potential fatigue indicators
        fatigue_indices = np.where(
            (df['games_in_last_5_days'] >= 4) &  # 4+ games in 5 days
            (df['avg_pitches_per_game'] > df['avg_pitches_per_game'].quantile(0.75)) &  # High pitch count
            (df['consistency_change'] < -0.1)  # Significant drop in consistency
        )[0].tolist()
        
        metrics = {
            'games_in_last_5_days': df['games_in_last_5_days'].tolist(),
            'avg_pitches_per_game': df['avg_pitches_per_game'].tolist(),
            'consistency_change': df['consistency_change'].tolist()
        }
        
        return fatigue_indices, metrics
        
    def _analyze_anomalies(self, df: pd.DataFrame) -> Dict:
        """Analyze anomalies across different types"""
        if df.empty:
            return {}
            
        anomaly_analysis = {}
        
        # Detect statistical anomalies for each metric
        for metric in self.VALID_RANGES.keys():
            if metric in df.columns:
                values = df[metric].values
                indices, metrics = self._detect_statistical_anomalies(values)
                if indices:
                    anomaly_analysis[f'statistical_{metric}'] = {
                        'indices': indices,
                        'metrics': metrics,
                        'type': 'statistical'
                    }
                    
        # Detect behavioral anomalies
        indices, metrics = self._detect_behavioral_anomalies(df)
        if indices:
            anomaly_analysis['behavioral'] = {
                'indices': indices,
                'metrics': metrics,
                'type': 'behavioral'
            }
            
        # Detect fatigue anomalies
        indices, metrics = self._detect_fatigue_anomalies(df)
        if indices:
            anomaly_analysis['fatigue'] = {
                'indices': indices,
                'metrics': metrics,
                'type': 'fatigue'
            }
            
        return anomaly_analysis
        
    def _get_trend_analysis(self, umpire_id: str) -> Dict:
        """Get trend analysis for all metrics with anomaly detection"""
        trend_analysis = {}
        
        for period_name, days in self.HISTORICAL_PERIODS.items():
            historical_data = self._get_historical_data(umpire_id, days)
            if historical_data.empty:
                continue
                
            period_analysis = {}
            for metric in self.VALID_RANGES.keys():
                if metric in historical_data.columns:
                    period_analysis[metric] = self._analyze_trends(historical_data, metric)
                    
            # Add anomaly analysis
            period_analysis['anomalies'] = self._analyze_anomalies(historical_data)
            
            trend_analysis[period_name] = period_analysis
            
        return trend_analysis
        
    def _scrape_umpire_page(self, umpire_id: str) -> Dict:
        """Scrape umpire data from Baseball Reference"""
        url = f"{self.base_url}/{umpire_id}.shtml"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract basic information
            name = soup.find('h1').text.strip()
            
            # Extract strike zone metrics
            strike_zone_table = soup.find('table', {'id': 'strike_zone'})
            if strike_zone_table:
                strike_zone_size = float(strike_zone_table.find('td', {'data-stat': 'sz_size'}).text)
                consistency_score = float(strike_zone_table.find('td', {'data-stat': 'consistency'}).text)
            else:
                strike_zone_size = None
                consistency_score = None
                
            # Extract game statistics
            stats_table = soup.find('table', {'id': 'umpire_stats'})
            if stats_table:
                home_team_bias = float(stats_table.find('td', {'data-stat': 'home_bias'}).text)
                strikeout_rate = float(stats_table.find('td', {'data-stat': 'k_rate'}).text)
                walk_rate = float(stats_table.find('td', {'data-stat': 'bb_rate'}).text)
                total_pitches = int(stats_table.find('td', {'data-stat': 'total_pitches'}).text)
            else:
                home_team_bias = None
                strikeout_rate = None
                walk_rate = None
                total_pitches = None
                
            # Get trend analysis
            trend_analysis = self._get_trend_analysis(umpire_id)
                
            umpire_data = {
                'umpire_id': umpire_id,
                'name': name,
                'strike_zone_size': strike_zone_size,
                'home_team_bias': home_team_bias,
                'strikeout_rate': strikeout_rate,
                'walk_rate': walk_rate,
                'total_pitches': total_pitches,
                'consistency_score': consistency_score,
                'last_updated': datetime.now().strftime("%Y-%m-%d"),
                'trend_analysis': trend_analysis
            }
            
            return umpire_data
            
        except Exception as e:
            logger.error(f"Error scraping umpire data for {umpire_id}: {e}")
            return None
            
    def get_umpire_stats(self, umpire_id: str) -> Dict:
        """Get umpire statistics and tendencies with historical analysis"""
        # Check cache first
        today = datetime.now().strftime("%Y-%m-%d")
        cached_data = self._get_from_cache(umpire_id, today)
        if cached_data:
            logger.info(f"Using cached data for umpire {umpire_id}")
            return cached_data
            
        # Scrape new data
        umpire_data = self._scrape_umpire_page(umpire_id)
        if umpire_data:
            self._save_to_cache(umpire_id, today, umpire_data)
            logger.info(f"Cached data for umpire {umpire_id}")
            
        return umpire_data or {}
        
    def get_umpires_for_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Get umpire data for a list of games with historical analysis"""
        if games_df.empty:
            return pd.DataFrame()
            
        umpire_data = []
        total_games = len(games_df)
        validation_errors = []
        
        for idx, game in games_df.iterrows():
            logger.info(f"Processing umpire data for game {idx + 1}/{total_games}")
            if 'umpire_id' not in game:
                logger.warning(f"No umpire_id found for game {game['game_id']}")
                continue
                
            umpire = self.get_umpire_stats(game['umpire_id'])
            
            # Validate the data
            is_valid, errors = self._validate_umpire_data(umpire)
            if not is_valid:
                validation_errors.append(f"Game {game['game_id']}: {', '.join(errors)}")
                continue
                
            umpire['game_id'] = game['game_id']
            umpire_data.append(umpire)
            
            # Add a small delay to respect rate limits
            time.sleep(0.5)
            
        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} validation errors:\n" + "\n".join(validation_errors))
            
        return pd.DataFrame(umpire_data)

def collect_daily_data():
    """Main function to collect all daily data, including advanced metrics"""
    # Set up collectors
    mlb_collector = MLBStatsCollector("data/raw/player_stats")
    statcast_collector = StatcastCollector("data/raw/statcast")
    weather_collector = WeatherCollector("data/contextual/weather_cache")
    umpire_collector = UmpireCollector("data/contextual/umpires")
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Collect player stats (with advanced metrics)
    logger.info("Collecting player stats (with advanced metrics)...")
    player_stats = mlb_collector.get_player_stats(today, today)
    mlb_collector.save_data(player_stats, f"player_stats_{today}.csv")
    # Save advanced stats to processed directory for modeling
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    player_stats.to_csv(processed_dir / f"advanced_player_stats_{today}.csv", index=False)
    
    # Collect Statcast data (raw)
    logger.info("Collecting Statcast data...")
    statcast_data = statcast_collector.get_statcast_data(today, today)
    statcast_collector.save_data(statcast_data, f"statcast_{today}.csv")
    
    # Collect weather data
    logger.info("Collecting weather data...")
    ballparks_df = pd.read_csv("data/contextual/ballpark_database_with_coords.csv")
    weather_data = weather_collector.get_weather_for_games(ballparks_df)
    weather_collector.save_data(weather_data, f"weather_{today}.csv")
    
    # Collect umpire data
    logger.info("Collecting umpire data...")
    # Load games with umpire assignments
    games_df = pd.read_csv("data/contextual/games_with_umpires.csv")
    umpire_data = umpire_collector.get_umpires_for_games(games_df)
    umpire_collector.save_data(umpire_data, f"umpires_{today}.csv")

if __name__ == "__main__":
    collect_daily_data() 
import os
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

class WeatherDataIntegrator:
    """
    Class to fetch and integrate weather data for MLB games
    """
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the weather data integrator
        
        Args:
            api_key (str, optional): API key for weather service
            cache_dir (str, optional): Directory to cache weather data
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get('WEATHER_API_KEY')
        
        if not self.api_key:
            print("Warning: No weather API key provided. Please set WEATHER_API_KEY environment variable.")
            print("Using mock weather data for demonstration purposes.")
        
        # Set up cache directory
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(base_dir, 'data', 'contextual', 'weather_cache')
        else:
            self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load ballpark database for location information
        self.load_ballpark_data()
    
    def load_ballpark_data(self):
        """
        Load ballpark database with location information
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ballpark_file = os.path.join(base_dir, 'data', 'contextual', 'ballpark_database.csv')
        
        if os.path.exists(ballpark_file):
            self.ballparks_df = pd.read_csv(ballpark_file)
            print(f"Loaded ballpark data with {len(self.ballparks_df)} stadiums")
            
            # Add latitude and longitude for each ballpark
            # This would be better sourced from a geocoding API or more precise data
            # Using approximate values for demonstration
            ballpark_locations = {
                'ARI': {'lat': 33.4452, 'lon': -112.0667},  # Chase Field
                'ATL': {'lat': 33.8911, 'lon': -84.4680},   # Truist Park
                'BAL': {'lat': 39.2839, 'lon': -76.6216},   # Camden Yards
                'BOS': {'lat': 42.3467, 'lon': -71.0972},   # Fenway Park
                'CHC': {'lat': 41.9484, 'lon': -87.6553},   # Wrigley Field
                'CWS': {'lat': 41.8299, 'lon': -87.6338},   # Guaranteed Rate Field
                'CIN': {'lat': 39.0979, 'lon': -84.5082},   # Great American Ball Park
                'CLE': {'lat': 41.4962, 'lon': -81.6852},   # Progressive Field
                'COL': {'lat': 39.7559, 'lon': -104.9942},  # Coors Field
                'DET': {'lat': 42.3390, 'lon': -83.0485},   # Comerica Park
                'HOU': {'lat': 29.7573, 'lon': -95.3555},   # Minute Maid Park
                'KC': {'lat': 39.0517, 'lon': -94.4803},    # Kauffman Stadium
                'LAA': {'lat': 33.8003, 'lon': -117.8827},  # Angel Stadium
                'LAD': {'lat': 34.0739, 'lon': -118.2400},  # Dodger Stadium
                'MIA': {'lat': 25.7781, 'lon': -80.2197},   # LoanDepot Park
                'MIL': {'lat': 43.0280, 'lon': -87.9712},   # American Family Field
                'MIN': {'lat': 44.9817, 'lon': -93.2778},   # Target Field
                'NYM': {'lat': 40.7571, 'lon': -73.8458},   # Citi Field
                'NYY': {'lat': 40.8296, 'lon': -73.9262},   # Yankee Stadium
                'OAK': {'lat': 37.7516, 'lon': -122.2005},  # Oakland Coliseum
                'PHI': {'lat': 39.9061, 'lon': -75.1665},   # Citizens Bank Park
                'PIT': {'lat': 40.4468, 'lon': -80.0056},   # PNC Park
                'SD': {'lat': 32.7073, 'lon': -117.1566},   # Petco Park
                'SEA': {'lat': 47.5914, 'lon': -122.3425},  # T-Mobile Park
                'SF': {'lat': 37.7786, 'lon': -122.3893},   # Oracle Park
                'STL': {'lat': 38.6226, 'lon': -90.1928},   # Busch Stadium
                'TB': {'lat': 27.7682, 'lon': -82.6534},    # Tropicana Field
                'TEX': {'lat': 32.7512, 'lon': -97.0832},   # Globe Life Field
                'TOR': {'lat': 43.6414, 'lon': -79.3894},   # Rogers Centre
                'WSH': {'lat': 38.8730, 'lon': -77.0074}    # Nationals Park
            }
            
            # Add lat/lon to dataframe
            self.ballparks_df['latitude'] = self.ballparks_df['team_code'].map(
                lambda x: ballpark_locations.get(x, {}).get('lat', None)
            )
            self.ballparks_df['longitude'] = self.ballparks_df['team_code'].map(
                lambda x: ballpark_locations.get(x, {}).get('lon', None)
            )
            
            # Save updated ballpark data with coordinates
            updated_ballpark_file = os.path.join(base_dir, 'data', 'contextual', 'ballpark_database_with_coords.csv')
            self.ballparks_df.to_csv(updated_ballpark_file, index=False)
            print(f"Updated ballpark data with coordinates saved to {updated_ballpark_file}")
        else:
            print(f"Ballpark database not found at {ballpark_file}")
            print("Please run ballpark_database.py first to create the database")
            self.ballparks_df = None
    
    def get_location_for_team(self, team_code):
        """
        Get latitude and longitude for a team's ballpark
        
        Args:
            team_code (str): Team code (e.g. 'NYY')
            
        Returns:
            tuple: (latitude, longitude) or (None, None) if not found
        """
        if self.ballparks_df is None:
            return None, None
        
        team_data = self.ballparks_df[self.ballparks_df['team_code'] == team_code]
        if len(team_data) > 0:
            lat = team_data['latitude'].iloc[0]
            lon = team_data['longitude'].iloc[0]
            return lat, lon
        else:
            return None, None
    
    def fetch_historical_weather(self, team_code, game_date, game_time='19:00'):
        """
        Fetch historical weather data for a specific game
        
        Args:
            team_code (str): Team code (e.g. 'NYY')
            game_date (str): Date of the game in 'YYYY-MM-DD' format
            game_time (str): Time of the game in 'HH:MM' 24-hour format
            
        Returns:
            dict: Weather data for the game
        """
        # Check if weather data is already cached
        cache_file = os.path.join(self.cache_dir, f"{team_code}_{game_date}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Get location data for the team
        lat, lon = self.get_location_for_team(team_code)
        if lat is None or lon is None:
            print(f"No location data found for team {team_code}")
            return self._generate_mock_weather_data(team_code, game_date)
        
        # If we don't have an API key, use mock data
        if not self.api_key:
            return self._generate_mock_weather_data(team_code, game_date)
        
        # Formulate the datetime string for the API
        datetime_str = f"{game_date} {game_time}"
        
        # This is a placeholder for the actual API call
        # In practice, you would use a weather API like OpenWeatherMap, WeatherAPI, etc.
        # Example with WeatherAPI.com historical data:
        try:
            url = f"http://api.weatherapi.com/v1/history.json?key={self.api_key}&q={lat},{lon}&dt={game_date}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process the response to extract relevant weather data
            # This would depend on the API you're using
            weather_data = {
                'team_code': team_code,
                'game_date': game_date,
                'game_time': game_time,
                'temperature_f': None,  # Extract from API response
                'humidity_pct': None,   # Extract from API response
                'wind_speed_mph': None, # Extract from API response
                'wind_direction': None, # Extract from API response
                'precipitation_in': None, # Extract from API response
                'pressure_mb': None,    # Extract from API response
                'weather_condition': None, # Extract from API response
                'data_source': 'weather_api'
            }
            
            # Cache the weather data
            with open(cache_file, 'w') as f:
                json.dump(weather_data, f)
            
            return weather_data
        
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._generate_mock_weather_data(team_code, game_date)
    
    def _generate_mock_weather_data(self, team_code, game_date):
        """
        Generate mock weather data for demonstration purposes
        
        Args:
            team_code (str): Team code
            game_date (str): Game date
            
        Returns:
            dict: Simulated weather data
        """
        # Create a deterministic random seed based on team and date
        # so we get consistent mock data for the same queries
        seed = hash(f"{team_code}_{game_date}") % 10000
        np.random.seed(seed)
        
        # Get month for seasonal variations
        month = int(game_date.split('-')[1])
        
        # Seasonal temperature adjustment (warmer in summer, cooler in spring/fall)
        temp_adjustment = 0
        if month in [4, 9, 10]:  # Spring/Fall
            temp_adjustment = -5
        elif month in [6, 7, 8]:  # Summer
            temp_adjustment = 10
        
        # Get ballpark elevation for temperature adjustment
        elevation_adjustment = 0
        if self.ballparks_df is not None:
            team_data = self.ballparks_df[self.ballparks_df['team_code'] == team_code]
            if len(team_data) > 0:
                elevation = team_data['elevation_feet'].iloc[0]
                # Temperature decreases about 3.5°F per 1000 feet
                elevation_adjustment = -3.5 * (elevation / 1000)
        
        # Get environment type (dome/retractable/outdoor)
        environment = 'outdoor'
        if self.ballparks_df is not None:
            team_data = self.ballparks_df[self.ballparks_df['team_code'] == team_code]
            if len(team_data) > 0:
                environment = team_data['environment_type'].iloc[0]
        
        # Base weather conditions by region
        # This is a simplistic model - in reality, you'd want more nuanced data
        regional_conditions = {
            'Northeast': {'base_temp': 65, 'humidity': 60, 'wind': 8},     # NYY, NYM, BOS, PHI, PIT, WSH
            'Midwest': {'base_temp': 68, 'humidity': 65, 'wind': 10},      # CHC, CWS, CLE, DET, MIN, MIL, STL, CIN
            'South': {'base_temp': 78, 'humidity': 75, 'wind': 7},         # ATL, MIA, TB, TEX, HOU
            'West': {'base_temp': 72, 'humidity': 45, 'wind': 9},          # LAD, LAA, SF, SD, SEA, OAK, ARI, COL
            'Canada': {'base_temp': 62, 'humidity': 60, 'wind': 8}         # TOR
        }
        
        # Map teams to regions
        team_regions = {
            'NYY': 'Northeast', 'NYM': 'Northeast', 'BOS': 'Northeast', 'PHI': 'Northeast', 
            'PIT': 'Northeast', 'WSH': 'Northeast', 'BAL': 'Northeast',
            'CHC': 'Midwest', 'CWS': 'Midwest', 'CLE': 'Midwest', 'DET': 'Midwest', 
            'MIN': 'Midwest', 'MIL': 'Midwest', 'STL': 'Midwest', 'CIN': 'Midwest', 'KC': 'Midwest',
            'ATL': 'South', 'MIA': 'South', 'TB': 'South', 'TEX': 'South', 'HOU': 'South',
            'LAD': 'West', 'LAA': 'West', 'SF': 'West', 'SD': 'West', 'SEA': 'West', 
            'OAK': 'West', 'ARI': 'West', 'COL': 'West',
            'TOR': 'Canada'
        }
        
        # Get base conditions for the team's region
        region = team_regions.get(team_code, 'Midwest')  # Default to Midwest if not found
        base_conditions = regional_conditions[region]
        
        # Override for dome stadiums
        if environment == 'dome':
            weather_data = {
                'team_code': team_code,
                'game_date': game_date,
                'game_time': '19:00',
                'temperature_f': 72,  # Typical dome temperature
                'humidity_pct': 50,
                'wind_speed_mph': 0,
                'wind_direction': 'N/A',
                'precipitation_in': 0,
                'pressure_mb': 1012,
                'weather_condition': 'Dome - Indoor',
                'data_source': 'mock'
            }
        else:
            # Generate realistic variations for outdoor and retractable roof stadiums
            base_temp = base_conditions['base_temp'] + temp_adjustment + elevation_adjustment
            
            weather_data = {
                'team_code': team_code,
                'game_date': game_date,
                'game_time': '19:00',
                'temperature_f': round(base_temp + np.random.normal(0, 5), 1),
                'humidity_pct': round(base_conditions['humidity'] + np.random.normal(0, 10)),
                'wind_speed_mph': round(base_conditions['wind'] + np.random.exponential(3), 1),
                'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
                'precipitation_in': round(max(0, np.random.exponential(0.1) - 0.05), 2),
                'pressure_mb': round(1013 + np.random.normal(0, 3)),
                'data_source': 'mock'
            }
            
            # Determine weather condition based on humidity, temp, and precipitation
            precip = weather_data['precipitation_in']
            humidity = weather_data['humidity_pct']
            temp = weather_data['temperature_f']
            
            if precip > 0.1:
                weather_data['weather_condition'] = 'Rain'
            elif precip > 0:
                weather_data['weather_condition'] = 'Light Rain'
            elif humidity > 80 and temp < 60:
                weather_data['weather_condition'] = 'Fog'
            elif humidity < 40 and temp > 75:
                weather_data['weather_condition'] = 'Clear and Dry'
            elif temp > 85:
                weather_data['weather_condition'] = 'Hot and Humid' if humidity > 60 else 'Hot'
            elif temp < 50:
                weather_data['weather_condition'] = 'Cold'
            else:
                weather_data['weather_condition'] = 'Partly Cloudy' if np.random.rand() > 0.5 else 'Clear'
        
        # For retractable roofs, indicate if roof is likely closed
        if environment == 'retractable':
            # Assume roof is closed for rain, extreme temperatures, or high winds
            if (weather_data['weather_condition'] in ['Rain', 'Light Rain'] or 
                weather_data['temperature_f'] < 40 or 
                weather_data['temperature_f'] > 95 or
                weather_data['wind_speed_mph'] > 20):
                weather_data['weather_condition'] = f"Roof Likely Closed - {weather_data['weather_condition']}"
        
        # Cache the mock weather data
        cache_file = os.path.join(self.cache_dir, f"{team_code}_{game_date}.json")
        with open(cache_file, 'w') as f:
            json.dump(weather_data, f)
        
        return weather_data
    
    def integrate_weather_with_game_data(self, games_df, output_file=None):
        """
        Integrate weather data with game data
        
        Args:
            games_df (DataFrame): DataFrame with game data (must have home_team and game_date columns)
            output_file (str, optional): File to save integrated data
            
        Returns:
            DataFrame: Games with weather data
        """
        if 'home_team' not in games_df.columns or 'game_date' not in games_df.columns:
            print("Error: games_df must have 'home_team' and 'game_date' columns")
            return games_df
        
        print(f"Integrating weather data for {len(games_df)} games...")
        
        # Create a copy of the game data
        games_with_weather = games_df.copy()
        
        # Add weather data columns
        weather_columns = [
            'temperature_f', 'humidity_pct', 'wind_speed_mph', 
            'wind_direction', 'precipitation_in', 'weather_condition'
        ]
        
        for col in weather_columns:
            games_with_weather[col] = None
        
        # Process each game
        for idx, game in games_with_weather.iterrows():
            # Get home team (may need to convert team_id to team_code)
            home_team = game['home_team']
            
            # Simple mapping if needed (adjust based on your data format)
            # This converts numeric team IDs or full names to team codes if needed
            team_code_map = {
                # Add mappings if your team identifiers don't match the codes in ballpark DB
                # e.g. '1': 'NYY', 'New York Yankees': 'NYY'
            }
            
            team_code = team_code_map.get(str(home_team), home_team)
            
            # Format game date if needed
            game_date = game['game_date']
            if isinstance(game_date, pd.Timestamp):
                game_date = game_date.strftime('%Y-%m-%d')
            
            # Get game time if available
            game_time = game.get('game_time', '19:00')  # Default to 7 PM if not specified
            
            # Fetch weather data
            weather_data = self.fetch_historical_weather(team_code, game_date, game_time)
            
            # Add weather data to the game record
            for col in weather_columns:
                if col in weather_data:
                    games_with_weather.loc[idx, col] = weather_data[col]
        
        # Save to file if requested
        if output_file:
            games_with_weather.to_csv(output_file, index=False)
            print(f"Saved games with weather data to {output_file}")
        
        return games_with_weather

def test_weather_integration():
    """
    Test the weather data integration
    """
    # Create a sample games DataFrame
    games_data = {
        'game_id': range(1, 11),
        'game_date': [f'2024-04-{i:02d}' for i in range(1, 11)],
        'home_team': ['NYY', 'BOS', 'CHC', 'LAD', 'SF', 'COL', 'ATL', 'HOU', 'MIA', 'TB'],
        'away_team': ['BOS', 'NYY', 'MIL', 'SD', 'LAA', 'ARI', 'WSH', 'TEX', 'PHI', 'BAL']
    }
    
    games_df = pd.DataFrame(games_data)
    
    # Initialize the weather integrator
    weather_integrator = WeatherDataIntegrator()
    
    # Set output path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_weather.csv')
    
    # Integrate weather data
    games_with_weather = weather_integrator.integrate_weather_with_game_data(games_df, output_file)
    
    # Print sample of integrated data
    print("\nSample of games with weather data:")
    print(games_with_weather.head())

if __name__ == "__main__":
    test_weather_integration()
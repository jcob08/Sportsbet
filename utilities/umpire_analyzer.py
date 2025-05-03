import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime, timedelta

class UmpireAnalyzer:
    """
    Class to analyze MLB umpire tendencies and integrate them with game data
    """
    def __init__(self, data_dir=None):
        """
        Initialize the umpire analyzer
        
        Args:
            data_dir (str, optional): Base directory for data files
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, 'data', 'contextual')
        else:
            self.data_dir = data_dir
        
        # Create umpire directory if it doesn't exist
        self.umpire_dir = os.path.join(self.data_dir, 'umpires')
        os.makedirs(self.umpire_dir, exist_ok=True)
        
        # Initialize umpire database
        self.umpire_db = self._create_umpire_database()
    
    def _create_umpire_database(self):
        """
        Create a database of MLB umpires with their tendencies
        
        Returns:
            DataFrame: Umpire database
        """
        # Check if umpire database already exists
        umpire_file = os.path.join(self.umpire_dir, 'umpire_database.csv')
        
        if os.path.exists(umpire_file):
            print(f"Loading existing umpire database from {umpire_file}")
            return pd.read_csv(umpire_file)
        
        print("Creating MLB umpire database...")
        
        # In a real implementation, you'd scrape this data from sources like:
        # - Baseball Savant
        # - UmpireScorecards.com
        # - Baseball Reference
        # For this example, we'll use mock data for the full-time MLB umpires
        
        # List of active MLB umpires (as of 2024)
        umpire_names = [
            'Mark Carlson', 'Gabe Morales', 'Todd Tichenor', 'Brian Knight',
            'Tom Hallion', 'Angel Hernandez', 'Joe West', 'CB Bucknor',
            'Doug Eddings', 'Bill Miller', 'Mike Winters', 'Tim Timmons',
            'Alfonso Marquez', 'Larry Vanover', 'Jerry Meals', 'Ron Kulpa',
            'Jim Reynolds', 'Jeff Nelson', 'Laz Diaz', 'Vic Carapazza',
            'Sam Holbrook', 'Dan Iassogna', 'James Hoye', 'Bill Welke',
            'Mike Muchlinski', 'Tony Randazzo', 'Lance Barrett', 'Cory Blaser',
            'Mike Estabrook', 'Greg Gibson', 'Marvin Hudson', 'Adrian Johnson'
        ]
        
        # Create mock umpire data
        umpire_data = []
        np.random.seed(42)  # For reproducibility
        
        for ump_id, name in enumerate(umpire_names, 1):
            # Generate realistic umpire tendencies
            # Strike zone size varies from tight to wide
            zone_size = np.random.normal(100, 8)  # 100 is average, lower is tighter
            
            # Strike zone consistency (how predictable their calls are)
            consistency = np.random.normal(85, 7)
            
            # Favor hitter or pitcher (negative favors hitters, positive favors pitchers)
            favor_factor = np.random.normal(0, 3)
            
            # Generate additional metrics
            correct_call_pct = np.random.normal(94, 1.5)  # Percentage of correct calls
            avg_game_length = np.random.normal(180, 10)  # Average game length in minutes
            
            # Calculate runs impact
            # This represents how many more/fewer runs scored with this umpire
            # compared to average, per game
            runs_impact = -0.2 * favor_factor + np.random.normal(0, 0.3)
            
            # Calculate strikeout impact
            # This represents how many more/fewer strikeouts with this umpire
            # compared to average, per game
            strikeout_impact = 0.5 * favor_factor + 0.2 * (zone_size - 100) + np.random.normal(0, 0.4)
            
            # Generate career stats
            years_experience = np.random.randint(1, 30)
            career_games = years_experience * np.random.randint(90, 140)
            
            umpire_data.append({
                'umpire_id': ump_id,
                'name': name,
                'years_experience': years_experience,
                'career_games': career_games,
                'zone_size': round(zone_size, 1),
                'consistency': round(consistency, 1),
                'favor_factor': round(favor_factor, 2),
                'correct_call_pct': round(correct_call_pct, 1),
                'avg_game_length_min': round(avg_game_length, 1),
                'runs_impact': round(runs_impact, 2),
                'strikeout_impact': round(strikeout_impact, 2)
            })
        
        # Convert to DataFrame
        umpires_df = pd.DataFrame(umpire_data)
        
        # Add categorical versions of key metrics
        umpires_df['zone_size_category'] = pd.cut(
            umpires_df['zone_size'],
            bins=[0, 95, 103, 200],
            labels=['tight', 'average', 'wide']
        )
        
        umpires_df['consistency_category'] = pd.cut(
            umpires_df['consistency'],
            bins=[0, 80, 90, 100],
            labels=['inconsistent', 'average', 'consistent']
        )
        
        umpires_df['favor_category'] = pd.cut(
            umpires_df['favor_factor'],
            bins=[-10, -1.5, 1.5, 10],
            labels=['hitter_friendly', 'neutral', 'pitcher_friendly']
        )
        
        # Save the umpire database
        umpires_df.to_csv(umpire_file, index=False)
        print(f"Umpire database created and saved to {umpire_file}")
        
        # Print some summary statistics
        print(f"Total umpires: {len(umpires_df)}")
        
        print("\nMost pitcher-friendly umpires:")
        print(umpires_df.sort_values('favor_factor', ascending=False)[['name', 'favor_factor', 'strikeout_impact']].head(5))
        
        print("\nMost hitter-friendly umpires:")
        print(umpires_df.sort_values('favor_factor')[['name', 'favor_factor', 'runs_impact']].head(5))
        
        return umpires_df
    
    def get_umpire_by_name(self, name):
        """
        Get umpire data by name
        
        Args:
            name (str): Umpire name (can be partial match)
            
        Returns:
            Series: Umpire data or None if not found
        """
        matches = self.umpire_db[self.umpire_db['name'].str.contains(name, case=False)]
        if len(matches) > 0:
            return matches.iloc[0]
        return None
    
    def get_umpire_by_id(self, umpire_id):
        """
        Get umpire data by ID
        
        Args:
            umpire_id (int): Umpire ID
            
        Returns:
            Series: Umpire data or None if not found
        """
        matches = self.umpire_db[self.umpire_db['umpire_id'] == umpire_id]
        if len(matches) > 0:
            return matches.iloc[0]
        return None
    
    def integrate_umpire_data_with_games(self, games_df, output_file=None):
        """
        Integrate umpire data with game data
        
        Args:
            games_df (DataFrame): DataFrame with game data (must have home_plate_umpire column)
            output_file (str, optional): File to save integrated data
            
        Returns:
            DataFrame: Games with umpire data
        """
        if 'home_plate_umpire' not in games_df.columns:
            print("Warning: 'home_plate_umpire' column not found in games_df")
            print("Adding random umpire assignments for demonstration purposes")
            
            # Create a copy of the games DataFrame
            games_with_umpires = games_df.copy()
            
            # Assign random umpires to games
            umpire_ids = self.umpire_db['umpire_id'].tolist()
            umpire_names = self.umpire_db['name'].tolist()
            
            # Create a mapping of IDs to names
            umpire_map = dict(zip(umpire_ids, umpire_names))
            
            # Randomly assign umpires
            np.random.seed(42)
            games_with_umpires['home_plate_umpire_id'] = np.random.choice(
                umpire_ids, size=len(games_with_umpires)
            )
            games_with_umpires['home_plate_umpire'] = games_with_umpires['home_plate_umpire_id'].map(umpire_map)
        else:
            games_with_umpires = games_df.copy()
        
        print(f"Integrating umpire data for {len(games_with_umpires)} games...")
        
        # Initialize columns for umpire metrics
        umpire_metrics = [
            'umpire_zone_size', 'umpire_consistency', 'umpire_favor_factor', 
            'umpire_runs_impact', 'umpire_strikeout_impact', 'umpire_zone_category',
            'umpire_consistency_category', 'umpire_favor_category'
        ]
        
        for metric in umpire_metrics:
            games_with_umpires[metric] = None
        
        # Add umpire data to each game
        for idx, game in games_with_umpires.iterrows():
            umpire_id = game.get('home_plate_umpire_id')
            umpire_name = game.get('home_plate_umpire')
            
            # Get umpire data
            umpire_data = None
            if umpire_id is not None:
                umpire_data = self.get_umpire_by_id(umpire_id)
            elif umpire_name is not None:
                umpire_data = self.get_umpire_by_name(umpire_name)
            
            # If umpire data found, add metrics to game
            if umpire_data is not None:
                games_with_umpires.loc[idx, 'umpire_zone_size'] = umpire_data['zone_size']
                games_with_umpires.loc[idx, 'umpire_consistency'] = umpire_data['consistency']
                games_with_umpires.loc[idx, 'umpire_favor_factor'] = umpire_data['favor_factor']
                games_with_umpires.loc[idx, 'umpire_runs_impact'] = umpire_data['runs_impact']
                games_with_umpires.loc[idx, 'umpire_strikeout_impact'] = umpire_data['strikeout_impact']
                games_with_umpires.loc[idx, 'umpire_zone_category'] = umpire_data['zone_size_category']
                games_with_umpires.loc[idx, 'umpire_consistency_category'] = umpire_data['consistency_category']
                games_with_umpires.loc[idx, 'umpire_favor_category'] = umpire_data['favor_category']
        
        # Save to file if requested
        if output_file:
            games_with_umpires.to_csv(output_file, index=False)
            print(f"Saved games with umpire data to {output_file}")
        
        return games_with_umpires
    
    def analyze_umpire_matchups(self, pitcher_id, umpire_id=None, umpire_name=None):
        """
        Analyze how a specific pitcher might perform with different umpires
        
        Args:
            pitcher_id (int/str): Pitcher ID
            umpire_id (int, optional): Specific umpire ID to analyze
            umpire_name (str, optional): Specific umpire name to analyze
            
        Returns:
            dict: Analysis of pitcher-umpire matchup
        """
        # This would typically use historical pitcher performance data
        # For demonstration, we'll simulate a pitcher's traits
        
        # Simulate pitcher traits (would be looked up from your pitcher database)
        np.random.seed(hash(str(pitcher_id)) % 10000)
        
        pitcher_traits = {
            'pitcher_id': pitcher_id,
            'control': np.random.normal(50, 15),  # 0-100 scale
            'strikeout_rate': np.random.normal(22, 5),  # Percentage
            'walk_rate': np.random.normal(8, 2),  # Percentage
            'pitch_to_edges': np.random.normal(25, 10),  # Percentage of pitches at zone edges
            'pitch_to_corners': np.random.normal(10, 5)  # Percentage of pitches to corners
        }
        
        # Normalize traits to 0-100 scales
        for trait in ['control', 'pitch_to_edges', 'pitch_to_corners']:
            pitcher_traits[trait] = max(0, min(100, pitcher_traits[trait]))
        
        # If specific umpire requested, analyze just that matchup
        if umpire_id is not None or umpire_name is not None:
            umpire_data = None
            if umpire_id is not None:
                umpire_data = self.get_umpire_by_id(umpire_id)
            elif umpire_name is not None:
                umpire_data = self.get_umpire_by_name(umpire_name)
            
            if umpire_data is not None:
                return self._analyze_single_matchup(pitcher_traits, umpire_data)
            else:
                return {"error": "Umpire not found"}
        
        # Analyze matchups with all umpires
        matchups = []
        for _, umpire in self.umpire_db.iterrows():
            matchup = self._analyze_single_matchup(pitcher_traits, umpire)
            matchups.append(matchup)
        
        # Sort by expected impact
        matchups.sort(key=lambda x: x['expected_strikeout_impact'], reverse=True)
        
        return {
            'pitcher_id': pitcher_id,
            'best_umpire_matchups': matchups[:3],
            'worst_umpire_matchups': matchups[-3:],
            'all_matchups': matchups
        }
    
    def _analyze_single_matchup(self, pitcher_traits, umpire_data):
        """
        Analyze a single pitcher-umpire matchup
        
        Args:
            pitcher_traits (dict): Pitcher traits
            umpire_data (Series): Umpire data
            
        Returns:
            dict: Matchup analysis
        """
        # Calculate compatibility factors
        compatibility = 0
        
        # Pitchers with good control benefit from consistent umpires
        control_consistency = (pitcher_traits['control'] / 100) * ((umpire_data['consistency'] - 80) / 20)
        
        # Pitchers who pitch to edges benefit from wider zones
        edges_zone_size = (pitcher_traits['pitch_to_edges'] / 100) * ((umpire_data['zone_size'] - 100) / 10)
        
        # All pitchers benefit from pitcher-friendly umpires
        pitcher_favor = umpire_data['favor_factor'] / 3  # Normalize to -1 to +1 range
        
        # Calculate overall compatibility
        compatibility = control_consistency + edges_zone_size + pitcher_favor
        
        # Calculate expected impacts
        # Base expected impacts are the umpire's general tendencies
        expected_strikeout_impact = umpire_data['strikeout_impact']
        expected_runs_impact = umpire_data['runs_impact']
        
        # Adjust for specific pitcher traits
        strikeout_adjustment = compatibility * 0.5  # Half a strikeout per game at max compatibility
        runs_adjustment = -compatibility * 0.3  # Up to 0.3 runs per game at max compatibility
        
        expected_strikeout_impact += strikeout_adjustment
        expected_runs_impact += runs_adjustment
        
        return {
            'umpire_id': umpire_data['umpire_id'],
            'umpire_name': umpire_data['name'],
            'compatibility_score': round(compatibility * 10, 1),  # Scale to -10 to +10
            'expected_strikeout_impact': round(expected_strikeout_impact, 2),
            'expected_runs_impact': round(expected_runs_impact, 2),
            'notes': self._generate_matchup_notes(pitcher_traits, umpire_data, compatibility)
        }
    
    def _generate_matchup_notes(self, pitcher_traits, umpire_data, compatibility):
        """
        Generate notes describing the pitcher-umpire matchup
        
        Args:
            pitcher_traits (dict): Pitcher traits
            umpire_data (Series): Umpire data
            compatibility (float): Calculated compatibility score
            
        Returns:
            str: Description of the matchup
        """
        notes = []
        
        # Zone size notes
        if pitcher_traits['pitch_to_edges'] > 60 and umpire_data['zone_size'] > 103:
            notes.append("Pitcher works the edges and umpire has a wide zone - favorable matchup.")
        elif pitcher_traits['pitch_to_edges'] > 60 and umpire_data['zone_size'] < 95:
            notes.append("Pitcher works the edges but umpire has a tight zone - challenging matchup.")
        
        # Consistency notes
        if pitcher_traits['control'] > 70 and umpire_data['consistency'] > 90:
            notes.append("Pitcher's good control pairs well with umpire's consistent zone.")
        elif pitcher_traits['control'] < 40 and umpire_data['consistency'] < 80:
            notes.append("Pitcher's control issues may be exacerbated by umpire's inconsistent zone.")
        
        # Favor factor notes
        if umpire_data['favor_factor'] > 1.5:
            notes.append("Umpire tends to favor pitchers, which could help with strike calls.")
        elif umpire_data['favor_factor'] < -1.5:
            notes.append("Umpire tends to favor hitters, which could lead to tighter strike zone.")
        
        # Overall compatibility notes
        if compatibility > 0.5:
            notes.append("Overall favorable matchup for the pitcher.")
        elif compatibility < -0.5:
            notes.append("Overall challenging matchup for the pitcher.")
        else:
            notes.append("Neutral matchup without significant advantages or disadvantages.")
        
        return " ".join(notes)

def test_umpire_analyzer():
    """
    Test the umpire analyzer functionality
    """
    # Initialize the umpire analyzer
    umpire_analyzer = UmpireAnalyzer()
    
    # Create sample games data
    games_data = {
        'game_id': range(1, 11),
        'game_date': [f'2024-04-{i:02d}' for i in range(1, 11)],
        'home_team': ['NYY', 'BOS', 'CHC', 'LAD', 'SF', 'COL', 'ATL', 'HOU', 'MIA', 'TB'],
        'away_team': ['BOS', 'NYY', 'MIL', 'SD', 'LAA', 'ARI', 'WSH', 'TEX', 'PHI', 'BAL'],
        'home_starting_pitcher': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'away_starting_pitcher': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
    }
    
    games_df = pd.DataFrame(games_data)
    
    # Integrate umpire data with games
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_umpires.csv')
    
    games_with_umpires = umpire_analyzer.integrate_umpire_data_with_games(games_df, output_file)
    
    # Test analyzing a pitcher-umpire matchup
    print("\nAnalyzing pitcher-umpire matchups for a sample pitcher:")
    pitcher_analysis = umpire_analyzer.analyze_umpire_matchups(101)
    
    print("\nBest umpire matchups for pitcher 101:")
    for matchup in pitcher_analysis['best_umpire_matchups']:
        print(f"Umpire: {matchup['umpire_name']}")
        print(f"Compatibility Score: {matchup['compatibility_score']}")
        print(f"Expected Strikeout Impact: {matchup['expected_strikeout_impact']}")
        print(f"Notes: {matchup['notes']}")
        print()
    
    # Merge everything to create a comprehensive dataset
    try:
        # Load weather data if it exists
        weather_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_weather.csv')
        if os.path.exists(weather_file):
            games_with_weather = pd.read_csv(weather_file)
            
            # Merge weather and umpire data
            all_contextual_data = pd.merge(
                games_with_weather, 
                games_with_umpires[['game_id'] + [col for col in games_with_umpires.columns if col.startswith('umpire_')]],
                on='game_id',
                how='outer'
            )
            
            # Save comprehensive dataset
            comprehensive_file = os.path.join(base_dir, 'data', 'contextual', 'games_with_all_context.csv')
            all_contextual_data.to_csv(comprehensive_file, index=False)
            print(f"\nCreated comprehensive contextual dataset: {comprehensive_file}")
            print("\nSample of comprehensive data:")
            print(all_contextual_data.head(3))
    except Exception as e:
        print(f"Error creating comprehensive dataset: {e}")

if __name__ == "__main__":
    test_umpire_analyzer()
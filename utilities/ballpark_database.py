import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

def create_ballpark_database(output_dir=None):
    """
    Create a comprehensive database of MLB ballpark information
    
    Args:
        output_dir (str, optional): Directory to save the ballpark database
        
    Returns:
        DataFrame: Compiled ballpark data
    """
    if output_dir is None:
        # Use default directory in project structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'contextual')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating MLB ballpark database...")
    
    # Dictionary of current MLB ballparks with their team codes
    ballparks = {
        'ARI': 'Chase Field',
        'ATL': 'Truist Park',
        'BAL': 'Oriole Park at Camden Yards',
        'BOS': 'Fenway Park',
        'CHC': 'Wrigley Field',
        'CWS': 'Guaranteed Rate Field',
        'CIN': 'Great American Ball Park',
        'CLE': 'Progressive Field',
        'COL': 'Coors Field',
        'DET': 'Comerica Park',
        'HOU': 'Minute Maid Park',
        'KC': 'Kauffman Stadium',
        'LAA': 'Angel Stadium',
        'LAD': 'Dodger Stadium',
        'MIA': 'LoanDepot Park',
        'MIL': 'American Family Field',
        'MIN': 'Target Field',
        'NYM': 'Citi Field',
        'NYY': 'Yankee Stadium',
        'OAK': 'Oakland Coliseum',
        'PHI': 'Citizens Bank Park',
        'PIT': 'PNC Park',
        'SD': 'Petco Park',
        'SEA': 'T-Mobile Park',
        'SF': 'Oracle Park',
        'STL': 'Busch Stadium',
        'TB': 'Tropicana Field',
        'TEX': 'Globe Life Field',
        'TOR': 'Rogers Centre',
        'WSH': 'Nationals Park'
    }
    
    # Initialize a list to store ballpark data
    ballpark_data = []
    
    # Base characteristics for all ballparks
    for team_code, ballpark_name in ballparks.items():
        # Add basic ballpark info
        park_info = {
            'team_code': team_code,
            'ballpark_name': ballpark_name,
            'year': 2024  # Current year
        }
        
        # Add to ballpark data list
        ballpark_data.append(park_info)
    
    # Convert to DataFrame
    ballparks_df = pd.DataFrame(ballpark_data)
    
    # Add ballpark characteristics
    # Based on research and known values, for a production version
    # you would want to scrape this from reliable sources
    
    # Dictionary of ballpark environment types
    park_environments = {
        'ARI': 'retractable',
        'ATL': 'outdoor',
        'BAL': 'outdoor',
        'BOS': 'outdoor',
        'CHC': 'outdoor',
        'CWS': 'outdoor',
        'CIN': 'outdoor',
        'CLE': 'outdoor',
        'COL': 'outdoor',
        'DET': 'outdoor',
        'HOU': 'retractable',
        'KC': 'outdoor',
        'LAA': 'outdoor',
        'LAD': 'outdoor',
        'MIA': 'retractable',
        'MIL': 'retractable',
        'MIN': 'outdoor',
        'NYM': 'outdoor',
        'NYY': 'outdoor',
        'OAK': 'outdoor',
        'PHI': 'outdoor',
        'PIT': 'outdoor',
        'SD': 'outdoor',
        'SEA': 'retractable',
        'SF': 'outdoor',
        'STL': 'outdoor',
        'TB': 'dome',
        'TEX': 'retractable',
        'TOR': 'retractable',
        'WSH': 'outdoor'
    }
    
    # Dictionary of ballpark elevations (feet above sea level)
    park_elevations = {
        'ARI': 1086,
        'ATL': 1050,
        'BAL': 36,
        'BOS': 20,
        'CHC': 597,
        'CWS': 594,
        'CIN': 490,
        'CLE': 653,
        'COL': 5280,  # Highest elevation - Coors Field
        'DET': 580,
        'HOU': 43,
        'KC': 886,
        'LAA': 152,
        'LAD': 510,
        'MIA': 7,
        'MIL': 602,
        'MIN': 840,
        'NYM': 9,
        'NYY': 14,
        'OAK': 42,
        'PHI': 39,
        'PIT': 724,
        'SD': 16,
        'SEA': 17,
        'SF': 45,
        'STL': 466,
        'TB': 42,
        'TEX': 545,
        'TOR': 266,
        'WSH': 25
    }
    
    # Add environment and elevation to DataFrame
    ballparks_df['environment_type'] = ballparks_df['team_code'].map(park_environments)
    ballparks_df['elevation_feet'] = ballparks_df['team_code'].map(park_elevations)
    
    # Dictionary of ballpark dimensions (approximate, would need verification)
    # Format: (left field, left-center, center field, right-center, right field)
    park_dimensions = {
        'ARI': {'left': 330, 'left_center': 374, 'center': 407, 'right_center': 374, 'right': 334},
        'ATL': {'left': 335, 'left_center': 385, 'center': 400, 'right_center': 385, 'right': 325},
        'BAL': {'left': 333, 'left_center': 364, 'center': 410, 'right_center': 373, 'right': 318},
        'BOS': {'left': 310, 'left_center': 379, 'center': 390, 'right_center': 380, 'right': 302},
        'CHC': {'left': 355, 'left_center': 368, 'center': 400, 'right_center': 368, 'right': 353},
        'CWS': {'left': 330, 'left_center': 375, 'center': 400, 'right_center': 375, 'right': 335},
        'CIN': {'left': 328, 'left_center': 379, 'center': 404, 'right_center': 370, 'right': 325},
        'CLE': {'left': 325, 'left_center': 370, 'center': 405, 'right_center': 375, 'right': 325},
        'COL': {'left': 347, 'left_center': 390, 'center': 415, 'right_center': 375, 'right': 350},
        'DET': {'left': 345, 'left_center': 370, 'center': 420, 'right_center': 365, 'right': 330},
        'HOU': {'left': 315, 'left_center': 362, 'center': 409, 'right_center': 373, 'right': 326},
        'KC': {'left': 330, 'left_center': 387, 'center': 410, 'right_center': 387, 'right': 330},
        'LAA': {'left': 330, 'left_center': 387, 'center': 408, 'right_center': 370, 'right': 330},
        'LAD': {'left': 330, 'left_center': 375, 'center': 400, 'right_center': 375, 'right': 330},
        'MIA': {'left': 344, 'left_center': 386, 'center': 407, 'right_center': 392, 'right': 335},
        'MIL': {'left': 342, 'left_center': 371, 'center': 400, 'right_center': 374, 'right': 345},
        'MIN': {'left': 339, 'left_center': 377, 'center': 407, 'right_center': 367, 'right': 328},
        'NYM': {'left': 335, 'left_center': 370, 'center': 408, 'right_center': 390, 'right': 330},
        'NYY': {'left': 318, 'left_center': 399, 'center': 408, 'right_center': 385, 'right': 314},
        'OAK': {'left': 330, 'left_center': 371, 'center': 400, 'right_center': 371, 'right': 330},
        'PHI': {'left': 329, 'left_center': 374, 'center': 401, 'right_center': 369, 'right': 330},
        'PIT': {'left': 325, 'left_center': 389, 'center': 399, 'right_center': 375, 'right': 320},
        'SD': {'left': 336, 'left_center': 390, 'center': 396, 'right_center': 390, 'right': 322},
        'SEA': {'left': 331, 'left_center': 378, 'center': 401, 'right_center': 381, 'right': 326},
        'SF': {'left': 339, 'left_center': 399, 'center': 404, 'right_center': 399, 'right': 309},
        'STL': {'left': 336, 'left_center': 375, 'center': 400, 'right_center': 375, 'right': 335},
        'TB': {'left': 315, 'left_center': 370, 'center': 404, 'right_center': 370, 'right': 322},
        'TEX': {'left': 329, 'left_center': 372, 'center': 407, 'right_center': 374, 'right': 326},
        'TOR': {'left': 328, 'left_center': 375, 'center': 400, 'right_center': 375, 'right': 328},
        'WSH': {'left': 336, 'left_center': 377, 'center': 402, 'right_center': 370, 'right': 335}
    }
    
    # Add dimensions to DataFrame
    for dimension in ['left', 'left_center', 'center', 'right_center', 'right']:
        ballparks_df[f'{dimension}_field_ft'] = ballparks_df['team_code'].apply(
            lambda x: park_dimensions[x][dimension] if x in park_dimensions else None
        )
    
    # The next step is to fetch park factors
    # Park factors measure how much a ballpark increases or decreases 
    # specific offensive stats compared to league average
    
    # Let's define a simple function to scrape park factors
    # This is a simplified version - in production you might want to scrape from 
    # Baseball Reference, FanGraphs, or other reliable sources
    
    # Placeholder for park factors - in a real implementation, you would scrape these
    # These are approximate values and would need to be updated with current data
    basic_park_factors = {
        'ARI': {'runs': 103, 'hr': 115, 'hits': 102},
        'ATL': {'runs': 101, 'hr': 108, 'hits': 99},
        'BAL': {'runs': 98, 'hr': 120, 'hits': 95},
        'BOS': {'runs': 107, 'hr': 95, 'hits': 110},
        'CHC': {'runs': 102, 'hr': 100, 'hits': 101},
        'CWS': {'runs': 105, 'hr': 116, 'hits': 102},
        'CIN': {'runs': 104, 'hr': 118, 'hits': 98},
        'CLE': {'runs': 96, 'hr': 94, 'hits': 97},
        'COL': {'runs': 115, 'hr': 117, 'hits': 113},
        'DET': {'runs': 98, 'hr': 92, 'hits': 100},
        'HOU': {'runs': 101, 'hr': 108, 'hits': 99},
        'KC': {'runs': 99, 'hr': 89, 'hits': 104},
        'LAA': {'runs': 98, 'hr': 93, 'hits': 100},
        'LAD': {'runs': 97, 'hr': 101, 'hits': 96},
        'MIA': {'runs': 94, 'hr': 89, 'hits': 96},
        'MIL': {'runs': 102, 'hr': 110, 'hits': 99},
        'MIN': {'runs': 99, 'hr': 103, 'hits': 98},
        'NYM': {'runs': 96, 'hr': 91, 'hits': 97},
        'NYY': {'runs': 104, 'hr': 118, 'hits': 98},
        'OAK': {'runs': 95, 'hr': 88, 'hits': 97},
        'PHI': {'runs': 103, 'hr': 113, 'hits': 100},
        'PIT': {'runs': 96, 'hr': 87, 'hits': 100},
        'SD': {'runs': 93, 'hr': 89, 'hits': 95},
        'SEA': {'runs': 94, 'hr': 93, 'hits': 95},
        'SF': {'runs': 92, 'hr': 86, 'hits': 96},
        'STL': {'runs': 100, 'hr': 96, 'hits': 102},
        'TB': {'runs': 96, 'hr': 94, 'hits': 97},
        'TEX': {'runs': 102, 'hr': 99, 'hits': 103},
        'TOR': {'runs': 103, 'hr': 108, 'hits': 101},
        'WSH': {'runs': 99, 'hr': 97, 'hits': 100}
    }
    
    # Add park factors to DataFrame
    for factor in ['runs', 'hr', 'hits']:
        ballparks_df[f'{factor}_factor'] = ballparks_df['team_code'].apply(
            lambda x: basic_park_factors[x][factor] if x in basic_park_factors else None
        )
    
    # Calculate a composite "hitter friendliness" score
    # This is a simple weighted average of the park factors
    ballparks_df['hitter_friendly_score'] = (
        ballparks_df['runs_factor'] * 0.4 + 
        ballparks_df['hr_factor'] * 0.4 + 
        ballparks_df['hits_factor'] * 0.2
    )
    
    # Add a categorical version of hitter friendliness
    ballparks_df['hitter_friendly_category'] = pd.cut(
        ballparks_df['hitter_friendly_score'],
        bins=[0, 95, 102, 150],
        labels=['pitcher_friendly', 'neutral', 'hitter_friendly']
    )

    # Add ballpark surface type
    ballparks_df['surface_type'] = 'grass'  # Most ballparks have grass
    # Exceptions
    ballparks_df.loc[ballparks_df['team_code'] == 'TB', 'surface_type'] = 'artificial'
    
    # Save the ballpark database to CSV
    output_file = os.path.join(output_dir, 'ballpark_database.csv')
    ballparks_df.to_csv(output_file, index=False)
    print(f"Ballpark database saved to {output_file}")
    
    # Print some summary statistics
    print("\nBallpark Database Summary:")
    print(f"Total ballparks: {len(ballparks_df)}")
    print("\nMost hitter-friendly ballparks:")
    print(ballparks_df.sort_values('hitter_friendly_score', ascending=False)[['team_code', 'ballpark_name', 'hitter_friendly_score']].head(5))
    
    print("\nMost pitcher-friendly ballparks:")
    print(ballparks_df.sort_values('hitter_friendly_score')[['team_code', 'ballpark_name', 'hitter_friendly_score']].head(5))
    
    return ballparks_df

def test_ballpark_database():
    """
    Test the ballpark database creation
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'contextual')
    
    # Create the ballpark database
    ballparks_df = create_ballpark_database(output_dir)
    
    # Print sample of the data
    print("\nSample of ballpark database:")
    sample_columns = ['team_code', 'ballpark_name', 'environment_type', 
                       'elevation_feet', 'center_field_ft', 'runs_factor', 
                       'hr_factor', 'hitter_friendly_score']
    print(ballparks_df[sample_columns].head())

if __name__ == "__main__":
    test_ballpark_database()
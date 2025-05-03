import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_statcast_data(statcast_df, player_type='batter', time_period=None, verbose=True):
    """
    Process raw Statcast data into aggregated player metrics with comprehensive 
    advanced statistics and time-based filtering
    
    Args:
        statcast_df (DataFrame): Raw Statcast data from Baseball Savant
        player_type (str): Either 'batter' or 'pitcher'
        time_period (str, optional): Time period to filter data ('last_7', 'last_15', 'last_30', or None for all data)
        verbose (bool): Whether to print processing information
        
    Returns:
        DataFrame with processed player-level Statcast metrics
    """
    if statcast_df.empty:
        if verbose:
            print("No Statcast data to process")
        return pd.DataFrame()
    
    if verbose:
        print(f"Processing {len(statcast_df)} Statcast records for {player_type}s")
    
    # Clean data: handle missing values in key columns
    numeric_cols = [
        'launch_speed', 'launch_angle', 'release_speed', 'release_spin_rate', 
        'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
        'effective_speed', 'release_extension', 'spin_axis', 'zone', 'plate_x', 'plate_z'
    ]
    
    for col in numeric_cols:
        if col in statcast_df.columns:
            statcast_df[col] = pd.to_numeric(statcast_df[col], errors='coerce')
    
    # Handle missing player names
    if 'player_name' in statcast_df.columns:
        statcast_df['player_name'] = statcast_df['player_name'].fillna('Unknown Player')
    
    # Convert game_date to datetime
    if 'game_date' in statcast_df.columns:
        statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
        
        # Apply time-based filtering if specified
        if time_period:
            latest_date = statcast_df['game_date'].max()
            
            if time_period == 'last_7':
                cutoff_date = latest_date - timedelta(days=7)
            elif time_period == 'last_15':
                cutoff_date = latest_date - timedelta(days=15)
            elif time_period == 'last_30':
                cutoff_date = latest_date - timedelta(days=30)
            else:
                cutoff_date = latest_date - timedelta(days=365)  # Default to last year
                
            statcast_df = statcast_df[statcast_df['game_date'] >= cutoff_date]
            if verbose:
                print(f"Filtered to {time_period} days: {len(statcast_df)} records")
    
    # Filter to specific events for batted ball calculations
    batted_ball_events = ['single', 'double', 'triple', 'home_run', 'field_out', 
                          'force_out', 'grounded_into_double_play', 'fielders_choice_out',
                          'fielders_choice', 'sac_fly', 'sac_bunt', 'double_play']
    
    batted_balls = statcast_df[statcast_df['events'].isin(batted_ball_events)].copy()
    
    # Process differently depending on player type
    if player_type == 'batter':
        # Group by batter ID
        grouped = statcast_df.groupby('batter')
        
        # Create empty list for results
        results = []
        
        for player_id, group in grouped:
            # Skip if no data
            if len(group) == 0:
                continue
            
            # Basic player info
            player_name = group['player_name'].iloc[0] if 'player_name' in group.columns else f"Player {player_id}"
            
            # Get batted ball data for this player
            player_batted = batted_balls[batted_balls['batter'] == player_id]
            
            # Calculate metrics - handle case where player has no batted ball events
            metric_dict = {
                'player_id': player_id,
                'player_name': player_name,
                'total_pitches': len(group),
                'total_pas': len(group['at_bat_number'].unique())
            }
            
            # Add stadiums and count of games per stadium if available
            if 'home_team' in group.columns:
                stadiums = group['home_team'].value_counts().to_dict()
                for stadium, count in stadiums.items():
                    metric_dict[f'games_at_{stadium.lower()}'] = count
            
            # Calculate platoon splits if pitcher_throws is available
            if 'p_throws' in group.columns:
                vs_left = group[group['p_throws'] == 'L']
                vs_right = group[group['p_throws'] == 'R']
                
                metric_dict['pitches_vs_left'] = len(vs_left)
                metric_dict['pitches_vs_right'] = len(vs_right)
                
                # Calculate exit velocity splits
                if len(batted_balls[batted_balls['batter'] == player_id]) > 0:
                    vs_left_batted = player_batted[player_batted['p_throws'] == 'L']
                    vs_right_batted = player_batted[player_batted['p_throws'] == 'R']
                    
                    if len(vs_left_batted) > 0:
                        metric_dict['exit_velo_vs_left'] = vs_left_batted['launch_speed'].mean()
                    
                    if len(vs_right_batted) > 0:
                        metric_dict['exit_velo_vs_right'] = vs_right_batted['launch_speed'].mean()
            
            # Pitch recognition and plate discipline metrics
            swings = pd.Series([False] * len(group))
            if 'description' in group.columns:
                swings = group['description'].isin(['hit_into_play', 'swinging_strike', 
                                                'foul', 'foul_tip', 'swinging_strike_blocked'])
                
                metric_dict['swing_count'] = swings.sum()
                metric_dict['swing_rate'] = swings.mean() if len(group) > 0 else 0
                
                # Whiff metrics (swing and miss)
                whiffs = group['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
                metric_dict['whiff_count'] = whiffs.sum()
                metric_dict['whiff_rate'] = whiffs.sum() / swings.sum() if swings.sum() > 0 else 0
                
                # Chase rate (swing at pitches outside zone)
                if 'zone' in group.columns:
                    outside_zone = group['zone'] > 9  # Zones 11-14 are outside
                    outside_swings = outside_zone & swings
                    
                    metric_dict['chase_count'] = outside_swings.sum()
                    metric_dict['chase_rate'] = outside_swings.sum() / outside_zone.sum() if outside_zone.sum() > 0 else 0
                    
                    # Zone contact (contact on pitches in zone)
                    in_zone = group['zone'].between(1, 9)  # Zones 1-9 are in the strike zone
                    zone_swings = in_zone & swings
                    zone_contact = in_zone & group['description'].isin(['hit_into_play', 'foul', 'foul_tip'])
                    
                    metric_dict['zone_swing_rate'] = zone_swings.sum() / in_zone.sum() if in_zone.sum() > 0 else 0
                    metric_dict['zone_contact_rate'] = zone_contact.sum() / zone_swings.sum() if zone_swings.sum() > 0 else 0
            
            # Batted ball metrics (only if player has batted ball events)
            if len(player_batted) > 0:
                metric_dict['total_batted_balls'] = len(player_batted)
                
                # Basic exit velocity and launch angle
                exit_velo = player_batted['launch_speed'].dropna()
                launch_angle = player_batted['launch_angle'].dropna()
                
                if len(exit_velo) > 0:
                    metric_dict['exit_velocity_avg'] = exit_velo.mean()
                    metric_dict['exit_velocity_max'] = exit_velo.max()
                    metric_dict['exit_velocity_median'] = exit_velo.median()
                    metric_dict['exit_velocity_std'] = exit_velo.std()  # Consistency measure
                
                if len(launch_angle) > 0:
                    metric_dict['launch_angle_avg'] = launch_angle.mean()
                    metric_dict['launch_angle_median'] = launch_angle.median()
                    metric_dict['launch_angle_std'] = launch_angle.std()
                
                # Hard hit metrics (95+ mph exit velocity)
                hard_hits = player_batted['launch_speed'] >= 95
                metric_dict['hard_hit_count'] = hard_hits.sum()
                metric_dict['hard_hit_percent'] = hard_hits.mean() * 100
                
                # Barrel metrics
                if 'barrel' in player_batted.columns:
                    barrels = player_batted['barrel'] == 1
                    metric_dict['barrel_count'] = barrels.sum()
                    metric_dict['barrel_percent'] = barrels.mean() * 100
                
                # Sweet spot metrics (launch angle between 8-32 degrees)
                sweet_spots = player_batted['launch_angle'].between(8, 32)
                metric_dict['sweet_spot_count'] = sweet_spots.sum()
                metric_dict['sweet_spot_percent'] = sweet_spots.mean() * 100
                
                # Expected statistics
                if 'estimated_ba_using_speedangle' in player_batted.columns:
                    xba = player_batted['estimated_ba_using_speedangle'].dropna()
                    if len(xba) > 0:
                        metric_dict['xba_avg'] = xba.mean()
                
                if 'estimated_woba_using_speedangle' in player_batted.columns:
                    xwoba = player_batted['estimated_woba_using_speedangle'].dropna()
                    if len(xwoba) > 0:
                        metric_dict['xwoba_avg'] = xwoba.mean()
                
                # Batted ball directions
                if 'hit_location' in player_batted.columns:
                    pull_field = player_batted['hit_location'].between(1, 3)
                    center_field = player_batted['hit_location'].between(4, 6)
                    oppo_field = player_batted['hit_location'].between(7, 9)
                    
                    metric_dict['pull_percent'] = pull_field.mean() * 100
                    metric_dict['center_percent'] = center_field.mean() * 100
                    metric_dict['oppo_percent'] = oppo_field.mean() * 100
                
                # Batted ball types
                if 'bb_type' in player_batted.columns:
                    gb = player_batted['bb_type'] == 'ground_ball'
                    fb = player_batted['bb_type'] == 'fly_ball'
                    ld = player_batted['bb_type'] == 'line_drive'
                    pu = player_batted['bb_type'] == 'popup'
                    
                    metric_dict['ground_ball_percent'] = gb.mean() * 100
                    metric_dict['fly_ball_percent'] = fb.mean() * 100
                    metric_dict['line_drive_percent'] = ld.mean() * 100
                    metric_dict['popup_percent'] = pu.mean() * 100
                    
                    # Fly ball and line drive exit velocity (good for power prediction)
                    fb_ld = player_batted['bb_type'].isin(['fly_ball', 'line_drive'])
                    fb_ld_velo = player_batted.loc[fb_ld, 'launch_speed'].dropna()
                    
                    if len(fb_ld_velo) > 0:
                        metric_dict['fb_ld_exit_velo'] = fb_ld_velo.mean()
            
            results.append(metric_dict)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Fill NaN values
        if not results_df.empty:
            # For numeric columns, fill with 0 or median based on column type
            for col in results_df.columns:
                if col not in ['player_id', 'player_name'] and pd.api.types.is_numeric_dtype(results_df[col]):
                    # If it's a count or total, fill with 0
                    if 'count' in col or 'total' in col or col.startswith('games_at_'):
                        results_df[col] = results_df[col].fillna(0)
                    else:
                        # For rate/percentage columns, fill with median
                        results_df[col] = results_df[col].fillna(results_df[col].median())
        
        return results_df
            
    elif player_type == 'pitcher':
        # Group by pitcher ID
        grouped = statcast_df.groupby('pitcher')
        
        # Create empty list for results
        results = []
        
        for player_id, group in grouped:
            # Skip if no data
            if len(group) == 0:
                continue
            
            # Basic player info
            player_name = group['player_name'].iloc[0] if 'player_name' in group.columns else f"Pitcher {player_id}"
            
            # Get batted ball data for this player
            player_batted = batted_balls[batted_balls['pitcher'] == player_id]
            
            # Calculate metrics
            metric_dict = {
                'player_id': player_id,
                'player_name': player_name,
                'total_pitches': len(group),
                'total_batters_faced': len(group['at_bat_number'].unique())
            }
            
            # Add stadiums and count of games per stadium if available
            if 'home_team' in group.columns:
                stadiums = group['home_team'].value_counts().to_dict()
                for stadium, count in stadiums.items():
                    metric_dict[f'games_at_{stadium.lower()}'] = count
            
            # Calculate platoon splits if batter handedness is available
            if 'stand' in group.columns:
                vs_left = group[group['stand'] == 'L']
                vs_right = group[group['stand'] == 'R']
                
                metric_dict['pitches_vs_left'] = len(vs_left)
                metric_dict['pitches_vs_right'] = len(vs_right)
            
            # Pitch velocity and movement metrics
            if 'release_speed' in group.columns:
                velo = group['release_speed'].dropna()
                if len(velo) > 0:
                    metric_dict['velocity_avg'] = velo.mean()
                    metric_dict['velocity_max'] = velo.max()
                    metric_dict['velocity_std'] = velo.std()  # Consistency measure
            
            if 'release_spin_rate' in group.columns:
                spin = group['release_spin_rate'].dropna()
                if len(spin) > 0:
                    metric_dict['spin_rate_avg'] = spin.mean()
            
            # Pitch type breakdown
            if 'pitch_type' in group.columns:
                pitch_counts = group['pitch_type'].value_counts()
                for pitch, count in pitch_counts.items():
                    if pd.notna(pitch) and pitch != '':
                        metric_dict[f'pitch_{pitch}_count'] = count
                        metric_dict[f'pitch_{pitch}_pct'] = count / len(group) * 100
                        
                        # Calculate average velocity and spin by pitch type
                        pitch_data = group[group['pitch_type'] == pitch]
                        
                        if 'release_speed' in pitch_data:
                            pitch_velo = pitch_data['release_speed'].dropna()
                            if len(pitch_velo) > 0:
                                metric_dict[f'pitch_{pitch}_velo'] = pitch_velo.mean()
                        
                        if 'release_spin_rate' in pitch_data:
                            pitch_spin = pitch_data['release_spin_rate'].dropna()
                            if len(pitch_spin) > 0:
                                metric_dict[f'pitch_{pitch}_spin'] = pitch_spin.mean()
            
            # Plate discipline metrics
            if 'description' in group.columns:
                # Strikes and balls
                strikes = group['description'].isin(['called_strike', 'swinging_strike', 
                                                  'swinging_strike_blocked', 'foul', 'foul_tip'])
                balls = group['description'].isin(['ball', 'blocked_ball', 'intent_ball', 'pitchout'])
                
                metric_dict['strike_count'] = strikes.sum()
                metric_dict['ball_count'] = balls.sum()
                metric_dict['strike_pct'] = strikes.sum() / len(group) if len(group) > 0 else 0
                
                # Swing metrics
                swings = group['description'].isin(['swinging_strike', 'swinging_strike_blocked',
                                                'foul', 'foul_tip', 'hit_into_play'])
                whiffs = group['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
                
                metric_dict['swing_count'] = swings.sum()
                metric_dict['whiff_count'] = whiffs.sum()
                metric_dict['whiff_rate'] = whiffs.sum() / swings.sum() if swings.sum() > 0 else 0
                
                # Zone metrics
                if 'zone' in group.columns:
                    in_zone = group['zone'].between(1, 9)  # Zones 1-9 are in strike zone
                    out_zone = group['zone'] > 9  # Zones 11-14 are outside
                    
                    metric_dict['zone_rate'] = in_zone.mean() * 100
                    
                    # Chase and contact metrics
                    out_zone_swings = out_zone & swings
                    in_zone_contact = in_zone & group['description'].isin(['hit_into_play', 'foul', 'foul_tip'])
                    
                    metric_dict['chase_count'] = out_zone_swings.sum()
                    metric_dict['chase_rate'] = out_zone_swings.sum() / out_zone.sum() if out_zone.sum() > 0 else 0
                    metric_dict['zone_contact_rate'] = in_zone_contact.sum() / in_zone.sum() if in_zone.sum() > 0 else 0
            
            # Batted ball metrics against
            if len(player_batted) > 0:
                metric_dict['total_batted_balls_against'] = len(player_batted)
                
                # Exit velocity and launch angle against
                exit_velo = player_batted['launch_speed'].dropna()
                launch_angle = player_batted['launch_angle'].dropna()
                
                if len(exit_velo) > 0:
                    metric_dict['exit_velocity_against_avg'] = exit_velo.mean()
                    metric_dict['exit_velocity_against_max'] = exit_velo.max()
                
                if len(launch_angle) > 0:
                    metric_dict['launch_angle_against_avg'] = launch_angle.mean()
                
                # Hard hit metrics against
                hard_hits = player_batted['launch_speed'] >= 95
                metric_dict['hard_hit_against_count'] = hard_hits.sum()
                metric_dict['hard_hit_against_percent'] = hard_hits.mean() * 100
                
                # Barrel metrics against
                if 'barrel' in player_batted.columns:
                    barrels = player_batted['barrel'] == 1
                    metric_dict['barrel_against_count'] = barrels.sum()
                    metric_dict['barrel_against_percent'] = barrels.mean() * 100
                
                # Expected statistics against
                if 'estimated_ba_using_speedangle' in player_batted.columns:
                    xba = player_batted['estimated_ba_using_speedangle'].dropna()
                    if len(xba) > 0:
                        metric_dict['xba_against_avg'] = xba.mean()
                
                if 'estimated_woba_using_speedangle' in player_batted.columns:
                    xwoba = player_batted['estimated_woba_using_speedangle'].dropna()
                    if len(xwoba) > 0:
                        metric_dict['xwoba_against_avg'] = xwoba.mean()
                
                # Batted ball types against
                if 'bb_type' in player_batted.columns:
                    gb = player_batted['bb_type'] == 'ground_ball'
                    fb = player_batted['bb_type'] == 'fly_ball'
                    ld = player_batted['bb_type'] == 'line_drive'
                    pu = player_batted['bb_type'] == 'popup'
                    
                    metric_dict['ground_ball_against_percent'] = gb.mean() * 100
                    metric_dict['fly_ball_against_percent'] = fb.mean() * 100
                    metric_dict['line_drive_against_percent'] = ld.mean() * 100
                    metric_dict['popup_against_percent'] = pu.mean() * 100
            
            results.append(metric_dict)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Fill NaN values
        if not results_df.empty:
            # For numeric columns, fill with 0 or median based on column type
            for col in results_df.columns:
                if col not in ['player_id', 'player_name'] and pd.api.types.is_numeric_dtype(results_df[col]):
                    # If it's a count or total, fill with 0
                    if 'count' in col or 'total' in col or col.startswith('games_at_'):
                        results_df[col] = results_df[col].fillna(0)
                    else:
                        # For rate/percentage columns, fill with median
                        results_df[col] = results_df[col].fillna(results_df[col].median())
        
        return results_df
    
    else:
        print(f"Invalid player_type: {player_type}. Must be 'batter' or 'pitcher'.")
        return pd.DataFrame()

def test_statcast_processor_with_time_periods():
    """
    Test the enhanced Statcast processor with time-based analysis
    """
    # Look for the most recent batter and pitcher data files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'raw', 'statcast')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find the most recent batter file
    batter_files = [f for f in os.listdir(data_dir) if f.startswith('statcast_batter_')]
    
    if batter_files:
        # Sort by creation time (newest first)
        batter_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
        newest_batter_file = os.path.join(data_dir, batter_files[0])
        
        print(f"Processing newest batter file: {newest_batter_file}")
        batter_data = pd.read_csv(newest_batter_file)
        
        # Process data with different time periods
        time_periods = [None, 'last_7', 'last_15', 'last_30']
        
        for period in time_periods:
            period_name = period if period else 'all_time'
            print(f"\nProcessing batter data for {period_name}:")
            
            processed_batters = process_statcast_data(batter_data, 'batter', time_period=period)
            
            if not processed_batters.empty:
                print(f"Successfully processed {len(processed_batters)} batters")
                
                # Sample of processed data
                print("\nSample metrics (first 3 batters):")
                columns_to_show = ['player_name', 'total_pitches', 'exit_velocity_avg', 
                                  'hard_hit_percent', 'whiff_rate', 'chase_rate']
                available_cols = [col for col in columns_to_show if col in processed_batters.columns]
                print(processed_batters[available_cols].head(3))
                
                # Save processed data
                processed_dir = os.path.join(os.path.dirname(data_dir), 'processed', 'statcast')
                os.makedirs(processed_dir, exist_ok=True)
                
                output_file = os.path.join(processed_dir, f"processed_batter_{period_name}.csv")
                processed_batters.to_csv(output_file, index=False)
                print(f"Saved processed batter data to {output_file}")
    else:
        print("No batter files found")

if __name__ == "__main__":
    test_statcast_processor_with_time_periods()
# config.py - Central configuration for MLB betting project
import os

# Base directory is where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "sports_data", "mlb")
GAMES_DIR = os.path.join(DATA_DIR, "games")
TEAMS_DIR = os.path.join(DATA_DIR, "teams")
PITCHERS_DIR = os.path.join(DATA_DIR, "pitchers")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
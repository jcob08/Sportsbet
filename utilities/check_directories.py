import pandas as pd
import os

# Set up directories
DATA_DIR = "sports_data/mlb"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Check directory structure
print(f"Checking if base directory exists: {os.path.exists(DATA_DIR)}")
print(f"Checking if processed directory exists: {os.path.exists(PROCESSED_DIR)}")

# If processed directory exists, list its contents
if os.path.exists(PROCESSED_DIR):
    print("\nFiles in processed directory:")
    for file in os.listdir(PROCESSED_DIR):
        print(f" - {file}")
else:
    print("\nProcessed directory not found.")
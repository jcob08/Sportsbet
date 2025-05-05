# test_imports.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import statsapi  # This should now be the MLB-StatsAPI package
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

print("All imports successful!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"StatsAPI available: {statsapi.__name__ if 'statsapi' in globals() else 'Not found'}")
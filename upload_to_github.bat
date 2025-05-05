@echo off
REM Initialize git if not already a repo
if not exist ".git" (
    git init
)

REM Add remote if not already set
git remote -v | findstr /C:"origin" >nul
if %errorlevel% neq 0 (
    git remote add origin https://github.com/jcob08/Sportsbet.git
)

REM Create .gitignore if it doesn't exist
if not exist ".gitignore" (
    echo # Ignore data and secrets > .gitignore
    echo *.csv >> .gitignore
    echo *.json >> .gitignore
    echo __pycache__/ >> .gitignore
    echo .env >> .gitignore
    echo .DS_Store >> .gitignore
)

REM Create README.md if it doesn't exist
if not exist "README.md" (
    echo # MLB Betting Pipeline > README.md
    echo. >> README.md
    echo This project contains a contextual feature engineering and betting simulation pipeline for MLB games, including odds integration and model evaluation. >> README.md
    echo. >> README.md
    echo ## Main Features >> README.md
    echo - Contextual feature engineering (ballpark, weather, umpire, player stats) >> README.md
    echo - Real-time odds fetching and merging >> README.md
    echo - Betting edge calculation and simulation >> README.md
    echo. >> README.md
    echo ## How to Run >> README.md
    echo 1. Install dependencies: `pip install -r requirements.txt` >> README.md
    echo 2. Run: `python utilities/contextual_feature_engineer.py` >> README.md
    echo. >> README.md
    echo ## Author >> README.md
    echo jcob08 >> README.md
)

REM Stage all files
git add .

REM Commit
git commit -m "Initial upload of MLB betting pipeline with contextual feature engineering and odds integration"

REM Set branch to master and push
git branch -M master
git push -u origin master

pause
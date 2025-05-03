import os
import json
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

# Create a Flask app
app = Flask(__name__)

# Function to load the model
def load_model():
    model_path = "models/nba_betting_model.pkl"
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Function to make predictions
def predict_game(model, home_team, away_team, home_win_pct, away_win_pct):
    # Calculate additional features
    win_pct_diff = home_win_pct - away_win_pct
    home_home_win_pct = min(1.0, home_win_pct * 1.1)
    away_away_win_pct = max(0.0, away_win_pct * 0.9)
    
    # Create feature array (we'll use dummy values for scores since they're unknown)
    # Adjust this to match your model's expected features
    features = {
        'home_score': 100,  # Dummy value
        'away_score': 100,  # Dummy value
        'home_win_pct': home_win_pct,
        'away_win_pct': away_win_pct,
        'win_pct_diff': win_pct_diff,
        'home_home_win_pct': home_home_win_pct,
        'away_away_win_pct': away_away_win_pct
    }
    
    # Convert to DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction
    win_prob = model.predict_proba(X)[0, 1]
    prediction = "Home Win" if win_prob > 0.5 else "Away Win"
    confidence = abs(win_prob - 0.5) * 2
    
    # Generate recommendation
    if win_prob > 0.67:
        recommendation = "Strong bet on Home"
    elif win_prob > 0.58:
        recommendation = "Consider bet on Home"
    elif win_prob < 0.33:
        recommendation = "Strong bet on Away"
    elif win_prob < 0.42:
        recommendation = "Consider bet on Away"
    else:
        recommendation = "No strong edge"
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'win_probability': float(win_prob),
        'prediction': prediction,
        'confidence': float(confidence),
        'recommendation': recommendation
    }

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')
    
    try:
        home_win_pct = float(request.form.get('home_win_pct'))
        away_win_pct = float(request.form.get('away_win_pct'))
    except ValueError:
        return jsonify({'error': 'Win percentages must be valid numbers between 0 and 1'})
    
    # Validate input
    if not home_team or not away_team:
        return jsonify({'error': 'Please provide both home and away teams'})
    
    if not 0 <= home_win_pct <= 1 or not 0 <= away_win_pct <= 1:
        return jsonify({'error': 'Win percentages must be between 0 and 1'})
    
    # Load model
    model = load_model()
    if not model:
        return jsonify({'error': 'Model not found. Please train the model first.'})
    
    # Make prediction
    result = predict_game(model, home_team, away_team, home_win_pct, away_win_pct)
    
    # Return prediction as JSON
    return jsonify(result)

# Route for viewing saved predictions
@app.route('/predictions')
def view_predictions():
    predictions_file = "predictions/upcoming_game_predictions.json"
    
    if not os.path.exists(predictions_file):
        return render_template('predictions.html', predictions=[])
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    return render_template('predictions.html', predictions=predictions)

# Run the app
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Check if template files exist, if not create them
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>NBA Betting Prediction System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>NBA Betting Prediction System</h1>
        
        <div class="form-container">
            <h2>Make a Prediction</h2>
            <form id="prediction-form">
                <div class="form-group">
                    <label for="home_team">Home Team:</label>
                    <input type="text" id="home_team" name="home_team" required>
                </div>
                
                <div class="form-group">
                    <label for="away_team">Away Team:</label>
                    <input type="text" id="away_team" name="away_team" required>
                </div>
                
                <div class="form-group">
                    <label for="home_win_pct">Home Team Win %:</label>
                    <input type="number" id="home_win_pct" name="home_win_pct" min="0" max="1" step="0.01" value="0.5" required>
                </div>
                
                <div class="form-group">
                    <label for="away_win_pct">Away Team Win %:</label>
                    <input type="number" id="away_win_pct" name="away_win_pct" min="0" max="1" step="0.01" value="0.5" required>
                </div>
                
                <button type="submit">Predict</button>
            </form>
        </div>
        
        <div id="result" class="result-container hidden">
            <h2>Prediction Result</h2>
            <div id="result-content"></div>
        </div>
        
        <div class="links">
            <a href="/predictions">View Saved Predictions</a>
        </div>
    </div>
    
    <script>
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            
            // Hide previous result
            document.getElementById('result').classList.add('hidden');
            
            // Send request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Show result
                const resultContent = document.getElementById('result-content');
                
                if (data.error) {
                    resultContent.innerHTML = `<div class="error">${data.error}</div>`;
                } else {
                    // Format the result
                    let resultHTML = `
                        <div class="game-info">
                            <span class="team">${data.home_team}</span> vs 
                            <span class="team">${data.away_team}</span>
                        </div>
                        <div class="prediction">
                            <span class="label">Prediction:</span> 
                            <span class="value">${data.prediction}</span>
                        </div>
                        <div class="probability">
                            <span class="label">Win Probability:</span> 
                            <span class="value">${(data.win_probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="confidence">
                            <span class="label">Confidence:</span> 
                            <span class="value">${(data.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="recommendation">
                            <span class="label">Recommendation:</span> 
                            <span class="value">${data.recommendation}</span>
                        </div>
                    `;
                    
                    resultContent.innerHTML = resultHTML;
                }
                
                // Show result container
                document.getElementById('result').classList.remove('hidden');
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result-content').innerHTML = `<div class="error">An error occurred. Please try again later.</div>`;
                document.getElementById('result').classList.remove('hidden');
            });
        });
    </script>
</body>
</html>
            ''')
    
    if not os.path.exists('templates/predictions.html'):
        with open('templates/predictions.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Saved Predictions - NBA Betting Prediction System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Saved Predictions</h1>
        
        <div class="predictions-container">
            {% if predictions %}
                {% for prediction in predictions %}
                    <div class="prediction-card">
                        <h3>{{ prediction.game_info.home_team }} vs {{ prediction.game_info.away_team }}</h3>
                        <p><strong>Date:</strong> {{ prediction.game_info.date }}</p>
                        <p><strong>Prediction:</strong> {{ prediction.prediction.prediction }}</p>
                        <p><strong>Win Probability:</strong> {{ "%.1f"|format(prediction.prediction.win_probability * 100) }}%</p>
                        <p><strong>Confidence:</strong> {{ "%.1f"|format(prediction.prediction.confidence * 100) }}%</p>
                        <p><strong>Recommendation:</strong> {{ prediction.prediction.recommendation }}</p>
                    </div>
                {% endfor %}
            {% else %}
                <p>No saved predictions found.</p>
            {% endif %}
        </div>
        
        <div class="links">
            <a href="/">Back to Home</a>
        </div>
    </div>
</body>
</html>
            ''')
    
    if not os.path.exists('static/style.css'):
        with open('static/style.css', 'w') as f:
            f.write('''
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
}

h2 {
    color: #444;
    margin-bottom: 20px;
}

.form-container, .result-container, .predictions-container {
    background-color: white;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="text"], input[type="number"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
}

button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background-color: #45a049;
}

.hidden {
    display: none;
}

.error {
    color: red;
    margin-bottom: 10px;
}

.links {
    text-align: center;
    margin-top: 20px;
}

.links a {
    color: #4CAF50;
    text-decoration: none;
}

.links a:hover {
    text-decoration: underline;
}

.prediction-card {
    background-color: #f9f9f9;
    border-left: 4px solid #4CAF50;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 4px;
}

.game-info {
    font-size: 18px;
    margin-bottom: 15px;
}

.team {
    font-weight: bold;
}

.prediction, .probability, .confidence, .recommendation {
    margin-bottom: 8px;
}

.label {
    font-weight: bold;
    display: inline-block;
    width: 140px;
}
            ''')
    
    # Run the Flask app
    app.run(debug=True)
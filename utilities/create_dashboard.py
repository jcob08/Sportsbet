# utilities/create_dashboard.py
import os
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def create_analysis_dashboard():
    """Create a visual dashboard of betting predictions and analysis"""
    print("Creating analysis dashboard...")
    
    # Set up directories
    output_dir = os.path.join("data", "dashboard")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find prediction files
    predictions_dir = os.path.join("data", "predictions")
    props_files = glob.glob(os.path.join(predictions_dir, "player_props_*.csv"))
    
    # Create props analysis if files exist
    if props_files:
        for file in props_files:
            try:
                prop_type = os.path.basename(file).replace("player_props_", "").replace("_predictions.csv", "")
                props_df = pd.read_csv(file)
                
                if 'over_prob' in props_df.columns:
                    # Calculate edge
                    if 'edge' not in props_df.columns:
                        props_df['edge'] = (props_df['over_prob'] - 0.5) * 2
                    
                    # Create distribution plot
                    plt.figure(figsize=(10, 6))
                    sns.histplot(props_df['edge'], kde=True, bins=20)
                    plt.title(f"Distribution of Betting Edge - {prop_type.title()} Props")
                    plt.xlabel("Edge (Over - 0.5) * 2")
                    plt.ylabel("Count")
                    plt.axvline(0, color='red', linestyle='--')
                    plt.savefig(os.path.join(output_dir, f"edge_distribution_{prop_type}.png"))
                    plt.close()
                    
                    # Create best bets table
                    best_bets = props_df[props_df['edge'].abs() > 0.1].sort_values('edge', ascending=False)
                    if len(best_bets) > 0:
                        best_bets_html = f"""
                        <div class="card">
                            <h2>Best {prop_type.title()} Prop Bets</h2>
                            <p>Bets with edge > 10%</p>
                            <table>
                                <tr>
                                    <th>Player</th>
                                    <th>Team</th>
                                    <th>Line</th>
                                    <th>Expected</th>
                                    <th>Edge</th>
                                    <th>Bet</th>
                                </tr>
                        """
                        
                        for _, row in best_bets.head(10).iterrows():
                            edge = row['edge']
                            bet_type = "OVER" if edge > 0 else "UNDER"
                            edge_str = f"{edge:.1%}"
                            edge_class = "positive" if edge > 0 else "negative"
                            
                            best_bets_html += f"""
                                <tr class="edge-{edge_class}">
                                    <td>{row['player_name']}</td>
                                    <td>{row['team']}</td>
                                    <td>{row['line']}</td>
                                    <td>{row['expected_value']:.1f}</td>
                                    <td>{edge_str}</td>
                                    <td>{bet_type}</td>
                                </tr>
                            """
                        
                        best_bets_html += """
                            </table>
                        </div>
                        """
                        
                        # Save to HTML file
                        with open(os.path.join(output_dir, f"best_{prop_type}_bets.html"), 'w') as f:
                            f.write(f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>Best {prop_type.title()} Prop Bets</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                    .card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                                    table {{ width: 100%; border-collapse: collapse; }}
                                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                                    th {{ background-color: #f2f2f2; }}
                                    .edge-positive {{ background-color: rgba(0, 255, 0, 0.1); }}
                                    .edge-negative {{ background-color: rgba(255, 0, 0, 0.1); }}
                                </style>
                            </head>
                            <body>
                                <h1>MLB Betting Analysis - {datetime.now().strftime('%Y-%m-%d')}</h1>
                                {best_bets_html}
                                <p>
                                    <img src="edge_distribution_{prop_type}.png" alt="Edge Distribution" style="max-width: 100%;">
                                </p>
                            </body>
                            </html>
                            """)
            except Exception as e:
                print(f"Error creating analysis for {file}: {e}")
    
    # Create main dashboard
    # Create an index.html that links to all the individual analyses
    dashboard_links = []
    for prop_type in ['hits', 'hr', 'strikeouts']:
        file_path = os.path.join(output_dir, f"best_{prop_type}_bets.html")
        if os.path.exists(file_path):
            dashboard_links.append(f'<li><a href="best_{prop_type}_bets.html">Best {prop_type.title()} Prop Bets</a></li>')
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLB Betting Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MLB Betting Analysis Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Available Reports</h2>
                    <ul>
                        {"".join(dashboard_links)}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Dashboard created at {os.path.join(output_dir, 'index.html')}")

if __name__ == "__main__":
    create_analysis_dashboard()
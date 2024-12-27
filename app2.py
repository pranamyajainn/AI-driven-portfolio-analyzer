from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import requests
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class PortfolioAnalyzer:
    def __init__(self, risk_tolerance, timeline):
        self.risk_tolerance = risk_tolerance
        self.timeline = timeline
        self.risk_scores = {
            "Conservative": 1,
            "Moderate": 2,
            "Aggressive": 3
        }
        self.timeline_scores = {
            "Short-term": 1,
            "Medium-term": 2,
            "Long-term": 3
        }

    def extract_portfolio_data(self, text):
        """Extract portfolio information from text using OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Extract portfolio information from the following text. 
            Identify stocks, bonds, ETFs, and their quantities/percentages:
            {text}"""
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
            )
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in AI extraction: {e}")
            return None

    def get_market_data(self, symbol):
        """Fetch market data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            return hist
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def calculate_optimal_allocation(self):
        """Calculate optimal portfolio allocation based on risk and timeline"""
        risk_score = self.risk_scores[self.risk_tolerance]
        timeline_score = self.timeline_scores[self.timeline]
        
        # Base allocations
        allocations = {
            "stocks": 40 + (risk_score * 10) + (timeline_score * 5),
            "bonds": 40 - (risk_score * 5) - (timeline_score * 5),
            "ETFs": 10 + (risk_score * 2),
            "cash": 10 - (risk_score * 2)
        }
        
        # Ensure minimum allocations
        allocations = {k: max(v, 5) for k, v in allocations.items()}
        
        # Normalize to 100%
        total = sum(allocations.values())
        allocations = {k: round((v/total) * 100, 1) for k, v in allocations.items()}
        
        return allocations

    def generate_visualization(self, allocations):
        """Generate portfolio visualization"""
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn')
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Pie chart
        colors = sns.color_palette('husl', n_colors=len(allocations))
        ax1.pie(allocations.values(), labels=allocations.keys(), autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Recommended Portfolio Allocation')
        
        # Bar chart
        ax2.bar(allocations.keys(), allocations.values(), color=colors)
        ax2.set_title('Asset Distribution')
        ax2.set_ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        
        # Save plot to bytes
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        
        return base64.b64encode(img.getvalue()).decode()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Extract text from image
        image = Image.open(file.stream)
        extracted_text = pytesseract.image_to_string(image)
        
        return jsonify({"success": True, "data": extracted_text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze_portfolio():
    try:
        data = request.json
        portfolio_text = data.get("details", "")
        risk_tolerance = data.get("risk_tolerance", "Moderate")
        timeline = data.get("timeline", "Medium-term")

        # Initialize analyzer
        analyzer = PortfolioAnalyzer(risk_tolerance, timeline)
        
        # Extract portfolio data using AI
        portfolio_data = analyzer.extract_portfolio_data(portfolio_text)
        
        # Calculate optimal allocation
        recommended_allocation = analyzer.calculate_optimal_allocation()
        
        # Generate visualization
        plot_url = analyzer.generate_visualization(recommended_allocation)
        
        # Generate analysis report
        analysis_report = f"""
        <div class="analysis-report">
            <h2>Portfolio Analysis Report</h2>
            
            <div class="risk-profile">
                <h3>Risk Profile Analysis</h3>
                <p>Risk Tolerance: <strong>{risk_tolerance}</strong></p>
                <p>Investment Timeline: <strong>{timeline}</strong></p>
            </div>
            
            <div class="portfolio-summary">
                <h3>Current Portfolio Summary</h3>
                <pre>{portfolio_data}</pre>
            </div>
            
            <div class="recommendations">
                <h3>Recommended Asset Allocation</h3>
                <ul>
                    {' '.join([f"<li><strong>{asset}:</strong> {percentage}%</li>" 
                             for asset, percentage in recommended_allocation.items()])}
                </ul>
            </div>
            
            <div class="visualization">
                <h3>Portfolio Visualization</h3>
                <img src="data:image/png;base64,{plot_url}" 
                     alt="Portfolio Analysis Visualization" 
                     style="max-width:100%; height:auto;">
            </div>
            
            <div class="action-items">
                <h3>Recommended Actions</h3>
                <ul>
                    <li>Consider rebalancing your portfolio to match the recommended allocation</li>
                    <li>Review and adjust your investments quarterly</li>
                    <li>Maintain an emergency fund of 3-6 months of expenses</li>
                </ul>
            </div>
        </div>
        """
        
        return jsonify({"success": True, "data": analysis_report})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

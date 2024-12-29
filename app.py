from flask import Flask, request, jsonify, render_template
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

from llama_index.llms.groq import Groq

class GroqLLM:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.api_key.startswith('gsk_'):
            raise ValueError("Invalid Groq API key format - should start with 'gsk_'")
        
        self.model = "llama3-70b-8192"
        self.client = Groq(model=self.model, api_key=self.api_key)

    def test_connection(self):
        """Test Groq connection with a simple prompt."""
        try:
            response = self.client.complete("Test prompt for connection.")
            return True, response.text.strip()
        except Exception as e:
            return False, f"Groq connection error: {e}"

    def analyze(self, prompt):
        """Process prompt with Groq and ensure valid JSON response."""
        try:
            formatted_prompt = f"""
            {prompt}
            
            IMPORTANT: Your response must be ONLY a valid JSON object with exactly this structure, nothing else:
            {{
                "allocation": {{
                    "stocks": <number>,
                    "bonds": <number>,
                    "etfs": <number>,
                    "cash": <number>
                }},
                "symbols": ["<symbol1>", "<symbol2>", ...],
                "risk_assessment": "<text>",
                "recommendations": ["<rec1>", "<rec2>", "<rec3>"]
            }}
            """
            
            response = self.client.complete(formatted_prompt)
            if not response or not hasattr(response, 'text'):
                raise ValueError("Invalid response from Groq API")
                
            response_text = response.text.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
                
            json_str = response_text[start_idx:end_idx]
            
            try:
                parsed_json = json.loads(json_str)
                required_keys = {'allocation', 'symbols', 'risk_assessment', 'recommendations'}
                if not all(key in parsed_json for key in required_keys):
                    raise ValueError("Missing required keys in JSON response")
                return json_str
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Error during Groq analysis: {e}")

def get_stock_data(symbols):
    """Fetch stock data using yfinance"""
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if not hist.empty:
                data[symbol] = {
                    'current_price': hist['Close'].iloc[-1],  # Using iloc for position-based indexing
                    'price_change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                    'volatility': hist['Close'].std() / hist['Close'].mean() * 100
                }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return data
def generate_visualization(portfolio_data, market_data):
    """Generate portfolio visualization"""
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create color map for consistency
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    # Create allocation pie chart
    plt.subplot(2, 1, 1)
    plt.pie(portfolio_data.values(), 
            labels=portfolio_data.keys(), 
            autopct='%1.1f%%', 
            colors=colors[:len(portfolio_data)])
    plt.title('Portfolio Allocation', pad=20, fontsize=14)
    
    # Create market performance chart
    if market_data:
        plt.subplot(2, 1, 2)
        symbols = list(market_data.keys())
        x = np.arange(len(symbols))
        width = 0.35
        
        returns = [data['price_change'] for data in market_data.values()]
        volatilities = [data['volatility'] for data in market_data.values()]
        
        # Create bars
        plt.bar(x - width/2, returns, width, label='Return %', color='#66b3ff')
        plt.bar(x + width/2, volatilities, width, label='Volatility %', color='#ff9999')
        
        # Customize the plot
        plt.xticks(x, symbols, rotation=45)
        plt.legend(loc='upper right')
        plt.title('Market Performance', pad=20, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_portfolio():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data received"}), 400

        portfolio_text = data.get("details", "")
        if not portfolio_text:
            return jsonify({"success": False, "error": "Portfolio details are required"}), 400

        risk_tolerance = data.get("risk_tolerance", "Moderate")
        timeline = data.get("timeline", "Medium-term")

        # Initialize Groq LLM
        llm = GroqLLM()
        is_connected, test_message = llm.test_connection()
        if not is_connected:
            return jsonify({"success": False, "error": test_message}), 500

        # Get analysis result
        analysis_result = llm.analyze(f"""
        Analyze this investment portfolio and provide recommendations:
        Portfolio details: {portfolio_text}
        Risk tolerance: {risk_tolerance}
        Investment timeline: {timeline}
        """)

        portfolio_analysis = json.loads(analysis_result)

        # Fetch market data for symbols
        market_data = get_stock_data(portfolio_analysis['symbols'])

        # Generate visualization
        plot_url = generate_visualization(portfolio_analysis['allocation'], market_data)

        # Format the response as HTML
        formatted_response = f"""
        <h3><b>Portfolio Analysis</b></h3>
        <ul>
            <li><b>Stocks</b>: {portfolio_analysis['allocation']['stocks']}%</li>
            <li><b>Bonds</b>: {portfolio_analysis['allocation']['bonds']}%</li>
            <li><b>ETFs</b>: {portfolio_analysis['allocation']['etfs']}%</li>
            <li><b>Cash</b>: {portfolio_analysis['allocation']['cash']}%</li>
        </ul>
        <h4><b>Recommendations:</b></h4>
        <ul>
            {''.join([f"<li>{rec}</li>" for rec in portfolio_analysis['recommendations']])}
        </ul>
        <h4><b>Risk Assessment:</b></h4>
        <p>{portfolio_analysis['risk_assessment']}</p>
        <h4><b>Symbols:</b></h4>
        <ul>
            {''.join([f"<li>{symbol}</li>" for symbol in portfolio_analysis['symbols']])}
        </ul>
        <h3><b>Market Data</b></h3>
        <ul>
            {''.join([f"<li><b>{symbol}</b>: Current Price: ${market_data[symbol]['current_price']:.2f}, Price Change: {market_data[symbol]['price_change']:.2f}%, Volatility: {market_data[symbol]['volatility']:.2f}%</li>" for symbol in market_data])}
        </ul>
        <h3><b>Visualization</b></h3>
        <img src="data:image/png;base64,{plot_url}" alt="Portfolio Visualization" style="max-width:100%; height:auto;">
        """

        return jsonify({
            "success": True,
            "data": formatted_response
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "details": error_details
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

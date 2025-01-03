# Import required libraries and modules
from flask import Flask, request, jsonify, render_template  # Flask web framework components
import yfinance as yf  # Yahoo Finance API wrapper
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization
import io  # Input/output operations
import base64  # Encoding/decoding binary data
import os  # Operating system interface
import requests  # HTTP library
import json  # JSON encoder and decoder
from dotenv import load_dotenv  # Environment variables management
from sklearn.preprocessing import MinMaxScaler  # Data scaling
from scipy.stats import norm  # Statistical functions
from dataclasses import dataclass  # Data class decorator
from typing import List, Dict, Optional  # Type hints
from llama_index.llms.groq import Groq  # Groq LLM interface

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Define analysis parameters data class
@dataclass
class AnalysisParameters:
    """Data class to store portfolio analysis parameters"""
    risk_weight: float = 0.4  # Weight for risk consideration
    return_weight: float = 0.4  # Weight for return consideration
    volatility_weight: float = 0.2  # Weight for volatility consideration
    lookback_period: str = "1y"  # Historical data period
    min_symbols: int = 3  # Minimum number of symbols in portfolio
    max_symbols: int = 10  # Maximum number of symbols in portfolio
    confidence_level: float = 0.95  # Confidence level for VaR calculation
    rebalancing_threshold: float = 0.05  # Threshold for portfolio rebalancing

class EnhancedGroqLLM:
    """Enhanced LLM class for portfolio analysis using Groq"""
    
    def __init__(self, parameters: Optional[AnalysisParameters] = None):
        """Initialize EnhancedGroqLLM with API key and parameters"""
        # Get API key from environment variables
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.api_key.startswith('gsk_'):
            raise ValueError("Invalid Groq API key format - should start with 'gsk_'")
        
        # Set model and parameters
        self.model = "llama3-70b-8192"
        self.client = Groq(model=self.model, api_key=self.api_key)
        self.params = parameters or AnalysisParameters()

    def test_connection(self):
        """Test connection to Groq API"""
        try:
            response = self.client.complete("Test prompt for connection.")
            return True, response.text.strip()
        except Exception as e:
            return False, f"Groq connection error: {e}"
        
    def calculate_portfolio_metrics(self, market_data: Dict) -> Dict:
        """Calculate advanced portfolio metrics including returns, volatility, and risk measures"""
        portfolio_metrics = {
            'total_return': 0,
            'weighted_volatility': 0,
            'sharpe_ratio': 0,
            'var_95': 0
        }
        
        returns = []
        weights = []
        
        # Calculate portfolio-level metrics
        for symbol, data in market_data.items():
            returns.append(data['price_change'] / 100)
            volatility = data['volatility'] / 100
            weights.append(1 / len(market_data))  # Equal weighting
            
            # Calculate weighted metrics
            portfolio_metrics['total_return'] += (data['price_change'] / 100) * (1 / len(market_data))
            portfolio_metrics['weighted_volatility'] += volatility * (1 / len(market_data))
        
        # Calculate Sharpe Ratio
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        portfolio_metrics['sharpe_ratio'] = (portfolio_metrics['total_return'] - risk_free_rate) / portfolio_metrics['weighted_volatility']
        
        # Calculate Value at Risk (VaR)
        portfolio_metrics['var_95'] = norm.ppf(1 - self.params.confidence_level) * portfolio_metrics['weighted_volatility']
        
        return portfolio_metrics

    def analyze(self, prompt: str, market_data: Optional[Dict] = None) -> str:
        """Perform enhanced analysis combining LLM with market data"""
        try:
            # Format prompt with analysis parameters
            formatted_prompt = f"""
            {prompt}
            
            Consider the following parameters in your analysis:
            - Risk Weight: {self.params.risk_weight}
            - Return Weight: {self.params.return_weight}
            - Volatility Weight: {self.params.volatility_weight}
            - Confidence Level: {self.params.confidence_level * 100}%
            
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
                "recommendations": ["<rec1>", "<rec2>", "<rec3>"],
                "rebalancing_needed": <boolean>
            }}
            """
            
            # Get LLM response
            response = self.client.complete(formatted_prompt)
            if not response or not hasattr(response, 'text'):
                raise ValueError("Invalid response from Groq API")
            
            # Parse and validate response
            response_text = response.text.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            base_analysis = json.loads(response_text[start_idx:end_idx])
            
            # Enhance analysis with market data if available
            if market_data:
                portfolio_metrics = self.calculate_portfolio_metrics(market_data)
                
                # Add recommendations based on metrics
                if portfolio_metrics['sharpe_ratio'] < 0.5:
                    base_analysis['recommendations'].append(
                        "Consider rebalancing portfolio due to low risk-adjusted returns"
                    )
                
                if abs(portfolio_metrics['var_95']) > self.params.rebalancing_threshold:
                    base_analysis['rebalancing_needed'] = True
                    base_analysis['recommendations'].append(
                        f"Portfolio VaR ({portfolio_metrics['var_95']:.2%}) exceeds threshold"
                    )
                
                # Add quantitative metrics to risk assessment
                base_analysis['risk_assessment'] += f" Portfolio Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}"
            
            return json.dumps(base_analysis)
            
        except Exception as e:
            raise RuntimeError(f"Error during enhanced analysis: {e}")

def get_enhanced_stock_data(symbols: List[str], params: AnalysisParameters) -> Dict:
    """Fetch and calculate enhanced stock market data with additional metrics"""
    data = {}
    for symbol in symbols:
        try:
            # Get historical data from Yahoo Finance
            stock = yf.Ticker(symbol)
            hist = stock.history(period=params.lookback_period)
            
            if not hist.empty:
                # Calculate technical indicators and metrics
                returns = hist['Close'].pct_change().dropna()
                rolling_std = returns.rolling(window=20).std()
                
                # Store calculated metrics
                data[symbol] = {
                    'current_price': hist['Close'].iloc[-1],
                    'price_change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                    'volatility': hist['Close'].std() / hist['Close'].mean() * 100,
                    'beta': returns.cov(returns) / returns.var() if len(returns) > 1 else 1,
                    'max_drawdown': ((hist['Close'].cummax() - hist['Close']) / hist['Close'].cummax()).max() * 100,
                    'rolling_volatility': rolling_std.iloc[-1] * np.sqrt(252) * 100 if len(rolling_std) > 0 else None
                }
        except Exception as e:
            print(f"Error fetching enhanced data for {symbol}: {e}")
    return data

def generate_enhanced_visualization(portfolio_data: Dict, market_data: Dict) -> str:
    """Generate enhanced portfolio visualization with multiple charts"""
    # Set matplotlib backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    colors = sns.color_palette("husl", n_colors=max(len(portfolio_data), len(market_data)))
    
    # Create portfolio allocation pie chart
    plt.subplot(3, 1, 1)
    plt.pie(portfolio_data.values(), 
            labels=portfolio_data.keys(), 
            autopct='%1.1f%%', 
            colors=colors[:len(portfolio_data)])
    plt.title('Portfolio Allocation', pad=20, fontsize=14)
    
    # Create performance metrics chart
    plt.subplot(3, 1, 2)
    symbols = list(market_data.keys())
    metrics = ['price_change', 'volatility', 'max_drawdown']
    x = np.arange(len(symbols))
    width = 0.25
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = [data[metric] for data in market_data.values()]
        plt.bar(x + (i-1)*width, values, width, label=metric.replace('_', ' ').title())
    
    plt.xticks(x, symbols, rotation=45)
    plt.legend(loc='upper right')
    plt.title('Performance Metrics', pad=20, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create rolling volatility chart
    plt.subplot(3, 1, 3)
    for symbol, data in market_data.items():
        if data['rolling_volatility'] is not None:
            plt.axhline(y=data['rolling_volatility'], 
                       label=f"{symbol} Rolling Vol", 
                       linestyle='--')
    
    plt.title('Rolling Volatility (20-day)', pad=20, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

# Flask routes
@app.route("/")
def home():
    """Render home page"""
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_portfolio():
    """Handle portfolio analysis API endpoint"""
    try:
        # Validate request data
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data received"}), 400

        # Extract input parameters
        portfolio_text = data.get("details", "")
        risk_tolerance = data.get("risk_tolerance", "Moderate")
        timeline = data.get("timeline", "Medium-term")

        # Define risk parameters based on risk tolerance
        risk_params = {
            "Conservative": AnalysisParameters(risk_weight=0.6, return_weight=0.2, volatility_weight=0.2),
            "Moderate": AnalysisParameters(risk_weight=0.4, return_weight=0.4, volatility_weight=0.2),
            "Aggressive": AnalysisParameters(risk_weight=0.2, return_weight=0.6, volatility_weight=0.2)
        }
        
        # Get appropriate parameters
        params = risk_params.get(risk_tolerance, AnalysisParameters())
        
        # Initialize LLM
        llm = EnhancedGroqLLM(parameters=params)
        
        # Test API connection
        is_connected, test_message = llm.test_connection()
        if not is_connected:
            return jsonify({"success": False, "error": test_message}), 500
        
        # Prepare analysis prompt
        analysis_prompt = f"""
        Analyze this investment portfolio and provide recommendations:
        Portfolio details: {portfolio_text}
        Risk tolerance: {risk_tolerance}
        Investment timeline: {timeline}
        """
        
        # Get initial analysis
        portfolio_analysis = json.loads(llm.analyze(analysis_prompt))
        
        # Get market data
        market_data = get_enhanced_stock_data(portfolio_analysis['symbols'], params)
        
        # Get final analysis with market data
        final_analysis = json.loads(llm.analyze(analysis_prompt, market_data))
        
        # Generate visualization
        plot_url = generate_enhanced_visualization(final_analysis['allocation'], market_data)
        
        # Build market analysis HTML
        market_analysis_html = ""
        for symbol, data in market_data.items():
            market_analysis_html += f"""
                <li><b>{symbol}</b>:
                    <ul>
                        <li>Current Price: ${data['current_price']:.2f}</li>
                        <li>Price Change: {data['price_change']:.2f}%</li>
                        <li>Volatility: {data['volatility']:.2f}%</li>
                        <li>Beta: {data['beta']:.2f}</li>
                        <li>Maximum Drawdown: {data['max_drawdown']:.2f}%</li>
                    </ul>
                </li>
            """
        
        # Format response HTML
        formatted_response = f"""
        <h3><b>Enhanced Portfolio Analysis</b></h3>
        <ul>
            <li><b>Stocks</b>: {final_analysis['allocation']['stocks']}%</li>
            <li><b>Bonds</b>: {final_analysis['allocation']['bonds']}%</li>
            <li><b>ETFs</b>: {final_analysis['allocation']['etfs']}%</li>
            <li><b>Cash</b>: {final_analysis['allocation']['cash']}%</li>
        </ul>
        <h4><b>Risk Assessment:</b></h4>
        <p>{final_analysis['risk_assessment']}</p>
        <h4><b>Recommendations:</b></h4>
        <ul>
            {''.join([f"<li>{rec}</li>" for rec in final_analysis['recommendations']])}
        </ul>
        <h4><b>Market Analysis:</b></h4>
        <ul>
            {market_analysis_html}
        </ul>
        <h3><b>Portfolio Visualization</b></h3>
        <img src="data:image/png;base64,{plot_url}" alt="Enhanced Portfolio Visualization" style="max-width:100%; height:auto;">
        """

        # Return successful response
        return jsonify({
            "success": True,
            "data": formatted_response
        })

    except Exception as e:
        # Handle errors
        import traceback
        error_details = traceback.format_exc()

Personalized Investment Portfolio Advisor

#Introduction:

The Personalized Investment Portfolio Advisor is a web application that provides tailored investment strategies to meet diverse client needs.
By leveraging Groq’s API and real-time market data, this tool empowers financial institutions and individual advisors to deliver customized portfolio recommendations.

Key Features
	•	Risk Tolerance Analysis:
	•	Customize strategies based on risk preferences: Conservative, Moderate, or Aggressive.
	•	Financial Goal Alignment:
	•	Focus on specific goals such as Retirement, Education, or Wealth Accumulation.
	•	Investment Timeline:
	•	Adapt strategies for Short-term, Medium-term, or Long-term investments.
	•	Current Financial Standing:
	•	Incorporate factors like Income, Expenses, Savings, and Debt.
	•	Market Data Insights:
	•	Utilize real-time market trends for better-informed strategies.

 Technology Stack
	1.	Frontend:
	•	HTML, CSS (TailwindCSS for styling), and JavaScript (for dynamic UI).
	2.	Backend:
	•	Flask (Python): Handles API endpoints and data processing.
	•	Groq API: Processes portfolio analysis and provides actionable recommendations.
	3.	Data and Visualization:
	•	yfinance: Fetches real-time stock market data.
	•	Matplotlib and Seaborn: Generate portfolio visualizations.
	4.	Environment Management:
	•	dotenv: Handles sensitive API keys securely.
#How It Works

Input

Users provide:
	•	Portfolio details (e.g., stock allocations like AAPL: 30%, TSLA: 15%).
	•	Risk Tolerance: Conservative, Moderate, or Aggressive.
	•	Investment Timeline: Short-term, Medium-term, or Long-term.

Processing
	1.	The portfolio is sent to Groq’s API for analysis.
	2.	The API returns:
	•	Allocation suggestions (stocks, bonds, ETFs, cash).
	•	Recommendations to optimize the portfolio.
	•	Risk assessment based on the provided details.
	3.	yfinance fetches real-time market data for stocks.
	4.	A visualization (pie and bar charts) is generated using Matplotlib.

Output

The app displays:
	•	Portfolio Analysis:
	•	Allocation percentages.
	•	Risk assessment.
	•	Specific recommendations.
	•	Market Data:
	•	Current prices, price changes, and volatility for the selected stocks.
	•	Visualization:
	•	Graphical representation of the portfolio allocation and market performance.

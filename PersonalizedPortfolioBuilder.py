#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:32:58 2024

@author: ekarthpatel
"""
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing, AutoARIMA, Prophet
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape


############################################################################### INVESTOR RISK ABILITY QUESTIONS

def get_risk_level(question, options, responses):
    print(question)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    choice = int(input("Enter the number corresponding to your choice: "))
    return responses[choice - 1]

def determine_overall_risk(risk_scores):
    high_count = risk_scores.count("High")
    medium_count = risk_scores.count("Medium")
    low_count = risk_scores.count("Low")
    
    if high_count > medium_count and high_count > low_count:
        return "High"
    elif medium_count > low_count:
        return "Medium"
    else:
        return "Low"

def main():
    risk_scores = []

    # Question 1: Age
    question1 = "What is your age group?"
    options1 = ["20-40", "41-60", "60+"]
    responses1 = ["High", "Medium", "Low"]
    risk_scores.append(get_risk_level(question1, options1, responses1))

    # Question 2: Annual Income
    question2 = "What is your annual income range?"
    options2 = ["5000-15000 USD", "15000-50000 USD", "50000+ USD"]
    responses2 = ["Low", "Medium", "High"]
    risk_scores.append(get_risk_level(question2, options2, responses2))

    # Question 3: Investment Time Horizon
    question3 = "What is your investment time horizon?"
    options3 = ["0-1 year", "1-5 years", "5+ years"]
    responses3 = ["Low", "Medium", "High"]
    risk_scores.append(get_risk_level(question3, options3, responses3))

    # Question 4: Investment Amount
    question4 = "What is your investment amount?"
    options4 = ["0-10K USD", "10K-50K USD", "50K+ USD"]
    responses4 = ["Low", "Medium", "High"]
    risk_scores.append(get_risk_level(question4, options4, responses4))

    # Determine overall risk-taking ability
    overall_risk = determine_overall_risk(risk_scores)
    
    # Save the result to a variable and print it
    print("\nYour overall risk-taking ability is:", overall_risk)
    return overall_risk  # Save as variable

if __name__ == "__main__":
    overall_risk_value = main()  # Save the return value of main() to a variable
   
    
#%%                       

##############################################################################  GET PE, PB, SD, CARG, DE FOR ALL CO.s and SECTOR AVERAGE

# Ensure unique tickers
sp500_tickers = list(set([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "V", "JNJ",
    "WMT", "PG", "MA", "PYPL", "UNH", "DIS", "HD", "INTC", "CSCO", "MRK",
    "XOM", "PFE", "VZ", "KO", "PEP", "ABT", "CRM", "NKE", "AVGO", "WFC",
    "T", "LLY", "ORCL", "MDT", "GE", "NEE", "CVX", "COST", "QCOM", "ABNB", "DUK", "NEE", 
    "SO", "EXC", "XEL", "AEP", "CAT", "HON", "MMM", "RTX", "UPS", "DE", "GE", "COP", "EOG", "SLB"
    # Add more tickers as needed...
]))

# Prepare a list to hold the results
data = []

for ticker in sp500_tickers:
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")  # Fetch 2 years of data
        
        # Check if data is available
        if hist.empty or "Close" not in hist.columns:
            print(f"Skipping {ticker}: No historical data available.")
            continue
        
        # Calculate CAGR
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        years = 2  # Fixed for 2 years
        cagr = ((end_price / start_price) ** (1 / years)) - 1

        # Calculate standard deviation of returns
        hist["Daily Returns"] = hist["Close"].pct_change().dropna()
        std_dev = hist["Daily Returns"].std() * np.sqrt(252)  # Annualized standard deviation

        # Fetch financial ratios from stock info
        info = stock.info
        pe_ratio = info.get("trailingPE", None)
        pb_ratio = info.get("priceToBook", None)
        debt_to_equity = info.get("debtToEquity", None)
        roe = info.get("returnOnEquity", None)
        roa = info.get("returnOnAssets", None)
        profit_margin = info.get("profitMargins", None)
        operating_margin = info.get("operatingMargins", None)
        gross_margin = info.get("grossMargins", None)
        

        # Append the collected data
        data.append({
            "Ticker": ticker,
            "Company Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "P/E Ratio": pe_ratio,
            "P/B Ratio": pb_ratio,
            "ROE Ratio": roe,
            "ROA Ratio": roa,
            "Profit Margin": profit_margin,
            "Operating Margin": operating_margin,
            "Gross Margin": gross_margin,
            "Debt-to-Equity": debt_to_equity,
            "Std Deviation": std_dev,
            "CAGR": cagr,
        })
    except Exception as e:
        if "delisted" in str(e).lower():
            print(f"{ticker}: Possibly delisted; skipping.")
        else:
            print(f"Error fetching data for {ticker}: {e}")

# Convert collected data into a DataFrame
df = pd.DataFrame(data)

# Check if DataFrame is not empty
if not df.empty:
    # Convert numeric columns to appropriate types for calculations
    numeric_columns = ["P/E Ratio", "P/B Ratio", "Debt-to-Equity", "Std Deviation", "CAGR"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Group by sector and calculate the mean for numeric columns
    sector_summary = df.groupby("Sector")[numeric_columns].mean().round(2)

    # Export results to CSV files
    #df.to_csv("sp500_2y_data.csv", index=False)
    #sector_summary.to_csv("sp500_2y_sector_summary.csv")

    # Display results
    print("Collected Data:\n", df)
    print("\nSector Summary:\n", sector_summary)
else:
    print("No valid data collected.")

#%%

############################################################################### GIVE BUY OR SELL RATINGS TO ALL CO.s

def evaluate_buy_rating(row, sector_summary):
    # Get the sector average values for the company’s sector
    sector_avg = sector_summary.loc[row["Sector"]]
    
    # Calculate whether each ratio is 'good' or 'bad'
    good_count = 0
    
    # P/E Ratio: Good if company's P/E is less than 110% of sector average
    if row["P/E Ratio"] and row["P/E Ratio"] < 1.1 * sector_avg["P/E Ratio"]:
        good_count += 1
    
    # P/B Ratio: Good if company's P/B is less than 110% of sector average
    if row["P/B Ratio"] and row["P/B Ratio"] < 1.1 * sector_avg["P/B Ratio"]:
        good_count += 1
    
    # Debt-to-Equity: Good if company's debt-to-equity is less than 110% of sector average
    if row["Debt-to-Equity"] and row["Debt-to-Equity"] < 1.1 * sector_avg["Debt-to-Equity"]:
        good_count += 1
    
    # Std Deviation: Good if company's std deviation is less than sector average
    if row["Std Deviation"] and row["Std Deviation"] < sector_avg["Std Deviation"]:
        good_count += 1
    
    # CAGR: Good if company's return is higher than sector average
    if row["CAGR"] and row["CAGR"] > sector_avg["CAGR"]:
        good_count += 1
    
    # If 3 or more ratios are good, the company is a "good" buy
    if good_count >= 3:
        return "Good"
    else:
        return "Bad"

# Apply the evaluation function to each row in the DataFrame
df["Buy Rating"] = df.apply(lambda row: evaluate_buy_rating(row, sector_summary), axis=1)

# Export updated results to CSV
df.to_csv("sp500_2y_with_ratings.csv", index=False)

# Display the updated DataFrame with ratings
print("Collected Data with Buy Ratings:\n", df[['Ticker', 'Sector', 'P/E Ratio', 'P/B Ratio', 'Debt-to-Equity', 'Std Deviation', 'CAGR', 'Buy Rating']])



#%%

############################################################################### SORT BUY STOCKS ACC TO CAGR, RECOMMEND ACC TO RISK ABILITY

def recommend_stocks(overall_risk, diversify, df):
    # Filter "Good" buy rating stocks
    good_stocks = df[df["Buy Rating"] == "Good"]

    # Sort by CAGR (highest returns)
    good_stocks = good_stocks.sort_values(by="CAGR", ascending=False)

    # Risk-based filtering
    if overall_risk == "Low":
        recommended_stocks = good_stocks[good_stocks["Std Deviation"] <= 0.20]
    elif overall_risk == "Medium":
        recommended_stocks = good_stocks[good_stocks["Std Deviation"] <= 0.28]
    else:  # High risk investors
        recommended_stocks = good_stocks

    # Ensure there are enough stocks to choose from
    if recommended_stocks.empty:
        print("\nNo stocks meet the criteria for your risk level.")
        return []

    if diversify:
        # Ensure sector diversification
        diversified_stocks = recommended_stocks.groupby("Sector").head(1)  # Pick top 1 stock per sector
        return diversified_stocks.head(5)  # Limit to top 5 diversified stocks
    else:
        # Simply take the top 5 based on CAGR
        return recommended_stocks.head(5)

# Ask investor about diversification preference
def get_investor_preference():
    print("\nDo you want a sector-diversified portfolio or the stocks with the highest past returns?")
    options = ["Sector-Diversified Portfolio", "Highest Past Returns"]
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    choice = int(input("Enter the number corresponding to your choice: "))
    return choice == 1  # Return True if diversification is preferred

# Main recommendation logic
if not df.empty and "Buy Rating" in df.columns:
    diversify = get_investor_preference()
    recommended_stocks = recommend_stocks(overall_risk_value, diversify, df)

    if not recommended_stocks.empty:
        print("\nRecommended Stocks:")
        print(recommended_stocks[["Ticker", "Company Name", "Sector", "CAGR", "Std Deviation", "Buy Rating"]])
        recommended_stocks.to_csv("recommended_stocks.csv", index=False)
    else:
        print("\nNo stocks meet the criteria for recommendation.")
else:
    print("\nStock data is unavailable or improperly processed.")
    
#%%

############################################################################### CALCULATE THE EXP RETURN AND STD DEVIATION OF RECOMMENDED PORTFOLIO

def calculate_portfolio_metrics(recommended_stocks):
    
    # Extract CAGR and Standard Deviation from recommended_stocks DataFrame
    cagr = recommended_stocks["CAGR"].values  # Expected returns
    std_dev = recommended_stocks["Std Deviation"].values  # Standard deviations

    # Number of stocks and equal weights
    n_stocks = len(cagr)
    weights = np.full(n_stocks, 1 / n_stocks)  # Equal weights

    # Placeholder: Historical correlation matrix (could replace with computed correlations)
    # Assuming correlation of 0.25 for simplicity, diagonal is 1
    correlation_matrix = np.eye(n_stocks) + 0.25 * (np.ones((n_stocks, n_stocks)) - np.eye(n_stocks))   #  confirm with scott
    
    # Calculate expected portfolio return
    expected_return = np.dot(weights, cagr)

    # Calculate portfolio variance
    portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(std_dev, std_dev), weights))

    # Calculate portfolio standard deviation
    portfolio_std_dev = np.sqrt(portfolio_variance)

    return expected_return, portfolio_std_dev

# Main recommendation logic
if not df.empty and "Buy Rating" in df.columns:
   # diversify = get_investor_preference()
    recommended_stocks = recommend_stocks(overall_risk_value, diversify, df)

    if not recommended_stocks.empty:
        print("\nRecommended Stocks:")
        print(recommended_stocks[["Ticker", "Company Name", "Sector", "CAGR", "Std Deviation", "Buy Rating"]])
        recommended_stocks.to_csv("recommended_stocks.csv", index=False)

        # Calculate expected return and standard deviation of the portfolio
        expected_return, portfolio_std_dev = calculate_portfolio_metrics(recommended_stocks)

        print(f"\nExpected Portfolio Return: {expected_return:.2%}")
        print(f"Expected Portfolio Standard Deviation: {portfolio_std_dev:.2%}")
    else:
        print("\nNo stocks meet the criteria for recommendation.")
else:
    print("\nStock data is unavailable or improperly processed.")

#%%
def display_fundamental_ratios(stocks):
    for _, row in stocks.iterrows():
        print(f"\n{row['Company Name']} ({row['Ticker']})")
        print(f"P/E Ratio: {row['P/E Ratio']}")
        print(f"P/B Ratio: {row['P/B Ratio']}")
        print(f"Debt-to-Equity Ratio: {row['Debt-to-Equity']}")
        print(f"CAGR: {row['CAGR']:.2%}")
        print(f"Standard Deviation: {row['Std Deviation']:.2%}")
        print(f"ROE Ratio: {row['ROE Ratio']:.2%}")
        print(f"ROA Ratio: {row['ROA Ratio']:.2%}")
        print(f"Profit Margin: {row['Profit Margin']:.2%}")
        print(f"Operating Margin: {row['Operating Margin']:.2%}")
        print(f"Gross Margin: {row['Gross Margin']:.2%}")
        

# Function to plot price chart for a given stock
def plot_price_chart(ticker):
    # Fetch the historical data for the given stock
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")  # Fetch 6 months of data

    # Check if data is available
    if hist.empty or "Close" not in hist.columns:
        print(f"Skipping {ticker}: No historical data available.")
        return

    # Create a plotly line chart for the stock price
    fig = go.Figure()

    # Add closing price line chart
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=ticker))

    # Update the layout for the chart
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    # Show the plot
    fig.show()

# Function to display price charts for the top 5 recommended stocks
def display_price_charts(recommended_stocks):
    for _, row in recommended_stocks.iterrows():
        ticker = row["Ticker"]
        plot_price_chart(ticker)

# After getting the recommended stocks
if not recommended_stocks.empty:
    print("\nFundamental Ratios of Recommended Stocks:")
    display_fundamental_ratios(recommended_stocks)

    # Display price charts for the 5 recommended stocks
    print("\nDisplaying Price Charts:")
    display_price_charts(recommended_stocks)
else:
    print("No recommended stocks available.")
    
 #%%
 #TIME
 
def download_price_data_for_recommended_stocks(recommended_stocks, period="2y"):
    # Create a dictionary to hold the price data for each recommended stock
    price_data_dict = {}

    for _, row in recommended_stocks.iterrows():
        ticker = row["Ticker"]
        print(f"\nDownloading price data for {ticker}...")

        # Fetch the historical data for the given stock (last 2 years)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)  # Fetch 2 years of data

        # Check if data is available
        if hist.empty or "Close" not in hist.columns:
            print(f"Skipping {ticker}: No historical data available.")
            continue
        
        # Add the data to the dictionary
        price_data_dict[ticker] = hist

    return price_data_dict

# After getting the recommended stocks
if not recommended_stocks.empty:
    # Download price data for the top 5 recommended stocks
    price_data_dict = download_price_data_for_recommended_stocks(recommended_stocks)

    # Optionally, you can display the downloaded data for verification
    for ticker, data in price_data_dict.items():
        print(f"\nPrice data for {ticker}:")
        print(data.head())  # Display the first few rows of the data

    # Now, let's assign each stock's data to separate variables (tables)
    for ticker, data in price_data_dict.items():
        globals()[f'{ticker}_price_data'] = data
        print(f"Price data for {ticker} is stored in the variable {ticker}_price_data")
else:
    print("No recommended stocks available.")
 
 #%%
 
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to forecast and plot prices
def forecast_and_plot(stock_ticker, price_data):
    # Prepare data for Prophet
    price_data = price_data.reset_index()
    price_data['Date'] = price_data['Date'].dt.tz_localize(None)  # Remove timezone
    price_data = price_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(price_data)
    
    # Create a dataframe for future predictions (next 30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(price_data['ds'], price_data['y'], label='Historical Prices')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Prices', color='orange')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='orange',
        alpha=0.2,
        label='Prediction Interval'
    )
    plt.title(f"Price Prediction for {stock_ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

# Loop through the stored data and forecast for each stock
for stock_ticker, price_data in price_data_dict.items():
    print(f"Forecasting for {stock_ticker}...")
    forecast_and_plot(stock_ticker, price_data)
    
#%%

# Function to forecast and plot prices
def forecast_and_plot(stock_ticker, price_data):
    # Prepare data for Prophet
    price_data = price_data.reset_index()
    price_data['Date'] = price_data['Date'].dt.tz_localize(None)  # Remove timezone
    price_data = price_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(price_data)
    
    # Create a dataframe for future predictions (next 120 days)
    future = model.make_future_dataframe(periods=120)
    forecast = model.predict(future)
    
    # Separate historical and predicted parts
    historical_forecast = forecast[forecast['ds'] <= price_data['ds'].max()]
    predicted_forecast = forecast[forecast['ds'] > price_data['ds'].max()]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(price_data['ds'], price_data['y'], label='Historical Prices', color='black')
    plt.plot(historical_forecast['ds'], historical_forecast['yhat'], label='Model Fit (Historical)', color='blue')
    plt.plot(predicted_forecast['ds'], predicted_forecast['yhat'], label='Forecasted Prices', color='green')
    plt.fill_between(
        predicted_forecast['ds'],
        predicted_forecast['yhat_lower'],
        predicted_forecast['yhat_upper'],
        color='orange',
        alpha=0.2,
        label='Prediction Interval'
    )
    plt.title(f"Price Prediction for {stock_ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

# Loop through the stored data and forecast for each stock
for stock_ticker, price_data in price_data_dict.items():
    print(f"Forecasting for {stock_ticker}...")
    forecast_and_plot(stock_ticker, price_data)
    
#%%


import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Step 1: Prepare data (historical stock prices)
# price_data_dict contains price data for each stock
price_df = pd.concat(
    [data['Close'] for data in price_data_dict.values()], axis=1
)
price_df.columns = price_data_dict.keys()

# Step 2: Calculate expected returns and covariance
expected_returns = mean_historical_return(price_df)
covariance_matrix = CovarianceShrinkage(price_df).ledoit_wolf()

# Step 3: Initialize the Efficient Frontier
ef = EfficientFrontier(expected_returns, covariance_matrix)

# Step 4: Optimize for maximum Sharpe ratio
weights = ef.max_sharpe()  # Maximize return per unit of risk
cleaned_weights = ef.clean_weights()  # Cleaned weight allocations

# Display the optimized weights
print("Optimized Portfolio Weights:")
for stock, weight in cleaned_weights.items():
    print(f"{stock}: {weight:.2%}")

# Step 5: Portfolio performance metrics
performance = ef.portfolio_performance(verbose=True)

# Optional: Convert weights to discrete allocation
latest_prices = get_latest_prices(price_df)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da.lp_portfolio()
print("\nDiscrete Allocation:", allocation)
print("Funds Remaining: ${:.2f}".format(leftover))

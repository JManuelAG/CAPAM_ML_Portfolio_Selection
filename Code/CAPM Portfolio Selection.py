# Put it all in a function

from scipy import stats
import pandas as pd
import numpy as np

import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime

import scipy.optimize as sco

import matplotlib.pyplot as plt
%matplotlib inline

# Get S&P 500 ticketer symbol
payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = pd.DataFrame(payload[0])
stock_symbols = df['Symbol'].values.tolist()
#stock_symbols = stock_symbols[0:20]

# Variables
start = '2020-01-01'
end = '2022-01-01'
bench_mark = '^GSPC'

# Moving variables

sp500_dict = {'compny': [],
              'Betha': [],
              'Alpha': [],
              'expected_return': []}


# Read bench March
df_bch = pdr.DataReader(bench_mark,start=start, end=end)['Adj Close']
df_bch.name = bench_mark

# Return
df_bch = df_bch.pct_change().dropna()

# Market Returns
returns = df_bch

for stock in stock_symbols:
    # Read Stock 
    df = pdr.DataReader(stock,start=start, end=end)['Adj Close']
    df.name = stock

    # Quarterly data
    #df_bch.resample('Q').last()
    #df.resample('Q').last()

    # Percent changes on a day
    df = df.pct_change().dropna()

    # Add to market returns
    returns = pd.concat([returns, df], axis=1).dropna()

    # Set the risk-free rate (e.g., 10-year Treasury yield)
    risk_free_rate = 0.04

    # Build Model 
    try:
        LR = stats.linregress(df_bch, df)
        beta,alpha,r_val,p_val,std_err = LR
    except Exception:
    # Code block to handle any exception generically
        beta = 1
        alpha = np.nan

    # Calculate expected return using CAPM formula
    expected_return = risk_free_rate + beta * (np.mean(market_returns) - risk_free_rate)
    
    # Add them to the stock
    sp500_dict['compny'].append(stock)
    sp500_dict['Betha'].append(beta)
    sp500_dict['Alpha'].append(alpha)
    sp500_dict['expected_return'].append(expected_return)


# Calculate Cov Matrix
cov_matrix = returns.iloc[:,1:].cov()

# Calculate expected returns using CAPM
betas = np.array(sp500_dict['Betha'])
expected_returns = np.array(sp500_dict['expected_return'])

# Define the portfolio optimization objective function
def portfolio_objective(weights, expected_returns, cov_matrix, target_return):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    penalty = 100 * abs(portfolio_return - target_return)
    return portfolio_volatility + penalty

# Define the constraint for maximum weight per stock
def max_weight_constraint(weights):
    return np.sum(weights) - 1.0

# Set the target return for the portfolio
target_return = 0.15

# Perform portfolio optimization to find the optimal weights
num_assets = len(stock_symbols)
constraints = ({'type': 'eq', 'fun': max_weight_constraint},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return})
bounds = tuple((0, 0.25) for _ in range(num_assets))  # Restrict weights to a maximum of 30%
initial_weights = np.ones(num_assets) / num_assets
optimal_weights = sco.minimize(portfolio_objective, initial_weights,
                                args=(expected_returns, cov_matrix, target_return),
                                method='SLSQP', bounds=bounds, constraints=constraints)['x']

# Print the optimal portfolio weights
for i, symbol in enumerate(stock_symbols):
    print(f"{symbol}: {optimal_weights[i]}")

# Filter stocks based on minimum weight threshold
filtered_stocks = [stock for stock, weight in zip(stock_symbols, optimal_weights) if weight > .01]

filtered_stocks
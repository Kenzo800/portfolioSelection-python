#Import the pyhton libraries
from cProfile import label
from turtle import title
from xmlrpc.client import ProtocolError
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


plt.style.use('fivethirtyeight')

#Get the stock symbols/ tickers in the portfolio
#FAANG
assets = ['META','AMZN','AAPL','NFLX','GOOG']

# Assign weights to the stocks.
weights = np.array([0.2,0.2,0.2,0.2,0.2])

# Get the stock starting date
stockStartDate = '2013-01-01'

# Get the stocks ending date (today)
today = datetime.today().strftime('%Y-%m-%d')
print(today)

# Create a dataframe to store the adjusted close proce of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the stock into the df
for stock in assets:
    df[stock] = web.DataReader(stock,data_source='yahoo',start = stockStartDate,end = today)['Adj Close']

# Visually show the stock / portfolio
title = 'Portfolio Adj. Close Price History'

# Get the stock
my_stocks=df

# Create and plot the graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label=c)

plt.title(title)
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Adj. Price USD($)',fontsize = 18)
plt.legend(my_stocks.columns.values,loc='upper left')
#plt.show()

# Show the daily simple return
returns = df.pct_change()
#print(returns)

# Create and show the annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
#print(cov_matrix_annual)

# Calculate the portfolio variance
port_variance = np.dot(weights.T,np.dot(cov_matrix_annual,weights))
#print(port_variance)

# Calculate the portfolio volatility aka standard deviation
port_volatility = np.sqrt(port_variance)
#print(port_volatility)

# Calculate the portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
#print(portfolioSimpleAnnualReturn)

# Show the expected annual return, volatility(risk), and  variance
percent_ret = str(round(portfolioSimpleAnnualReturn*100,2))+'%'
percent_vol = str(round(port_volatility*100,2))+'%'
percent_var = str(round(port_variance*100,2))+'%'
print('Expected annual return: ' + percent_ret)
print('Annual Risk: ' + percent_vol)
print('Annual variance: ' + percent_var)

# Portfolio Optimization

# Calculate the expected returns and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for max sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
clean_weights = ef.clean_weights()
print(clean_weights)
ef.portfolio_performance(verbose=True)

# Get the discrete allocation of each share per stock
latest_prices = get_latest_prices(df)
weights = clean_weights
da = DiscreteAllocation(weights,latest_prices,total_portfolio_value= 15000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation:', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))



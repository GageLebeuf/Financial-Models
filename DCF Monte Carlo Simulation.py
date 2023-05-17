# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:07:18 2023

@author: Gage
"""
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas_datareader as pdr
import requests
from matplotlib.ticker import FuncFormatter
import pandas as pd
import time
import statistics


# Define the stock ticker you want to create a DCF model of
ticker ="T"

symbols = [ticker]

# The growth rate will need to be manually entered.  Unfortunately we are only able to pull 5 years worth of financial statements
# which often leads to incredibly volatile and unrealistic estimates of free cash flow growth rates.  I will be using the average growth of 
# the S&P 500.


# This API returns financial information as strings which we are unable to perform operations on.  
# This function attempts to turn numeric entries into an int or float while passing on inputs
# that cause an error
def convert_numeric(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

#########################################################################################################################
 
# This equation will calculate a stocks beta as well as return the growth rate and standard deviation we will use for our monte carlo simulation
# The growth rate will need to be manually entered.  Unfortunately we are only able to pull 5 years worth of financial statements
# which often leads to incredibly volatile and unrealistic estimates of free cash flow growth rates.  I will be using the average growth of 
# the S&P 500.

def calc_beta(ticker):
    
    start_date = datetime.now() - relativedelta(years=5)
    end_date = datetime.now()
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Retrieve historical price data using yfinance
    company_data = yf.download(ticker, start=start_date_str, end=end_date_str)
    market_data = yf.download('^GSPC', start=start_date_str, end=end_date_str)
    
    # Extract the adjusted closing prices
    company_prices = company_data['Adj Close']
    market_prices = market_data['Adj Close']
    
    # Calculate the periodic returns
    company_returns = company_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    
    # Calculate the covariance and variance
    covariance = company_returns.cov(market_returns)
    market_variance = market_returns.var()
    
    # Calculate the beta coefficient
    beta = covariance / market_variance
    
    growth_rate = sum(market_returns) / 5
    
    growth_rate_sd = market_variance ** (1/2)
    
    
    return beta, growth_rate, growth_rate_sd, company_prices

#########################################################################################################################



def calc_ERP(ticker, risk_free_rate):
   
    start_date = datetime.now() - relativedelta(years=5)
    end_date = datetime.now()
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Retrieve historical price data for the market using yfinance
    market_data = yf.download('^GSPC', start=start_date_str, end=end_date_str)  # ^GSPC is the ticker for the S&P 500
    
    # Extract the adjusted closing prices
    market_prices = market_data['Adj Close']

    # Calculate the periodic returns
    market_returns = market_prices.pct_change()[1:]
    
    # Calculate the average daily market return and then multiply by the number of active trading days
    average_market_return = market_returns.mean() * 252
    
    # Calculate the equity risk premium (ERP)
    equity_risk_premium = average_market_return - risk_free_rate
    
    return equity_risk_premium

#########################################################################################################################



# For cost of debt we will be using an average of the company's debt yield via interest expense / total debt
# It's important to remember that cost of debt should be calculated using an after tax value.
# this is because interest is tax deductible which will lead to less taxs paid which will have a net effect on interest expenses.
def calc_cost_of_debt(balance_sheet, income_statement):
    cost_of_debt_proxy = []
    
    for symbol in balance_sheet:
        for key in balance_sheet[symbol]['annualReports']:
            interest_expense = income_statement[symbol]['annualReports'][key]['interestExpense']
            total_current_liabilities = balance_sheet[symbol]['annualReports'][key]['totalCurrentLiabilities']
            
            cost_of_debt_proxy.append(interest_expense / total_current_liabilities)
    
    average_cost_of_debt_proxy = statistics.mean(cost_of_debt_proxy) * (1 - 0.30)
    return average_cost_of_debt_proxy



#########################################################################################################################



# FCF = net cash from operating activities - capx +/- changes in working capital +/- other non operating cashflows
# We will be using an average of the company's previous free cash flows for our DCF's fcf value.
def calc_fcf(symbol):
    
        fcf_list = []

        annual_reports = balance_sheet[symbol]['annualReports']
        years = sorted(list(annual_reports.keys()))  # Sort the dictionary keys in ascending order
        
        for i in range(1, len(years)):
            if i == 0:
                continue
            try:
                key = years[i]
        
                net_cash = cash_flow[symbol]['annualReports'][key]['operatingCashflow']
                capx = cash_flow[symbol]['annualReports'][key]['capitalExpenditures']
        
                start_work_cap = balance_sheet[symbol]['annualReports'][years[i-1]]['totalCurrentAssets'] - balance_sheet[symbol]['annualReports'][years[i-1]]['totalCurrentLiabilities']
                end_work_cap = balance_sheet[symbol]['annualReports'][key]['totalCurrentAssets'] - balance_sheet[symbol]['annualReports'][key]['totalCurrentLiabilities']
                change_in_work_cap = end_work_cap - start_work_cap
                other_op_cashflows = (
                                cash_flow[symbol]['annualReports'][key]['cashflowFromFinancing']
                                + cash_flow[symbol]['annualReports'][key]['cashflowFromInvestment']
                                + cash_flow[symbol]['annualReports'][key]['changeInCashAndCashEquivalents']
                                + cash_flow[symbol]['annualReports'][key]['depreciationDepletionAndAmortization']
                                + cash_flow[symbol]['annualReports'][key]['changeInOperatingAssets']
                                + cash_flow[symbol]['annualReports'][key]['changeInOperatingLiabilities'])
                
                fcf = net_cash - capx + change_in_work_cap + other_op_cashflows
                fcf_list.append(fcf)  # Store fcf in the list
            
            
            
            
            
            except (TypeError, KeyError):
                continue
            
            
            fcf_avg_value = statistics.mean(fcf_list)
    
        return fcf_avg_value




#########################################################################################################################

def calc_wacc(symbol, company_prices):
     # Due to missing dates we will be taking company returns and interpolate values for dates that correspond with our financials
     
     df_prices = company_prices.to_frame()
     
     all_dates = pd.date_range(df_prices.index.min(), df_prices.index.max())
     
     df_prices = df_prices.reindex(all_dates)
     
     df_prices = df_prices.interpolate(method='linear')
     
     
     WACC_Values = []
     
     # Calculating WACC for each year.  We will take the average WACC value as well as the series standard deviation for the DCF model
     for symbol in balance_sheet:
         annual_reports = balance_sheet[symbol]['annualReports']
         years = sorted(list(annual_reports.keys()))  # Sort the dictionary keys in ascending order
         for i in range(0, len(years)):
                 
                 market_val_equity = balance_sheet[symbol]['annualReports'][years[i]]['commonStockSharesOutstanding'] * df_prices.loc[years[i], "Adj Close"]
                 
                 # We do not have access to corp debt information so we will be using interest expense / book value of debt as a proxy for market value of debt
                 market_val_debt = income_statement[symbol]['annualReports'][years[i]]['interestExpense'] / ( balance_sheet[symbol]['annualReports'][years[i]]['shortTermDebt'] + balance_sheet[symbol]['annualReports'][years[i]]['longTermDebt'])
                 
                 V = market_val_equity + market_val_debt
                 
                 Corp_Tax_Rate = 0.21
                 
                 WACC = ((market_val_equity / V)* cost_of_equity) + ((market_val_debt/V) * average_cost_of_debt_proxy * ( 1 - Corp_Tax_Rate))
                 
                 WACC_Values.append(WACC)
        

     WACC_Avg = statistics.mean(WACC_Values)

     return WACC_Avg
    

#########################################################################################################################


# Create a dictionary to hold each symbols values

income_statement = {}

balance_sheet = {}

cash_flow = {}


#########################################################################################################################


# Iterate over the list of symbols and fetch the financial income_data for each
for ticker in symbols:




####################### Income Statement Portion ######################################

    url_income = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=' + str(ticker) + '&apikey=NWSENOW68BMNGL1Y'
    r_income = requests.get(url_income)
    income_data = r_income.json()

# turn the annualReports and quarterlyReports into dictionaries from lists as well as include symbol as an into the sub dictionaries
    for key in ['annualReports', 'quarterlyReports']:
        income_data[key] = {i: {**report, 'symbol': income_data['symbol']} for i, report in enumerate(income_data[key])}
        
# Remove the symbol from the income_data dictionary
    del income_data['symbol']

# Add the individual report's dates as their keys
            
    for key in income_data:
        income_data[key] = {datetime.strptime(value['fiscalDateEnding'], "%Y-%m-%d"): value for key, value in income_data[key].items()}

# Transform all eligble values from a string into an int or float

    for key in income_data:
        for sub_key in income_data[key]:
            for third_key, values in income_data[key][sub_key].items():
                    income_data[key][sub_key][third_key] = convert_numeric(values)
                    
                    
    income_statement[ticker] = income_data
  
###################### Balance Sheet portion ############################################
    
    url_balance = 'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=' + str(ticker) + '&apikey=NWSENOW68BMNGL1Y'
    r_balance = requests.get(url_balance)
    balance_sheet_data = r_balance.json()


    for key in ['annualReports', 'quarterlyReports']:
        balance_sheet_data[key] = {i: {**report, 'symbol': balance_sheet_data['symbol']} for i, report in enumerate(balance_sheet_data[key])}
        

    del balance_sheet_data['symbol']


            
    for key in balance_sheet_data:
        balance_sheet_data[key] = {datetime.strptime(value['fiscalDateEnding'], "%Y-%m-%d"): value for key, value in balance_sheet_data[key].items()}



    for key in balance_sheet_data:
        for sub_key in balance_sheet_data[key]:
            for third_key, values in balance_sheet_data[key][sub_key].items():
                    balance_sheet_data[key][sub_key][third_key] = convert_numeric(values)

    balance_sheet[ticker] = balance_sheet_data
    
    
################## Statement of Cash Flows portion #########################################


    url_cash_flow = 'https://www.alphavantage.co/query?function=CASH_FLOW&symbol=' + str(ticker) + '&apikey=NWSENOW68BMNGL1Y'
    r_cash_flow = requests.get(url_cash_flow)
    cash_flow_data = r_cash_flow.json()



    for key in ['annualReports', 'quarterlyReports']:
        cash_flow_data[key] = {i: {**report, 'symbol': cash_flow_data['symbol']} for i, report in enumerate(cash_flow_data[key])}
        

    del cash_flow_data['symbol']


            
    for key in cash_flow_data:
        cash_flow_data[key] = {datetime.strptime(value['fiscalDateEnding'], "%Y-%m-%d"): value for key, value in cash_flow_data[key].items()}



    for key in cash_flow_data:
        for sub_key in cash_flow_data[key]:
            for third_key, values in cash_flow_data[key][sub_key].items():
                    cash_flow_data[key][sub_key][third_key] = convert_numeric(values)

    cash_flow[ticker] = cash_flow_data


# The AlphaVantage API allows 5 queries per minute for their free subscription.  
# Add a sleep timer to avoid losing data from over querying in the set time span.
    time.sleep(60)



#########################################################################################################################





# Fetch the current yield of the 10-year US Treasury bond from the U.S. Department of the Treasury
treasury_ticker = 'DGS10'

treasury_data = pdr.get_data_fred(treasury_ticker)
risk_free_rate = treasury_data.iloc[-1]['DGS10'] / 100 # Dividing by 100 to convert to decimal

# Calculate data for WACC Calculation

fcf = calc_fcf(ticker) # Free Cash Flow

beta, growth_rate, growth_rate_sd, company_prices = calc_beta(ticker)

erp = calc_ERP(ticker , risk_free_rate)

# cost of equity using the CAPM model: 
cost_of_equity = risk_free_rate + beta*erp

average_cost_of_debt_proxy = calc_cost_of_debt(balance_sheet, income_statement)


# WACC = (Equity / Total Capital) * Cost of Equity + (Debt / Total Capital) * Cost of Debt * (1 - Tax Rate)
WACC = calc_wacc(ticker, company_prices)



# The number of years for your projection
years = 10

# Monte Carlo parameters
n_simulations = 1000

# The discount rate we will use the company's WACC
discount_rate = WACC

# standard deviation of the discount rate
discount_rate_sd = 0.02  

# Placeholder for the simulation results
simulated_dcf_values = np.zeros(n_simulations)

#### ADD A TERMINAL VALUE INTO THE VALUATION.  MAKE FIRST 5 YEARS THE CALCULATED GROWTH RATE THEN CUT IT IN HALF

# Running the Monte Carlo simulation
for sim in range(n_simulations):
    # Generate random growth and discount rates for this simulation
    sim_growth_rate = np.random.normal(growth_rate, growth_rate_sd)
    sim_discount_rate = np.random.normal(discount_rate, discount_rate_sd)
    
    # Calculating the future cash flows for each year
    future_cash_flows = np.array([fcf * ((1 + sim_growth_rate) ** year) for year in range(1, years+1)])
    
    # Discounting the future cash flows
    discounted_cash_flows = np.array([cf / ((1 + sim_discount_rate) ** year) for year, cf in enumerate(future_cash_flows, start=1)])
    
    # Summing up the discounted future cash flows
    dcf_value = discounted_cash_flows.sum()
    
    # Storing the simulation result
    simulated_dcf_values[sim] = dcf_value

# Define a function to convert to millions
def millions(x, pos):
    'The two arguments are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

formatter = FuncFormatter(millions)

# Calculate the average value
average_value = np.mean(simulated_dcf_values)

# Plotting the simulation results
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(simulated_dcf_values, bins=50, density=True, alpha=0.6, color='g')
ax.axvline(average_value, color='r', linestyle='--', linewidth=2, label='Average')  # Add the average line
ax.xaxis.set_major_formatter(formatter)
ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels by 45 degrees

# Add text next to the line showing the average value
ymax = np.max(counts)
ax.text(average_value, ymax, 'Avg: {:1.1f}M'.format(average_value * 1e-6), color='r', ha='left', va='bottom')

plt.xlabel('DCF Value (in millions)')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation of DCF Value')
plt.legend()  # Display the legend with the average line
plt.show()





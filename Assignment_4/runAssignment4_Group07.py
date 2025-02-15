# Importing libraries
import pandas as pd
import numpy as np
import datetime as dt
import RM_lib as rm
from FE_Library import yearfrac
from scipy import io
import math

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Importing data
data_eurst = pd.read_csv('EUROSTOXX50_Dataset.csv', parse_dates=True, index_col=0)
data_idx = pd.read_csv('_indexes.csv')
"""print("Table 1: ", data_eurst.head(),"\n")

# Data main information
print("Table 2: ", data_eurst.describe(), "\n")"""


# Exercise 0: Exploit Variance-Covariance method for VaR and ES in linear portfolio.
# You can see here how to perform a 5y daily VaR and ES estimation via t-student parametric
# approach performed in the AnalyticalNormalMeasures function defined above

initial_date = dt.datetime(2015,2,19) # We include the previous date in order to compute a complete structure of returns
final_date = dt.datetime(2020,2,20)
riskMeasureTimeIntervalInDay = 1

#portfolio construction
ptf_0 = data_eurst[data_eurst.index <= final_date]
ptf_0 = ptf_0[ptf_0.index >= initial_date]
ptf_0 = ptf_0[["ADSGn.DE",  "ALVG.DE", "MUVGn.DE", "OREP.PA"]]
ptf_0 = ptf_0.ffill()


#logreturns
logreturns_0 = np.log(ptf_0/ptf_0.shift(1))
logreturns_0 = logreturns_0.drop(logreturns_0.index[0])


#parameters
alpha = 0.99
weights = 0.25 * np.ones((1,4))
portfolioValue = 1.5e7
nu = 4

VaR, ES = rm.AnalyticalNormalMeasures(alpha, nu, weights, portfolioValue, riskMeasureTimeIntervalInDay, logreturns_0)
print("Exercise 0 results: \n")
print(f'Daily value at risk (confidence level:99%) is: {VaR.values.ravel()[0]:.2f} Eur')
print(f'Daily expected shortfall (confidence level:99%) is: {ES.values.ravel()[0]:.2f} Eur','\n')

# Exercise 1

#portfolio construction
initial_date = dt.datetime(2014,3,18) # dates are adjusted to obtain a full dataset of returns from 20/03/14 to 20/03/19
final_date = dt.datetime(2019,3,21)
ptf_1 = data_eurst[data_eurst.index < final_date]
ptf_1 = ptf_1[ptf_1.index > initial_date]
ptf_1 = ptf_1[["TTEF.PA",  "AXAF.PA", "SASY.PA", "VOWG_p.DE"]]

ptf_1 = ptf_1.ffill()

#logreturns
logreturns_1 = np.log(ptf_1/ptf_1.shift(1))
logreturns_1 = logreturns_1.drop(logreturns_1.index[0])

# Point A_1: Daily VaR and ES computation with a 5y estimation via a Historical
# Simulation approach

#parameter
alpha_1 = 0.95
shares = np.array([25e3, 20e3, 20e3, 10e3])
riskMeasureTimeIntervalInDay = 1

ptf_1.index = pd.to_datetime(ptf_1.index)
stock_price = ptf_1.loc['2019-03-20']
ptfValue = np.dot(stock_price,shares)
weights_1 = np.multiply(stock_price, shares) / ptfValue

VaR_1A1, ES_1A1 = rm.HSMeasurements(logreturns_1, alpha_1, weights_1, ptfValue, riskMeasureTimeIntervalInDay)

print("Exercise 1: Historical Simulation")
print(f"Daily value at Risk (confidence level:95%) is: {VaR_1A1:.2f}","Eur")
print(f"Daily expected shortfall (confidence level:95%) is: {ES_1A1:.2f}", "Eur\n")

# Point A_2: Daily VaR and ES computation with a 5y estimation via Bootstrap
# approach and HS

# parameters
numberOfSamplesToBootstrap = 200

# samples
sample = rm.bootstrapStatistical(numberOfSamplesToBootstrap,logreturns_1)

VaR_1A2, ES_1A2 = rm.HSMeasurements(sample, alpha_1, weights_1, ptfValue, riskMeasureTimeIntervalInDay)

print("Exercise 1: Bootstrap")
print(f"Daily value at Risk (confidence level:95%) is: {VaR_1A2:.2f}","Eur")
print(f"Daily expected shortfall (confidence level:95%) is: {ES_1A2:.2f}", "Eur\n")

# Plausibility check on the results magnitude
VaR_p = rm.plausibilityCheck(logreturns_1, weights_1, alpha_1, ptfValue, riskMeasureTimeIntervalInDay)
print("Exercise 1_A: Plausibility Check")
print(f"Daily value at Risk (confidence level:95%) is: {VaR_p:.2f}","Eur\n")

# # Point B: Daily VaR and ES computation with a 5y estimation via a Weighted Historical
# # Simulation approach

# portfolio construction
weights_2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
ptf_2 = data_eurst[data_eurst.index < final_date]
ptf_2 = ptf_2[ptf_2.index > initial_date]
ptf_2 = ptf_2[["ADSGn.DE",  "AIR.PA", "BBVA.MC", "BMWG.DE","DTEGn.DE"]]

ptf_2 = ptf_2.ffill()

#logreturns
logreturns_2 = np.log(ptf_2/ptf_2.shift(1))
logreturns_2 = logreturns_2.drop(logreturns_2.index[0])

#parameters
riskMeasureTimeIntervalInDay = 1
portfolioValue_2 = 1
lambda_val = 0.95

VaR_2, ES_2 = rm.WHSMeasurements(logreturns_2, alpha_1, lambda_val, weights_2, portfolioValue_2,
riskMeasureTimeIntervalInDay)

print("Exercise 1: Weighted Historical Simulations")
print(f'Daily value at Risk (confidence level:95%) is: {VaR_2.item() * 100:.2f}%')
print(f'Daily expected shortfall (confidence level:95%) is: {ES_2.item() * 100:.2f}%\n')


VaR_p = rm.plausibilityCheck(logreturns_2, weights_2, alpha_1, portfolioValue_2, riskMeasureTimeIntervalInDay)
print("Exercise 1_B: Plausibility Check")
print(f'Daily value at Risk (confidence level:95%) is: {VaR_p * 100:.2f}%\n')

# Point C: 10 Days VaR and ES computation with a 5y estimation via a Gaussian parametric
# PCA approach

ticker = data_idx.iloc[0:19,1]
ticker = ticker.drop(ticker.index[3])

# We can visualize which firms are under investigation
# print(ticker)

ptf_3 = data_eurst[data_eurst.index < final_date]
ptf_3 = ptf_3[ptf_3.index > initial_date]
ptf_3 = ptf_3[ticker]

ptf_3 = ptf_3.ffill()

#logreturns
logreturns_3 = np.log(ptf_3/ptf_3.shift(1))
logreturns_3 = logreturns_3.drop(logreturns_3.index[0])

#parameters
weights_3 = 1/18 * np.ones((1,18))
portfolioValue_3 = 1

mean_firms = logreturns_3.mean()
yearlyMeanReturns = mean_firms * 256
yearlyCovariance = logreturns_3.cov() * 256

H = 10 / 256

print("Exercise 1: PCA")
for numberOfPrincipalComponents in range(1,6):

    VaR_3, ES_3 = rm.PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights_3, H, alpha_1,
    numberOfPrincipalComponents, portfolioValue_3)
    print("k = ", numberOfPrincipalComponents)
    print(f'Daily value at Risk (confidence level:95%) is: {VaR_3.item() * 100:.4f}%')
    print(f'Daily expected shortfall (confidence level:95%) is: {ES_3.item() * 100:.4f}%\n')


VaR_p = rm.plausibilityCheck(logreturns_3, weights_3.T, alpha_1, portfolioValue_3, H * 256)
print("Exercise 1_C: Plausibility Check")
print(f'Daily value at Risk (confidence level:95%) is: {VaR_p * 100:.4f}%\n')

# Exercise 2

print("Exercise 2")

# We are interested in a 2y Weighted Historical Simulation
init_date = dt.datetime(2015,1,15)
final_date = dt.datetime(2017,1,16)

# portfolio construction
ptf_4 = data_eurst[data_eurst.index <= final_date]
ptf_4 = ptf_4[ptf_4.index >= init_date]
ptf_4 = ptf_4[["BMWG.DE"]]

ptf_4 = ptf_4.ffill()
stock_price = ptf_4.loc[final_date]
ptf_4 = ptf_4.values

# logreturns computation
logreturns_4 = np.log(ptf_4[1:]/ptf_4[:-1]) # it is a column

#parameters
Notional = 1186680 # notional of BMW portfolio and number of call options
# call options are shorted -> call option data
expiry_date = dt.datetime(2017,4,18)
K = 25
sigma = 0.154 # volatility
d = 0.031 # dividend yield
r = 0.005 # fixed interest rate

Act365 = 3
timeToMaturityInYears = yearfrac(final_date,expiry_date,Act365)

riskMeasureTimeIntervalInYears = 10 / 256
NumberOfDaysPerYears = 256
alpha = 0.95
lambda_val = 0.95


shares = math.floor(Notional/stock_price)
n_calls = shares

# Full Monte Carlo
VaR_4 = rm.FullMonteCarloVaR(logreturns_4, shares, n_calls, stock_price.values, K, r, d,
sigma, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,lambda_val,NumberOfDaysPerYears)

print(f"Var (95%) via Full Monte Carlo:\n VaR: {VaR_4.item():.2f}","Eur")

# Delta Normal
VaR_5 = rm.DeltaNormalVaR(logreturns_4, shares, n_calls, stock_price.values, K, r, d,
sigma, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, lambda_val, NumberOfDaysPerYears)

print(f"Var (95%) via Delta Normal:\n VaR: {VaR_5.item():.2f}","Eur")

## EXERCISE 3: Couplet Pricing ##

print("\n")
print("Exercise 3")

#importing clean data
data = pd.read_csv('EUROSTOXX50_Dataset.csv')
#Retrieving the ISP stock for each date
ISP = np.array(data["ISP.MI"]) #2599 columns

#data
L = 0.99
sigma = 0.20
notional = 30000000
d=0
recovery = 0.40

## IMPORTING INTERBANK DATA ## (in order to retrieve the zero rate curve)
discounts = io.loadmat("discounts.mat")["discounts"]
dates = io.loadmat("dates_str.mat")["dates_str"]
discount_swap = io.loadmat("discount_swap.mat")["discountswap_cliquet"][0]
dates_swap = io.loadmat("dates_swap.mat")["couponPaymentDates"]
dates_swap = [pd.to_datetime(date) for date in dates_swap]
dates_settlement = dates[0]
dates_settlement = pd.to_datetime(dates_settlement)

# we retrieve the interest rate curve by the discount curve
r = [-np.log(discount_swap[i])/yearfrac(dates_settlement, dates_swap[i], 3) for i in range(len(dates_swap))]

# we first not consider the case of no defaulting
S = np.ones(len(dates_swap))
# here we approximate the expiry of each call to 1y
delta = np.ones(len(dates_swap))
# the idea is to price a call with expiry 1 year for each year, and then actualize this payoff at time t0 = 0
cliquet_payoff = rm.BS_CALL(L/discount_swap[0],S[0],delta[0],r[0],sigma,d)
for i in range(1,6):
    call_price = rm.BS_CALL(L*discount_swap[i-1]*np.exp(-r[i]*yearfrac(dates_swap[i-1], dates_swap[i],6))/discount_swap[i],S[i],delta[i],r[i],sigma,d)
    cliquet_payoff = np.append(cliquet_payoff,call_price)
price_cliquet = cliquet_payoff[0]
# here we compute the cliquet without considering the default scenario
for i in range(1,6):
    price_cliquet += (np.exp(-r[i-1]*yearfrac(dates_settlement, dates_swap[i-1], 6))/discount_swap[i-1])*cliquet_payoff[i]
print(f"The price of the cliquet in a non default scenario is {price_cliquet*notional:.2f}","Eur")

# now we consider the case of defaulting
survProbs = io.loadmat("survProbs.mat")["survProbs1"][0]
survProbs = survProbs[1:]

tau = [survProbs[i] + recovery * (1-survProbs[i]) for i in range(6)]
price_cliquet_def = cliquet_payoff[0]*tau[0]
for i in range(1,6):
    price_cliquet_def += tau[i]*(np.exp(-r[i-1]*yearfrac(dates_settlement,dates_swap[i-1], 6))/discount_swap[i-1])*cliquet_payoff[i]
print(f"The price of the cliquet in a default scenario is {price_cliquet_def*notional:.2f}","Eur")




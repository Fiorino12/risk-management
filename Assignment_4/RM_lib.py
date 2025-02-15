# Importing common libraries
import numpy as np

from FE_Library import yearfrac
from scipy.stats import t
from scipy.stats import norm
import math
import statistics as sta

# Auxiliar functions to go through different VaR and ES calculation techniques

def AnalyticalNormalMeasures(alpha, nu, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    """ VaR and ES parametrical computational approach
    This function estimates the value at risk and expected shortfall
    in a parametrical manner, supposing a t-Student distribution

    :param alpha: Confidence level (float)
    :param nu: Degrees of freedom (float) for the t-Student distribution
    :param weights: Array of investment weights (numpy array)
    :param portfolioValue: Total current value (float) of the portfolio
    :param riskMeasureTimeIntervalInDay: The time horizon (int) in days over which the risk is measured
    :param returns: DataFrame or 2D-array containing historical returns (numpy array) for each asset in the portfolio

    :return: VaR (float), ES (float) - The estimated value at risk and expected shortfall for the portfolio over the specified time horizon.
    """

    # Calculate the average returns for each firm
    mean_firms = returns.mean()
    # Calculate the expected portfolio returns
    mean_ = weights.dot(mean_firms) * portfolioValue
    # Calculate the portfolio standard deviation
    sigma_ = np.sqrt(weights @ returns.cov() @ weights.T) * portfolioValue
    # Calculate the Value at Risk (VaR) for the portfolio
    VaR = riskMeasureTimeIntervalInDay * mean_ + sigma_ * t.ppf(alpha, nu) * math.sqrt(riskMeasureTimeIntervalInDay)
    # Calculate the scale factor for Expected Shortfall (ES) based on the t-Student distribution
    ES_std = (nu + pow(t.ppf(alpha, nu), 2)) / (nu - 1) * (t.pdf(t.ppf(alpha, nu), nu)) / (1 - alpha)
    # Calculate the Expected Shortfall (ES) for the portfolio
    ES = (riskMeasureTimeIntervalInDay * mean_ + math.sqrt(riskMeasureTimeIntervalInDay) * sigma_ * ES_std)

    return VaR, ES

def HSMeasurements(returns, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    """
    Historical Simulation approach for VaR and ES.
    This function estimates the value at risk (VaR) and expected shortfall (ES)
    using historical simulation based on historical returns.

    :param returns: DataFrame or 2D-array containing historical returns for each asset in the portfolio.
    :param alpha: Confidence level (float) for the calculation of VaR and ES.
    :param weights: Array of investment weights (numpy array) corresponding to each asset in the portfolio.
    :param portfolioValue: Total current value (float) of the portfolio.
    :param riskMeasureTimeIntervalInDay: The time horizon (int) in days over which the risk is measured.

    :return: VaR (float), ES (float) - The estimated value at risk and expected shortfall for the portfolio over the specified time horizon.
    """

    # Calculate historical losses from historical returns
    losses = -np.dot(returns, weights) * portfolioValue
    # Sort the losses in descending order
    loss_ord = [sorted(losses, reverse=True)]

    # Calculate Value at Risk (VaR) using the historical losses
    VaR = loss_ord[0][math.floor(len(loss_ord[0]) * (1 - alpha))] * math.sqrt(riskMeasureTimeIntervalInDay)

    # Extract the losses that contribute to the Expected Shortfall (ES)
    loss_val = loss_ord[0][:math.floor(len(loss_ord[0]) * (1 - alpha))]
    # Calculate Expected Shortfall (ES) as the average of the worst losses
    ES = sta.mean(loss_val) * math.sqrt(riskMeasureTimeIntervalInDay)

    return VaR, ES

def bootstrapStatistical(numberOfSamplesToBootstrap, returns):
    """
    Bootstrap statistical method for resampling.
    This function generates a bootstrap sample from the original data set
    by sampling with replacement, which can be used for further statistical analysis.

    :param numberOfSamplesToBootstrap: The number of samples to generate for the bootstrap sample (int).
    :param returns: DataFrame or Series containing the data from which to generate the bootstrap sample.

    :return: A DataFrame or Series that represents the bootstrap sample of the original data.
    """

    # Sample from the original data with replacement to create a bootstrap sample
    samples = returns.sample(numberOfSamplesToBootstrap, replace=True)

    return samples

def WHSMeasurements(returns, alpha, lambda_val, weights, portfolioValue,
riskMeasureTimeIntervalInDay):

    """
    Weighted Historical Simulation (WHS) for VaR and ES.
    This function estimates the value at risk (VaR) and expected shortfall (ES)
    using a weighted historical simulation approach based on historical returns.

    :param returns: DataFrame or 2D-array containing historical returns for each asset in the portfolio.
    :param alpha: Confidence level (float) for the calculation of VaR and ES.
    :param lambda_val: The decay factor (float) used in the weighting scheme.
    :param weights: Array of investment weights (numpy array) corresponding to each asset in the portfolio.
    :param portfolioValue: Total current value (float) of the portfolio.
    :param riskMeasureTimeIntervalInDay: The time horizon (int) in days over which the risk is measured.

    :return: VaR (float), ES (float) - The estimated value at risk and expected shortfall for the portfolio over the specified time horizon.
    """

    # Calculate portfolio losses based on historical returns and portfolio weights
    loss_ptf = -np.dot(returns, weights.T)
    # Calculate the normalization constant for the exponential weights
    C = (1 - lambda_val) / (1 - pow(lambda_val, len(loss_ptf)))
    # Create a vector of indices from 1 to number of returns
    vett = list(range(1, len(loss_ptf) + 1))
    # Calculate the exponential weights for each return
    w_weight = [C * pow(lambda_val, len(loss_ptf) - elem) for elem in vett]
    # Pair each loss with its corresponding weight
    vect = list(zip(loss_ptf, w_weight))
    # Sort the pairs based on losses in descending order
    vect_ord = sorted(vect, key=lambda tupla: tupla[0], reverse=True)

    # Extract just the weights from the sorted pairs and calculate their cumulative sum
    values = np.array([tupla[1] for tupla in vect_ord])
    cum_sum = np.cumsum(values)
    # Find the index where the cumulative weight exceeds the confidence level threshold
    indice = np.argmax(cum_sum > 1 - alpha)
    # Determine the denominator for the ES calculation, using the cumulative weight just before exceeding the threshold
    den = cum_sum[indice - 1] if indice > 0 else 0

    # Calculate VaR based on the loss at the threshold, adjusted for portfolio value and time horizon
    VaR = vect_ord[indice][0] * portfolioValue * math.sqrt(riskMeasureTimeIntervalInDay)

    # Calculate the numerator for the ES, summing over the weighted losses up to the threshold
    num = sum([tupla[0] * tupla[1] for tupla in vect_ord[:indice]])
    # Calculate ES based on the weighted average of losses, adjusted for portfolio value and time horizon
    ES = num / den * portfolioValue * math.sqrt(riskMeasureTimeIntervalInDay)

    return VaR, ES

def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha,
numberOfPrincipalComponents, portfolioValue):

    """
    Principal Component Analysis (PCA) for VaR and ES.
    This function estimates the value at risk (VaR) and expected shortfall (ES)
    using principal component analysis based on yearly covariance and returns.

    :param yearlyCovariance: Yearly covariance matrix (numpy array) for the portfolio assets.
    :param yearlyMeanReturns: Yearly mean returns (numpy array) for the portfolio assets.
    :param weights: Investment weights (numpy array) corresponding to each asset in the portfolio.
    :param H: Investment horizon (int).
    :param alpha: Confidence level (float) for the calculation of VaR and ES.
    :param numberOfPrincipalComponents: Number of principal components (int) to consider in the analysis.
    :param portfolioValue: Total current value (float) of the portfolio.

    :return: VaR (float), ES (float) - The estimated value at risk and expected shortfall for the portfolio over the specified time horizon.
    """

    # Perform eigendecomposition on the yearly covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(yearlyCovariance)
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    # Project the yearly mean returns and weights onto the new basis formed by the eigenvectors
    mu_hat = np.dot(sorted_eigenvectors.T, yearlyMeanReturns)
    weights_hat = np.dot(sorted_eigenvectors.T, weights.T)

    # Reduce the dimensionality by taking only the leading principal components
    mu_red = np.dot(weights_hat[:numberOfPrincipalComponents].T, mu_hat[:numberOfPrincipalComponents])
    # Square the reduced weights to get the variances
    weightsq_hat = weights_hat ** 2
    # Calculate the reduced variances by considering only the leading principal components
    sigmaq_red = np.dot(weightsq_hat[:numberOfPrincipalComponents].T, sorted_eigenvalue[:numberOfPrincipalComponents])

    # Calculate VaR using the reduced mean and variance, adjusted for the portfolio value and investment horizon
    VaR = (H * mu_red + math.sqrt(H * sigmaq_red[0]) * norm.ppf(alpha)) * portfolioValue
    # Calculate ES using the reduced mean and variance, with the adjustment for the normal distribution's density
    ES = (H * mu_red + math.sqrt(H * sigmaq_red[0]) * (norm.pdf(norm.ppf(alpha)) / (1 - alpha))) * portfolioValue

    return VaR, ES

def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    """
    Plausibility check for Value at Risk (VaR) using the historical simulation method.
    This function calculates the VaR based on historical returns, incorporating
    both the distribution of returns and the correlation structure of the portfolio.

    :param returns: DataFrame or 2D-array containing historical returns for each asset in the portfolio.
    :param portfolioWeights: Array of investment weights (numpy array) corresponding to each asset in the portfolio.
    :param alpha: Confidence level (float) for the calculation of VaR.
    :param portfolioValue: Total current value (float) of the portfolio.
    :param riskMeasureTimeIntervalInDay: The time horizon (int) in days over which the risk is measured.

    :return: VaR (float) - The estimated value at risk for the portfolio over the specified time horizon.
    """

    # Calculate the correlation matrix from the returns
    corr_matrix = returns.corr()
    # Calculate the lower percentile (loss) of the returns distribution
    l_i = np.abs(np.percentile(returns, (1 - alpha) * 100))
    # Calculate the upper percentile (gain) of the returns distribution
    u_i = np.abs(np.percentile(returns, alpha * 100))

    # Calculate the semi-VaR for individual assets based on the average of the absolute upper and lower percentiles
    sVaR_i = portfolioValue * portfolioWeights * (abs(l_i) + abs(u_i)) / 2
    # Calculate the portfolio VaR considering the time horizon and correlation between assets
    VaR = math.sqrt(riskMeasureTimeIntervalInDay) * math.sqrt(np.dot(np.dot(sVaR_i.T, corr_matrix), sVaR_i))

    return VaR

def BS_CALL(S, K, T, r, sigma,d):
    """
    Black-Scholes model for European call option pricing.

    :param S: Current stock price (float).
    :param K: Strike price of the option (float).
    :param T: Time to expiration in years (float).
    :param r: Risk-free interest rate (float).
    :param sigma: Volatility of the underlying asset (float).
    :param d: Dividend yield (float).

    :return: Call option price (float).
    """
    # Calculate d1 and d2 using Black-Scholes model parameters
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Return the Black-Scholes formula for European call option price
    return S * np.exp(-d * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def BS_CALL_DELTA(S, K, T, r, sigma, d):
    """
    Delta of a European call option according to the Black-Scholes model.

    :param S: Current stock price (float).
    :param K: Strike price of the option (float).
    :param T: Time to expiration in years (float).
    :param r: Risk-free interest rate (float).
    :param sigma: Volatility of the underlying asset (float).
    :param d: Dividend yield (float).

    :return: Delta of the call option (float).
    """
    # Calculate d1 using Black-Scholes model parameters
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    # Calculate and return the delta of the option
    delta = norm.cdf(d1) * math.exp(-d * T)
    return delta

def BS_CALL_GAMMA(S, K, T, r, sigma, d):
    """
    Gamma of a European call option according to the Black-Scholes model.

    :param S: Current stock price (float).
    :param K: Strike price of the option (float).
    :param T: Time to expiration in years (float).
    :param r: Risk-free interest rate (float).
    :param sigma: Volatility of the underlying asset (float).
    :param d: Dividend yield (float).

    :return: Gamma of the call option (float).
    """
    # Calculate d1 using Black-Scholes model parameters
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    # Calculate gamma, representing the rate of change of delta with respect to the underlying stock price
    gamma = (1 / (np.sqrt(2 * math.pi)) * S * sigma * np.sqrt(T)) * np.exp(-pow(d1, 2) / 2 - d * T)
    return gamma

def FullMonteCarloVaR(logReturns, numberOfShares, numberOfCalls, stockPrice, strike, rate, dividend,
volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, lambda_val, NumberOfDaysPerYears):

    """
    Full Monte Carlo simulation for estimating Value at Risk (VaR) of a portfolio with options.

    :param logReturns: Log returns of the stock (numpy array).
    :param numberOfShares: Number of shares held in the portfolio (int).
    :param numberOfCalls: Number of call options held in the portfolio (int).
    :param stockPrice: Current stock price (float).
    :param strike: Strike price of the call option (float).
    :param rate: Risk-free rate (float).
    :param dividend: Dividend yield of the stock (float).
    :param volatility: Volatility of the stock (float).
    :param timeToMaturityInYears: Time to maturity of the call option in years (float).
    :param riskMeasureTimeIntervalInYears: The time horizon over which risk is measured, in years (float).
    :param alpha: Confidence level for the VaR estimation (float).
    :param lambda_val: Decay factor for weighting past returns in the Monte Carlo simulation (float).
    :param NumberOfDaysPerYears: Number of trading days in a year (int).

    :return: Estimated VaR (float) - The maximum expected loss under normal market conditions at the specified confidence level.
    """
    # Number of iterations for the Monte Carlo simulation
    n_iter = 100000
    # Random selection of return indices for simulation
    n_vett = np.random.choice(len(logReturns), size=n_iter, replace=True)
    # Corresponding log returns for the simulation
    log_ret = logReturns[n_vett]

    # Calculation of weights for the exponential weighting of returns
    C = (1 - lambda_val) / (1 - np.power(lambda_val, len(logReturns)))
    w_weight = C * np.power(lambda_val, np.arange(len(logReturns))[n_vett])

    # Simulation of future stock prices based on the sampled log returns
    s_delta = stockPrice * np.exp(log_ret * riskMeasureTimeIntervalInYears * NumberOfDaysPerYears)
    # Price of call options today
    Call_today = BS_CALL(stockPrice, strike, timeToMaturityInYears, rate, volatility, dividend)
    # Price of call options in the future, simulated
    Call_fut = BS_CALL(s_delta, strike, timeToMaturityInYears - riskMeasureTimeIntervalInYears, rate, volatility, dividend)

    # Loss from the call option position
    L_call = numberOfCalls * (Call_fut - Call_today)
    # Loss from the stock position
    L_bmw = - numberOfShares * (s_delta - stockPrice)

    # Total portfolio loss
    L = L_call + L_bmw

    # Pair each loss with its corresponding weight and sort in descending order of losses
    vect = list(zip(L, w_weight))
    vect_ord = sorted(vect, key=lambda tupla: tupla[0], reverse=True)

    # Calculate the cumulative sum of the weighted losses
    values = np.array([tupla[1] for tupla in vect_ord])
    cum_sum = np.cumsum(values)
    # Find the index where the cumulative sum exceeds the VaR confidence level
    indice = np.argmax(cum_sum > 1 - alpha)

    # The VaR is the loss at this index
    VaR = vect_ord[indice][0]

    return VaR

def DeltaNormalVaR(logReturns, numberOfShares, numberOfCalls, stockPrice, strike, rate, dividend,
volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, lambda_val, NumberOfDaysPerYears):

    """
    Delta-Normal VaR calculation using weighted historical simulation and Black-Scholes Delta.

    :param logReturns: Log returns of the stock (numpy array).
    :param numberOfShares: Number of shares held in the portfolio (int).
    :param numberOfCalls: Number of call options held in the portfolio (int).
    :param stockPrice: Current stock price (float).
    :param strike: Strike price of the call option (float).
    :param rate: Risk-free rate (float).
    :param dividend: Dividend yield of the stock (float).
    :param volatility: Volatility of the stock (float).
    :param timeToMaturityInYears: Time to maturity of the call option in years (float).
    :param riskMeasureTimeIntervalInYears: The time horizon over which risk is measured, in years (float).
    :param alpha: Confidence level for the VaR estimation (float).
    :param lambda_val: Decay factor for weighting past returns in the historical simulation (float).
    :param NumberOfDaysPerYears: Number of trading days in a year (int).

    :return: Estimated VaR (float) - The maximum expected loss under normal market conditions at the specified confidence level.
    """
    # Calculate weighted historical volatility
    C = (1 - lambda_val) / (1 - pow(lambda_val, len(logReturns)))
    vett = list(range(1, len(logReturns) + 1))
    w_weight = [C * pow(lambda_val, len(logReturns) - elem) for elem in vett]

    # Select ten logreturns in the past in a random manner
    n_ten =  np.random.choice(len(logReturns), size=10, replace=True)
    val = logReturns[n_ten]
    # Sum them to obtain logreturns in a 10days span
    x_delta = sum(val)
    # Compute S(t+delta)
    s_delta = stockPrice * math.exp(x_delta)

    # calculate delta
    delta_call = BS_CALL_DELTA(s_delta, strike, timeToMaturityInYears - riskMeasureTimeIntervalInYears, rate, volatility, dividend)

    # Calculate the weighted loss for the call options and the stocks in 10 days
    L_call = numberOfCalls * delta_call * s_delta * x_delta
    L_bmw = -numberOfShares * s_delta * x_delta

    # Total portfolio loss
    L = L_call + L_bmw

    # Pair each loss with its corresponding weight and sort them
    vect = list(zip(L, w_weight))
    vect_ord = sorted(vect, key=lambda tupla: tupla[0], reverse=True)

    # Cumulative weights to identify the VaR at the specified confidence level
    values = np.array([tupla[1] for tupla in vect_ord])
    cum_sum = np.cumsum(values)
    indice = np.argmax(cum_sum > 1 - alpha)

    # Calculate VaR, adjusting for the time horizon
    VaR = vect_ord[indice][0]

    return VaR

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

# define the portfolio class
class portfolio:
    # Download the data of stocks in the portfolio and calculate the necessary financials for constructing
    # the efficiency frontier and CML
    def __init__(self, stocks, start_date, end_date, rf, weight_start, weight_end, weight_step):
        self.portfolio = []
        self.weights = []
        self.returns = []
        self.portfolio_return = []
        self.portfolio_SD = []
        self.portfolio_Sharpe = []
        self.annualized_return = []
        self.rf = rf
        for stock in stocks:
            data = yf.download(stock, start=start_date, end=end_date)
            data["Return"] = np.log(data["Close"] / data["Close"].shift())
            data.dropna(inplace=True)
            self.returns.append(data["Return"])
            data_annualized_return = np.expm1(252 * data["Return"].mean())
            self.annualized_return.append(data_annualized_return)
            self.weights.append(np.arange(weight_start, weight_end, weight_step))
        self.weight = list(product(*self.weights))

    # Calculate the expected return, standard deviation and Sharpe ratio for different weightings of the stocks, then
    # construct the efficiency frontier and CML
    def get_efficient_frontier_and_CML(self):
        for weight in self.weight:
            weight = list(weight)
            if round(np.sum(weight), 4) == 1:
                Return = np.matmul(weight, np.transpose(self.annualized_return))
                SD = np.expm1(np.sqrt(252 * np.matmul(np.matmul(weight, np.cov(self.returns)), np.transpose(weight))))
                self.portfolio_return.append(Return)
                self.portfolio_SD.append(SD)
                self.portfolio_Sharpe.append((Return - self.rf)/SD)
        plt.scatter(self.portfolio_SD, self.portfolio_return, s=0.05)
        SDs = np.linspace(0, np.max(self.portfolio_SD), 1000)
        CML = self.rf + np.max(self.portfolio_Sharpe) * SDs
        plt.plot(SDs, CML, linewidth=1, c="orange")
        plt.xlim([0, np.max(self.portfolio_Sharpe)+0.2])
        plt.show()


# Ask the user to input the necessary information for constructing the efficiency frontier and CML
N = int(input("How many stocks / ETFs do you want to include in your portfolio? "))
stocks = []
for i in range(N):
    stocks.append(input("Input the ticker of the stock / ETF: "))
start_date = input("When do you want your portfolio to start? ")
end_date = input("When do you want your portfolio to end? ")
rf = float(input("What is the risk-free rate? "))
weight_start = float(input("What is the start of weightings? "))
weight_end = float(input("What is the end of weightings? "))
weight_step = float(input("What is the step of weightings? "))
Portfolio = portfolio(stocks, start_date, end_date, rf, weight_start, weight_end, weight_step)
Portfolio.get_efficient_frontier_and_CML()
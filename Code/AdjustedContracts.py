import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
import dateutil as du
import datetime as dt

rootPath = "C:/Users/dream/Desktop/4th Year/Final Project/AdjustedData/"

class AdjustedContract:
    def __init__(self, name, dates, openPrices, highPrices, lowPrices):
        self.name = name
        
        # Remove all null/nan data:
        nan_array = pd.isnull(dates) | pd.isnull(openPrices) | pd.isnull(highPrices) | pd.isnull(lowPrices)
        not_nan_array = ~ nan_array

        self.date = dates[not_nan_array]
        self.open = openPrices[not_nan_array]
        self.high = highPrices[not_nan_array]
        self.low = lowPrices[not_nan_array]

    @classmethod
    def fromFileName(cls, fileName):
        data = pd.read_csv(fileName, dtype = {"open": np.float64, "high": np.float64, "low": np.float64}, parse_dates = ["date"], na_values = ['nan'])
        return cls(fileName.replace(rootPath, ''), data["date"].to_numpy(), data["open"].to_numpy(), data["high"].to_numpy(), data["low"].to_numpy())

    @classmethod
    def fromContract(cls, contract):
        return cls((contract.name).replace("1", '').replace("2", ''), contract.date, contract.open, contract.high, contract.low)

    # Utility function to return the index of the first date with available price data previous to the givenDate:
    def nearestDate(self, givenDate):
        startDate = self.date[-1]
        endDate = self.date[0]

        indexStart = len(self.date) - 1
        indexEnd = 0

        i = int((endDate - givenDate) / (endDate - startDate) * (indexStart - indexEnd))

        while givenDate <= endDate and givenDate >= startDate:
            if self.date[i] > givenDate:
                if i == len(self.date) - 1:
                    return i
                if self.date[i + 1] <= givenDate:
                    return i + 1
                else:
                    endDate = self.date[i]
                    indexEnd = i
                    j = int((endDate - givenDate) / (endDate - startDate) * (indexStart - indexEnd)) + indexEnd

                    if i == j:
                        i += 1
                    else:
                        i = j

            else:
                if i == 0:
                    return i
                if self.date[i - 1] > givenDate:
                    return i
                else:
                    startDate = self.date[i]
                    indexStart = i

                    j = int((endDate - givenDate) / (endDate - startDate) * (indexStart - indexEnd)) + indexEnd

                    if i == j:
                        i -= 1
                    else:
                        i = j

def initialize(names):
    adjustedContracts = []

    for contractName in names:
        adjustedContract = AdjustedContract.fromFileName(rootPath + contractName + ".csv")
        adjustedContracts.append(adjustedContract)

    return adjustedContracts
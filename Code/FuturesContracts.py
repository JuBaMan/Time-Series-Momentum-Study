import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
import dateutil as du
import datetime as dt

rootPath = "C:/Users/dream/Desktop/4th Year/Final Project/Data/"

class FuturesContract:
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Last": np.float64, "Change": np.float64, "Settle": np.float64, "Volume": np.float64, "Previous Day Open Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.last = data["Last"].to_numpy()
        self.change = data["Change"].to_numpy()
        self.settle = data["Settle"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Previous Day Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 62

    # Implement the specific contract's termination rule and save the dates in chronological order in the instance list terminationDates:
    def getTerminationDates(self):
    	print("Not implemented in base class.")

    # def nearestDate(self, givenDate):
    #     i = 0
    #     for date in self.date:
    #         if date - givenDate <= np.timedelta64(0, "ns"):
    #             return i
    #         else:
    #             i += 1
    
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

    # Utility function to delete the most recent datapoints until date.
    def discardRecent(self, date):
        stopIndex = self.nearestDate(date)

        self.date = self.date[stopIndex:len(self.date)]
        self.open = self.open[stopIndex:len(self.open)]
        self.high= self.high[stopIndex:len(self.high)]
        self.low = self.low[stopIndex:len(self.low)]
        self.last = self.last[stopIndex:len(self.last)]
        self.change = self.change[stopIndex:len(self.change)]
        self.settle = self.settle[stopIndex:len(self.settle)]
        self.volume = self.volume[stopIndex:len(self.volume)]
        self.previousDay = self.previousDay[stopIndex:len(self.previousDay)]

    def discardOldest(self, date):
        stopIndex = self.nearestDate(date)

        self.date = self.date[0:stopIndex]
        self.open = self.open[0:stopIndex]
        self.high = self.high[0:stopIndex]
        self.low = self.low[0:stopIndex]
        self.last = self.last[0:stopIndex]
        self.change = self.change[0:stopIndex]
        self.settle = self.settle[0:stopIndex]
        self.volume = self.volume[0:stopIndex]
        self.previousDay = self.previousDay[0:stopIndex]

    def ratioAdjustSeries(self, contract2, timeSeries1, timeSeries2, rollDates):
        ratio = 1.0
        updateStep = 0

        contract1Index = 0
        contract2Index = 0

        for rollDate in rollDates:
            rollDate = rollDate.astype("M8[ns]")

            temp1 = self.nearestDate(rollDate)
            temp2 = contract2.nearestDate(rollDate)

            # Get the first matching date across front-month and next-month contracts in the past from the roll date:
            while (not self.date[temp1] == contract2.date[temp2] or np.isnan(timeSeries1[temp1]) or timeSeries1[temp1] == 0 or np.isnan(timeSeries2[temp2]) or timeSeries2[temp2] == 0) and (temp1 + 1) < len(timeSeries1) and (temp2 + 1) < len(timeSeries2):
                if self.date[temp1] > contract2.date[temp2]:
                    temp1 += 1
                else:
                    temp2 += 1

            # If data is missing for more than an entire rolling period of "rollTolerance", discard all following datapoints:
            if temp1 - contract1Index > self.rollTolerance or temp2 - contract2Index > self.rollTolerance:
                for i in range(updateStep, len(timeSeries1)):
                    timeSeries1[i] = np.nan
                updateStep = len(timeSeries1)
                break
            else:
                contract1Index = temp1
                contract2Index = temp2

                term = timeSeries2[contract2Index] / timeSeries1[contract1Index]
                
                for i in range(updateStep, (contract1Index + 1)):
                    if not np.isnan(timeSeries1[i]):
                        # Cleaning 0 values in between roll dates, to discard at future computation:
                        if timeSeries1[i] == 0:
                            timeSeries1[i] = np.nan
                        else:
                            timeSeries1[i] = ratio * timeSeries1[i]

                updateStep = contract1Index + 1
                ratio *= term

        # Last adjustment and cleaning aspect:
        for i in range(updateStep, len(timeSeries1)):
            if not np.isnan(timeSeries1[i]):
                if timeSeries1[i] == 0:
                    timeSeries1[i] = np.nan
                else:
                    timeSeries1[i] = ratio * timeSeries1[i]

        return timeSeries1

    # Perform ratio backadjusting for the front-month contract given the next-month unadjusted contract:
    def ratioAdjust(self, contract2):
        self.getTerminationDates()
        contract2.getTerminationDates()

        # Reverse the chronological list of terminationDates to perform backadjusting and subtract one day to obtain the optimal rolling dates:
        if self.terminationDates[0] >= contract2.terminationDates[0]:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), self.terminationDates[::-1]))
        else:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), contract2.terminationDates[::-1]))            
        
        self.open = self.ratioAdjustSeries(contract2, self.open, contract2.open, rollDates)
        self.high = self.ratioAdjustSeries(contract2, self.high, contract2.high, rollDates)
        self.low = self.ratioAdjustSeries(contract2, self.low, contract2.low, rollDates)
        self.last = self.ratioAdjustSeries(contract2, self.last, contract2.last, rollDates)

# CME: 3 business days before the 25th day of the month if it is a business day, 4 business days before otherwise:
class CrudeOilFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = np.datetime64(str(startDate) + "-25")

            if np.is_busday(threshold):
                terminationDate = np.busday_offset(threshold, -3)
            else:
                while not np.is_busday(threshold):
                    threshold -= np.timedelta64(1, 'D')

                terminationDate = np.busday_offset(threshold, -3)

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: 2 business days before the 3rd Wednesday of the month:
class AustralianDollarFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Wed")

            terminationDate = np.busday_offset(threshold, -2)

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class BritishPoundFuturesContract(AustralianDollarFuturesContract):
    pass

# CME:
class EuroFXFuturesContract(AustralianDollarFuturesContract):
	pass

# CME: 1 business day before the 15th day of the month:
class SoybeanOilFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = np.datetime64(str(startDate) + "-15")

            if np.is_busday(threshold):
                terminationDate = np.busday_offset(threshold, -1)
            else:
                terminationDate = np.busday_offset(threshold, 0, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# MCX: Last day of the month, except if it is not a business day, then it is the most recent business day. (SOURCE: https://www.mcxindia.com/docs/default-source/default-document-library/cotton-may-2020-contract-onwards.pdf?sfvrsn=25c4bf90_0)
class CottonFuturesContract(FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Close": np.float64, "Change": np.float64, "Volume": np.int32, "Open Interest": np.int32}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.last = data["Close"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 62

    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = startDate - np.timedelta64(1, "D")
            
            terminationDate = np.busday_offset(threshold, 0, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class NYHarborFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = startDate - np.timedelta64(1, "D")
            
            terminationDate = np.busday_offset(threshold, 0, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()


# CME: Trading terminates on the business day prior to the USDA announcement of the price for that contract month. (SOURCE: https://usda.library.cornell.edu/concern/publications/ms35t8620?locale=en#release-items)
class ClassIIIMilkFuturesContract(FuturesContract):
    def getTerminationDates(self):
        thresholds = [np.datetime64(pd.to_datetime("Nov 20, 2019")), np.datetime64(pd.to_datetime("Oct 23, 2019")), np.datetime64(pd.to_datetime("Sep 18, 2019")), np.datetime64(pd.to_datetime("Aug 21, 2019")), np.datetime64(pd.to_datetime("Jul 17, 2019")), np.datetime64(pd.to_datetime("Jun 19, 2019")), np.datetime64(pd.to_datetime("May 22, 2019")), np.datetime64(pd.to_datetime("Apr 17, 2019")), np.datetime64(pd.to_datetime("Mar 20, 2019")), np.datetime64(pd.to_datetime("Feb 21, 2019")), np.datetime64(pd.to_datetime("Jan 16, 2019")), np.datetime64(pd.to_datetime("Dec 19, 2018")), np.datetime64(pd.to_datetime("Nov 21, 2018")), np.datetime64(pd.to_datetime("Oct 17, 2018")), np.datetime64(pd.to_datetime("Sep 19, 2018")), np.datetime64(pd.to_datetime("Aug 22, 2018")), np.datetime64(pd.to_datetime("Jul 18, 2018")), np.datetime64(pd.to_datetime("Jun 20, 2018")), np.datetime64(pd.to_datetime("May 23, 2018")), np.datetime64(pd.to_datetime("Apr 18, 2018")), np.datetime64(pd.to_datetime("Mar 21, 2018")), np.datetime64(pd.to_datetime("Feb 22, 2018")), np.datetime64(pd.to_datetime("Jan 18, 2018")), np.datetime64(pd.to_datetime("Dec 20, 2017")), np.datetime64(pd.to_datetime("Nov 22, 2017")), np.datetime64(pd.to_datetime("Oct 18, 2017")), np.datetime64(pd.to_datetime("Sep 20, 2017")), np.datetime64(pd.to_datetime("Aug 23, 2017")), np.datetime64(pd.to_datetime("Jul 19, 2017")), np.datetime64(pd.to_datetime("Jun 21, 2017")), np.datetime64(pd.to_datetime("May 17, 2017")), np.datetime64(pd.to_datetime("Apr 19, 2017")), np.datetime64(pd.to_datetime("Mar 22, 2017")), np.datetime64(pd.to_datetime("Feb 23, 2017")), np.datetime64(pd.to_datetime("Jan 19, 2017")), np.datetime64(pd.to_datetime("Dec 21, 2016")), np.datetime64(pd.to_datetime("Nov 23, 2016")), np.datetime64(pd.to_datetime("Oct 19, 2016")), np.datetime64(pd.to_datetime("Sep 21, 2016")), np.datetime64(pd.to_datetime("Aug 17, 2016")), np.datetime64(pd.to_datetime("Jul 20, 2016")), np.datetime64(pd.to_datetime("Jun 22, 2016")), np.datetime64(pd.to_datetime("May 18, 2016")), np.datetime64(pd.to_datetime("Apr 20, 2016")), np.datetime64(pd.to_datetime("Mar 23, 2016")), np.datetime64(pd.to_datetime("Feb 18, 2016")), np.datetime64(pd.to_datetime("Jan 21, 2016")), np.datetime64(pd.to_datetime("Dec 23, 2015")), np.datetime64(pd.to_datetime("Dec 17, 2015")), np.datetime64(pd.to_datetime("Nov 18, 2015")), np.datetime64(pd.to_datetime("Oct 21, 2015")), np.datetime64(pd.to_datetime("Sep 23, 2015")), np.datetime64(pd.to_datetime("Aug 19, 2015")), np.datetime64(pd.to_datetime("Jul 22, 2015")), np.datetime64(pd.to_datetime("Jun 17, 2015")), np.datetime64(pd.to_datetime("May 20, 2015")), np.datetime64(pd.to_datetime("Apr 22, 2015")), np.datetime64(pd.to_datetime("Mar 18, 2015")), np.datetime64(pd.to_datetime("Feb 19, 2015")), np.datetime64(pd.to_datetime("Jan 22, 2015")), np.datetime64(pd.to_datetime("Dec 17, 2014")), np.datetime64(pd.to_datetime("Nov 19, 2014")), np.datetime64(pd.to_datetime("Oct 22, 2014")), np.datetime64(pd.to_datetime("Sep 17, 2014")), np.datetime64(pd.to_datetime("Aug 20, 2014")), np.datetime64(pd.to_datetime("Jul 23, 2014")), np.datetime64(pd.to_datetime("Jun 18, 2014")), np.datetime64(pd.to_datetime("May 21, 2014")), np.datetime64(pd.to_datetime("Apr 23, 2014")), np.datetime64(pd.to_datetime("Mar 19, 2014")), np.datetime64(pd.to_datetime("Feb 20, 2014")), np.datetime64(pd.to_datetime("Jan 23, 2014")), np.datetime64(pd.to_datetime("Dec 18, 2013")), np.datetime64(pd.to_datetime("Nov 20, 2013")), np.datetime64(pd.to_datetime("Oct 23, 2013")), np.datetime64(pd.to_datetime("Sep 18, 2013")), np.datetime64(pd.to_datetime("Aug 21, 2013")), np.datetime64(pd.to_datetime("Jul 17, 2013")), np.datetime64(pd.to_datetime("Jun 19, 2013")), np.datetime64(pd.to_datetime("May 22, 2013")), np.datetime64(pd.to_datetime("Apr 17, 2013")), np.datetime64(pd.to_datetime("Mar 20, 2013")), np.datetime64(pd.to_datetime("Feb 21, 2013")), np.datetime64(pd.to_datetime("Jan 16, 2013")), np.datetime64(pd.to_datetime("Dec 19, 2012")), np.datetime64(pd.to_datetime("Nov 21, 2012")), np.datetime64(pd.to_datetime("Oct 17, 2012")), np.datetime64(pd.to_datetime("Sep 19, 2012")), np.datetime64(pd.to_datetime("Aug 22, 2012")), np.datetime64(pd.to_datetime("Jul 18, 2012")), np.datetime64(pd.to_datetime("Jun 20, 2012")), np.datetime64(pd.to_datetime("May 23, 2012")), np.datetime64(pd.to_datetime("Apr 18, 2012"))]
        
        self.terminationDates = list(map(lambda x: np.busday_offset(x.astype("M8[D]"), -1), thresholds))

        self.terminationDates = self.terminationDates[::-1]

# LIFFE: 5th calendar day of the following months: January, March, June, August and November. (SOURCE: https://live.euronext.com/en/product/commodities-futures/EMA-DPAR/contract-specification)
class CornFuturesContract(FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Change": np.float64, "Settle": np.float64, "Volume": np.float64, "Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.settle = data["Settle"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 120

    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "01" or str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "11":
                self.terminationDates.append(np.busday_offset(startDate + np.timedelta64(4, "D"), 0, roll="forward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

    def ratioAdjust(self, contract2):
        self.getTerminationDates()
        contract2.getTerminationDates()

        # Reverse the chronological list to perform backadjusting and subtract one day to obtain the optimal rolling dates:
        if self.terminationDates[0] >= contract2.terminationDates[0]:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), self.terminationDates[::-1]))
        else:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), contract2.terminationDates[::-1])) 
        
        self.open = self.ratioAdjustSeries(contract2, self.open, contract2.open, rollDates)
        self.high = self.ratioAdjustSeries(contract2, self.high, contract2.high, rollDates)
        self.low = self.ratioAdjustSeries(contract2, self.low, contract2.low, rollDates)

# LIFFE: (SOURCE: https://live.euronext.com/en/product/index-futures/FCE-DPAR/contract-specification)
class CAC40FuturesContract(CornFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Fri")

            self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: Quarterly contracts (Mar, Jun, Sep, Dec) listed for 5 consecutive quarters, terminating on the 3rd Friday of the contract month.
class SP500FuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Fri")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class EminiDowFuturesContract(SP500FuturesContract):
	pass

# CME:
class SP400MidCapFuturesContract(SP500FuturesContract):
    pass

# CME: Quarterly contracts (Mar, Jun, Sep, Dec) listed for 3 consecutive quarters, terminating on the last business calendar day of the contract month. (SOURCE: https://www.cmegroup.com/trading/interest-rates/us-treasury/2-year-us-treasury-note_contract_specifications.html)
class TwoYearTNoteFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "01":
                threshold = startDate - np.timedelta64(1, "D")

                terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class TenYearTNoteFuturesContract(TwoYearTNoteFuturesContract):
    pass

# EUREX: Quarterly contracts (Mar, Jun, Sep, Dec) listed for 5 consecutive quarters, terminating on the 3rd Friday of the contract month. (SOURCE: https://www.eurexchange.com/exchange-en/products/idx/dax/DAX-Futures-139902)
class DAXFuturesContract(SP500FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Change": np.float64, "Settle": np.float64, "Volume": np.float64, "Prev. Day Open Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.settle = data["Settle"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Prev. Day Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 150

    def ratioAdjust(self, contract2):
        self.getTerminationDates()
        contract2.getTerminationDates()

        # Reverse the chronological list to perform backadjusting and subtract one day to obtain the optimal rolling dates:
        if self.terminationDates[0] >= contract2.terminationDates[0]:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), self.terminationDates[::-1]))
        else:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), contract2.terminationDates[::-1])) 
        
        self.open = self.ratioAdjustSeries(contract2, self.open, contract2.open, rollDates)
        self.high = self.ratioAdjustSeries(contract2, self.high, contract2.high, rollDates)
        self.low = self.ratioAdjustSeries(contract2, self.low, contract2.low, rollDates)

# CME:
class Nasdaq100FuturesContract(SP500FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Last": np.float64, "Change": np.float64, "Settle": np.float64, "Volume": np.float64, "Open Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.last = data["Last"].to_numpy()
        self.settle = data["Settle"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 120

# CME: 8 monthly contracts of Jan, Mar, Apr, May, Aug, Sep, Oct, Nov, last Thursday. (SOURCE: https://www.cmegroup.com/trading/agricultural/livestock/feeder-cattle_contract_specifications.html)
class FeederCattleFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "02" or str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "11" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate - np.datetime64(1, "D"), 0, roll="backward", weekmask="Thu")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: 3rd last business day.
class GoldFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            terminationDate = np.busday_offset(startDate - np.datetime64(1, "D"), -2, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class CopperFuturesContract(GoldFuturesContract):
    pass

# CME:
class NaturalGasFuturesContract(GoldFuturesContract):
    pass

# CME:
class PlatinumFuturesContract(GoldFuturesContract):
    pass

# CME: 
class SilverFuturesContract(GoldFuturesContract):
    pass

# HKEX: (SOURCE: https://www.hkex.com.hk/Products/Listed-Derivatives/Equity-Index/Hang-Seng-Index-(HSI)/Hang-Seng-Index-Futures?sc_lang=en#&product=HSI)
class HangSengIndexFuturesContract(FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Net Change": np.float64, "Volume": np.float64, "Prev. Day Open Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Prev. Day Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 62

    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = startDate - np.timedelta64(1, "D")
            
            terminationDate = np.busday_offset(threshold, -1, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

    def ratioAdjust(self, contract2):
        self.getTerminationDates()
        contract2.getTerminationDates()

        # Reverse the chronological list to perform backadjusting and subtract one day to obtain the optimal rolling dates:
        if self.terminationDates[0] >= contract2.terminationDates[0]:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), self.terminationDates[::-1]))
        else:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), contract2.terminationDates[::-1])) 
        
        self.open = self.ratioAdjustSeries(contract2, self.open, contract2.open, rollDates)
        self.high = self.ratioAdjustSeries(contract2, self.high, contract2.high, rollDates)
        self.low = self.ratioAdjustSeries(contract2, self.low, contract2.low, rollDates)

# CME: Monthly contracts of Mar, May, Jul, Sep, Dec, with trading termination on the first business day prior to the 15th day of the contract month.
class WheatFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.datetime64(str(startDate) + "-15")

                self.terminationDates.append(np.busday_offset(threshold, 0, roll="backward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class OatsFuturesContract(WheatFuturesContract):
    pass

# CME:
class ChicagoWheatFuturesContract(WheatFuturesContract):
	pass

# MGEX: March, May, July, September (New Crop), December, The business day preceding the fifteenth calendar day of that contract month (SOURCE: http://www.mgex.com/documents/SpringWheatBrochure_NewDesign.pdf)
class HardRedSpringWheatFuturesContract(WheatFuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Last": np.float64, "Volume": np.int32, "Open Interest": np.int32}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.last = data["Last"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 104

# CME: Jan, Mar, May, Jul, Sep, Nov the business day immediately preceding the 16th day in the month.
class LumberFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "01" or str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "11":
                threshold = np.datetime64(str(startDate) + "-16")

                if np.is_busday(threshold):
                    terminationDate = np.busday_offset(threshold, -1)
                else:
                    terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: Feb, Apr, Jun, Aug, Oct, Dec, last business day.
class LiveCattleFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "01" or str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "11":
                threshold = startDate - np.timedelta64(1, "D")

                terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: Mar, Jun, Sep, Dec, Trading terminates on the 3rd last business day of the contract month.
class PalladiumFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        if str(startDate).split("-")[1] == "03":
            startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "01":
                terminationDate = np.busday_offset(startDate - np.datetime64(1, "D"), -2, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# SHFE: The 15th day of the delivery month (If it is a public holiday, the Last Trading Day shall be the 1st business day after the holiday) (SOURCE: http://www.shfe.com.cn/en/products/SteelRebar/contract/9220216.html)
class SteelRebarFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = np.datetime64(str(startDate) + "-15")

            self.terminationDates.append(np.busday_offset(threshold, 0, roll="forward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# LIFFE: March, June, September, December 2nd day prior to the 3rd Wednesday of the delivery month. (SOURCE: https://www.theice.com/products/37650324/Three-Month-Euro-Swiss-Franc-Euroswiss-Futures)
class EuroswissFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Wed")

                terminationDate = np.busday_offset(threshold, -2)

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ICE: (SOURCE: https://www.theice.com/products/194/US-Dollar-Index-Futures)
class USDollarIndexFuturesContract(EuroswissFuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Open": np.float64, "High": np.float64, "Low": np.float64, "Wave": np.float64, "Change": np.float64, "Settle": np.float64, "Volume": np.float64, "Prev. Day Open Interest": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Open"].to_numpy()
        self.high = data["High"].to_numpy()
        self.low = data["Low"].to_numpy()
        self.last = data["Wave"].to_numpy() # Not scientific and not programatic.
        self.change = data["Change"].to_numpy()
        self.settle = data["Settle"].to_numpy()
        self.volume = data["Volume"].to_numpy()
        self.previousDay = data["Prev. Day Open Interest"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 150

# ODE: February, April, June, August, October, December are delivery months, and the last trading day is the 5th last business day of those months. (SOURCE: http://ode.or.jp/english/dealing.html)
class USSoybeansFuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "11" or str(startDate).split("-")[1] == "01" :
                threshold = startDate - np.timedelta64(1, "D")
            
                terminationDate = np.busday_offset(threshold, -4, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: Jan, Mar, May, Aug, Sep, Jul, Oct, Dec, business day prior to the 15th of the month.
class SoybeanMealFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "01" or str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "12":
                threshold = np.datetime64(str(startDate) + "-15")

                self.terminationDates.append(np.busday_offset(threshold, 0, roll="backward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: January (F), March (H), May (K), July (N), September (U) & November (X), business day prior to the 15th of the contract month.
class RoughRiceFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "01" or str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "11":
                threshold = np.datetime64(str(startDate) + "-15")

                if np.is_busday(threshold):
                    terminationDate = np.busday_offset(threshold, -1)
                else:
                    terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class LeanHogsFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "02" or str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "05" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "12":
                threshold = startDate

                self.terminationDates.append(np.busday_offset(threshold, 10, roll="forward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME:
class FiveYearTNoteFuturesContract(FuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "07" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "01":
                threshold = startDate - np.timedelta64(1, "D")

                terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ICE: January, March, May, July, September, November, 14th business day prior to the last business day of the month. (SOURCE: https://www.theice.com/products/30/FCOJ-A-Futures)
class OrangeJuiceFuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "02" or str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "12":
                threshold = startDate - np.timedelta64(1, "D")

                if np.is_busday(threshold):
                    terminationDate = np.busday_offset(threshold, -14)
                else:
                    terminationDate = np.busday_offset(threshold, -13, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ICE: March, May, July, September, December, 8 business days prior to the last business day of these months. (SOURCE: https://www.theice.com/products/15/Coffee-C-Futures)
class CoffeeFuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "01":
                threshold = startDate - np.timedelta64(1, "D")

                if np.is_busday(threshold):
                    terminationDate = np.busday_offset(threshold, -8)
                else:
                    terminationDate = np.busday_offset(threshold, -7, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

        if self.terminationDates[0] <= self.date[-1]:
            self.terminationDates.pop(0)

# ICE: Last business day of each month. (SOURCE: https://www.theice.com/products/219/Brent-Crude-Futures)
class BrentCrudeOilFuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 62

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            threshold = startDate - np.timedelta64(1, "D")
            
            terminationDate = np.busday_offset(threshold, 0, roll="backward")

            self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ICE: March, May, July, September, December, 11 business days prior to the last business day.
class CocoaFuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        startDate += np.timedelta64(1, 'M')
        endDate = self.date[0]

        self.rollTolerance = 150

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "04" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "08" or str(startDate).split("-")[1] == "10" or str(startDate).split("-")[1] == "01":
                threshold = startDate - np.timedelta64(1, "D")

                self.terminationDates.append(np.busday_offset(threshold, -11, roll="backward"))

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# CME: (Mar, Jun, Sep, Dec) 3rd Friday.
class Russell2000FuturesContract(USDollarIndexFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Fri")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ICE: (SOURCE: https://www.theice.com/products/38716764/FTSE-100-Index-Future)
class FTSE100FuturesContract(DAXFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        self.rollTolerance = 140

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Fri")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# EUREX: (SOURCE: https://www.eurexchange.com/exchange-en/products/int/fix/government-bonds/Euro-Bund-Futures-137298)
class BundFuturesContract(DAXFuturesContract):
    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.datetime64(str(startDate) + "-08")

                terminationDate = np.busday_offset(threshold, 0, roll="backward")

                self.terminationDates.append(terminationDate)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

# ASX: (SOURCE: https://www.asx.com.au/products/index-derivatives/asx-index-futures-contract-specifications.htm)
class AustralianPriceIndexFuturesContract(FuturesContract):
    def __init__(self, fileName):
        data = pd.read_csv(fileName, dtype = {"Previous Settlement": np.float64}, parse_dates = ["Date"])
        self.name = fileName.replace(rootPath, '')
        self.date = data["Date"].to_numpy()
        self.open = data["Previous Settlement"].to_numpy()
        self.high = data["Previous Settlement"].to_numpy()
        self.low = data["Previous Settlement"].to_numpy()
        self.terminationDates = []
        self.rollTolerance = 140

    def getTerminationDates(self):
        startDate = self.date[-1].astype('M8[M]')
        endDate = self.date[0]

        while startDate < endDate:
            startDate += np.timedelta64(1, 'M')

            if str(startDate).split("-")[1] == "03" or str(startDate).split("-")[1] == "06" or str(startDate).split("-")[1] == "09" or str(startDate).split("-")[1] == "12":
                threshold = np.busday_offset(startDate, 2, roll="forward", weekmask="Thu")

                self.terminationDates.append(threshold)

        if self.terminationDates[-1] > endDate:
            self.terminationDates.pop()

    def ratioAdjust(self, contract2):
        self.getTerminationDates()
        contract2.getTerminationDates()

        # Reverse the chronological list of terminationDates to perform backadjusting and subtract one day to obtain the optimal rolling dates:
        if self.terminationDates[0] >= contract2.terminationDates[0]:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), self.terminationDates[::-1]))
        else:
            rollDates = list(map(lambda x: x - np.timedelta64(1, "D"), contract2.terminationDates[::-1]))            
        
        self.open = self.ratioAdjustSeries(contract2, self.open, contract2.open, rollDates)
        self.high = self.ratioAdjustSeries(contract2, self.high, contract2.high, rollDates)
        self.low = self.ratioAdjustSeries(contract2, self.low, contract2.low, rollDates)

def initialize(names):
    firstMonth = []
    secondMonth = []

    for contractName in names:
        if contractName == "CL":
            CL1 = CrudeOilFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(CL1)
            CL2 = CrudeOilFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(CL2)

        if contractName == "AD":
            AD1 = AustralianDollarFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(AD1)
            AD2 = AustralianDollarFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(AD2)

        if contractName == "BO":
            BO1 = SoybeanOilFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(BO1)
            BO2 = SoybeanOilFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(BO2)

        if contractName == "BP":
            BP1 = BritishPoundFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(BP1)
            BP2 = BritishPoundFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(BP2)

        # # Unusable
        # CT1 = CottonFuturesContract("C:/Users/dream/Desktop/4th Year/Final Project/Data/CT1.csv")
        # CT2 = CottonFuturesContract("C:/Users/dream/Desktop/4th Year/Final Project/Data/CT2.csv")

        if contractName == "DA":
            DA1 = ClassIIIMilkFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(DA1)
            DA2 = ClassIIIMilkFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(DA2)

        if contractName == "DX":
            DX1 = USDollarIndexFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(DX1)
            DX2 = USDollarIndexFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(DX2)

        if contractName == "EC":
            EC1 = EuroFXFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(EC1)
            EC2 = EuroFXFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(EC2)

        if contractName == "EMA":
            EMA1 = CornFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(EMA1)
            EMA2 = CornFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(EMA2)

        if contractName == "ES":
            ES1 = SP500FuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(ES1)
            ES2 = SP500FuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(ES2)

        if contractName == "FC":
            FC1 = FeederCattleFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(FC1)
            FC2 = FeederCattleFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(FC2)

        if contractName == "FCE":
            FCE1 = CAC40FuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(FCE1)
            FCE2 = CAC40FuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(FCE2)

        if contractName == "FDAX":
            FDAX1 = DAXFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(FDAX1)
            FDAX2 = DAXFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(FDAX2)

        if contractName == "GC":
            GC1 = GoldFuturesContract(rootPath + contractName + "1.csv")
            GC1.discardOldest(np.datetime64("1980-09-05"))
            firstMonth.append(GC1)
            GC2 = GoldFuturesContract(rootPath + contractName + "2.csv")
            GC2.discardOldest(np.datetime64("1980-09-05"))
            secondMonth.append(GC2)

        if contractName == "HG":
            HG1 = CopperFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(HG1)
            HG2 = CopperFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(HG2)

        if contractName == "HO":
            HO1 = NYHarborFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(HO1)
            HO2 = NYHarborFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(HO2)

        if contractName == "HSI":
            HSI1 = HangSengIndexFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(HSI1)
            HSI2 = HangSengIndexFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(HSI2)

        if contractName == "KW":
            KW1 = WheatFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(KW1)
            KW2 = WheatFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(KW2)

        if contractName == "LB":
            LB1 = LumberFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(LB1)
            LB2 = LumberFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(LB2)

        if contractName == "LC":
            LC1 = LiveCattleFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(LC1)
            LC2 = LiveCattleFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(LC2)

        if contractName == "MD":
            MD1 = SP400MidCapFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(MD1)
            MD2 = SP400MidCapFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(MD2)

        if contractName == "MW":
            MW1 = HardRedSpringWheatFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(MW1)
            MW2 = HardRedSpringWheatFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(MW2)

        if contractName == "ND":
            ND1 = Nasdaq100FuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(ND1)
            ND2 = Nasdaq100FuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(ND2)

        if contractName == "NG":
            NG1 = NaturalGasFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(NG1)
            NG2 = NaturalGasFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(NG2)

        if contractName == "O":
            O1 = OatsFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(O1)
            O2 = OatsFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(O2)

        if contractName == "PA":
            PA1 = PalladiumFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(PA1)
            PA2 = PalladiumFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(PA2)

        if contractName == "PL":
            PL1 = PalladiumFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(PL1)
            PL2 = PalladiumFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(PL2)

        if contractName == "RB":
            RB1 = SteelRebarFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(RB1)
            RB2 = SteelRebarFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(RB2)

        if contractName == "S":
            S1 = EuroswissFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(S1)
            S2 = EuroswissFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(S2)

        if contractName == "SB":
            SB1 = USSoybeansFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(SB1)
            SB2 = USSoybeansFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(SB2)

        if contractName == "SI":
            SI1 = SilverFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(SI1)
            SI2 = SilverFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(SI2)

        if contractName == "SM":
            SM1 = SoybeanMealFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(SM1)
            SM2 = SoybeanMealFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(SM2)

        if contractName == "TU":
            TU1 = TwoYearTNoteFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(TU1)
            TU2 = TwoYearTNoteFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(TU2)

        if contractName == "TY":
            TY1 = TenYearTNoteFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(TY1)
            TY2 = TenYearTNoteFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(TY2)

        if contractName == "W":
            W1 = ChicagoWheatFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(W1)
            W2 = ChicagoWheatFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(W2)

        if contractName == "YM":
            YM1 = EminiDowFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(YM1)
            YM2 = EminiDowFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(YM2)

        if contractName == "RR":
            RR1 = RoughRiceFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(RR1)
            RR2 = RoughRiceFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(RR2)

        if contractName == "LN":
            LN1 = LeanHogsFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(LN1)
            LN2 = LeanHogsFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(LN2)

        if contractName == "FV":
            FV1 = FiveYearTNoteFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(FV1)
            FV2 = FiveYearTNoteFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(FV2)

        if contractName == "JO":
            JO1 = OrangeJuiceFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(JO1)
            JO2 = OrangeJuiceFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(JO2)

        if contractName == "KC":
            KC1 = CoffeeFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(KC1)
            KC2 = CoffeeFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(KC2)

        if contractName == "B":
            B1 = BrentCrudeOilFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(B1)
            B2 = BrentCrudeOilFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(B2)

        if contractName == "CC":
            CC1 = CocoaFuturesContract(rootPath + contractName + "1.csv")
            CC1.discardOldest(np.datetime64("1972-09-25"))
            firstMonth.append(CC1)
            CC2 = CocoaFuturesContract(rootPath + contractName + "2.csv")
            CC2.discardOldest(np.datetime64("1972-09-25"))
            secondMonth.append(CC2)

        if contractName == "TF":
            TF1 = Russell2000FuturesContract(rootPath + contractName + "1.csv")
            TF1.discardRecent(np.datetime64("2018-03-14"))
            firstMonth.append(TF1)
            TF2 = Russell2000FuturesContract(rootPath + contractName + "2.csv")
            TF2.discardRecent(np.datetime64("2018-03-14"))
            secondMonth.append(TF2)

        if contractName == "Z":
            Z1 = FTSE100FuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(Z1)
            Z2 = FTSE100FuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(Z2)

        if contractName == "FGBL":
            FGBL1 = BundFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(FGBL1)
            FGBL2 = BundFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(FGBL2)

        if contractName == "AP":
            AP1 = AustralianPriceIndexFuturesContract(rootPath + contractName + "1.csv")
            firstMonth.append(AP1)
            AP2 = AustralianPriceIndexFuturesContract(rootPath + contractName + "2.csv")
            secondMonth.append(AP2)

    return firstMonth, secondMonth
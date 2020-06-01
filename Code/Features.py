import AdjustedContracts
import Export

import numpy as np

# We use the terminology "horizon" when computations are done in calendar days, 
# and "timeScale" when computations are done in business days.

# On a horizon of 365 days, this would give the Moskowitz et al. trend estimator.
def pastReturns(contract, horizon):
    i = 0

    trend = np.zeros(len(contract.date))

    for date in contract.date:
        beginningTrade = contract.nearestDate(date - np.timedelta64(horizon, 'D'))

        if beginningTrade == None:
            break

        trend[i] = (contract.open[i] - contract.open[beginningTrade]) / contract.open[beginningTrade]
        i += 1

    return trend

def EMWA(contract, series, timeScale):
    EMWA = np.zeros(len(contract.date))

    EMWA[len(contract.date) - 1] = series[len(contract.date) - 1]

    for i in range(len(contract.date) - 2, -1, -1):
        EMWA[i] = series[i] / timeScale + (1 - 1 / timeScale) * EMWA[i + 1]

    return EMWA

def EMWSTD(contract, series, EMWA, timeScale):
    EMWVar = np.zeros(len(contract.date))

    for i in range(len(contract.date) - 2, -1, -1):
        EMWVar[i] = ((series[i] - EMWA[i + 1]) * (series[i] - EMWA[i + 1]) / timeScale + EMWVar[i + 1]) * (1 - 1 / timeScale)

    return np.sqrt(EMWVar)

def MACD(contract, shortTimeScale, longTimeScale):
    return EMWA(contract, contract.open, shortTimeScale) - EMWA(contract, contract.open, longTimeScale)

# timeSeries is mapped over the contract's dates from the most recent datapoint to the last available one:
def STD(contract, timeSeries, horizon):
    i = 0

    stop = 0

    STD = np.zeros(len(contract.date))

    for date in contract.date:
        beginningTrade = contract.nearestDate(date - np.timedelta64(horizon, 'D'))

        if beginningTrade == len(contract.date) - 1 or beginningTrade == None or beginningTrade > len(timeSeries) - 1:
            stop = i
            break

        subarray = timeSeries[i : beginningTrade + 1]

        STD[i] = np.std(subarray)

        i += 1

    return STD, stop

# The trend estimator used in Baz et al.
def MACDIndicator(contract):
    trendEstimation = np.zeros(len(contract.date))

    # 63 business day span approximated to 3 months of calendar days.
    STDprices, stop = STD(contract, contract.open, 91)

    trendEstimation[0: stop] = np.divide(MACD(contract, 8, 24)[0: stop] + MACD(contract, 16, 48)[0: stop] + MACD(contract, 32, 96)[0: stop], STDprices[0: stop])

    # 252 business days approximated to an entire calendar year.
    STDtrendEstimation, stop = STD(contract, trendEstimation[0:stop], 365)

    contract.trendEstimation2 = np.zeros(len(contract.date))
    contract.trendEstimation2[0: stop] = np.divide(trendEstimation[0:stop], STDtrendEstimation[0:stop])

def singleMACDIndicator(contract, shortTimeScale, longTimeScale):
    trendEstimation = np.zeros(len(contract.date))

    # 63 business day span approximated to 3 months of calendar days.
    STDprices, stop = STD(contract, contract.open, 91)

    trendEstimation[0: stop] = np.divide(MACD(contract, shortTimeScale, longTimeScale)[0: stop], STDprices[0: stop])

    # 252 business days approximated to an entire calendar year.
    STDtrendEstimation, stop = STD(contract, trendEstimation[0:stop], 365)

    trendEstimation2 = np.zeros(len(contract.date))
    trendEstimation2[0: stop] = np.divide(trendEstimation[0:stop], STDtrendEstimation[0:stop])

    return trendEstimation2

def computeFeatures(contract):
    pastYearReturns = pastReturns(contract, 365)
    past6MonthsReturns = pastReturns(contract, 182)
    past3MonthsReturns = pastReturns(contract, 91)
    pastMonthReturns = pastReturns(contract, 30)
    pastDayReturns = pastReturns(contract, 1)

    EMWAPastDayReturns = EMWA(contract, pastDayReturns, 60)

    # EMWSTDPastYearReturns = EMWSTD(contract, pastYearReturns, EMWAPastYearReturns, 60)
    # EMWSTDPast6MonthsReturns = EMWSTD(contract, past6MonthsReturns, EMWAPast6MonthsReturns, 60)
    # EMWSTDPast3MonthsReturns = EMWSTD(contract, past3MonthsReturns, EMWAPast3MonthsReturns, 60)
    # EMWSTDPastMonthReturns = EMWSTD(contract, pastMonthReturns, EMWAPastMonthReturns, 60)

    # Daily returns volatility:
    EXAnteVolatilityEstimate = EMWSTD(contract, pastDayReturns, EMWAPastDayReturns, 60)

    contract.pastYearReturnsNormalised = np.zeros(len(contract.date))
    contract.past6MonthsReturnsNormalised = np.zeros(len(contract.date))
    contract.past3MonthsReturnsNormalised = np.zeros(len(contract.date))
    contract.pastMonthReturnsNormalised = np.zeros(len(contract.date))
    contract.pastDayReturnsNormalised = np.zeros(len(contract.date))

    for i in range(len(contract.date) - 4, -1, -1):
        contract.pastYearReturnsNormalised[i] = pastYearReturns[i] / (EXAnteVolatilityEstimate[i + 2] * np.sqrt(252))
        contract.past6MonthsReturnsNormalised[i] = past6MonthsReturns[i] / (EXAnteVolatilityEstimate[i + 2] * np.sqrt(126))
        contract.past3MonthsReturnsNormalised[i] = past3MonthsReturns[i] / (EXAnteVolatilityEstimate[i + 2] * np.sqrt(63))
        contract.pastMonthReturnsNormalised[i] = pastMonthReturns[i] / (EXAnteVolatilityEstimate[i + 2] * np.sqrt(21))
        contract.pastDayReturnsNormalised[i] = pastDayReturns[i] / EXAnteVolatilityEstimate[i + 2]

    contract.MACD8_24 = singleMACDIndicator(contract, 8, 24)
    contract.MACD16_48 = singleMACDIndicator(contract, 16, 48)
    contract.MACD32_96 = singleMACDIndicator(contract, 32, 96)

    # Export the featurized contracts as csv files for checking purposes:
    ctrcts = []
    ctrcts.append(contract)

    Export.exportCSV(ctrcts, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/")

    return contract

def concatenateDataPoints(contract, lookBack):
    stop = 0

    for i in range(len(contract.date)):
        if contract.MACD8_24[i] == 0 and contract.MACD16_48[i] == 0 and contract.MACD32_96[i] == 0:
            stop = i
            break

    j = 0
    dataPoints = [[] for i in range(stop - lookBack)]
    targets = [[] for i in range(stop - lookBack)]

    for i in range(stop - lookBack, 0, -1):
        # Present data:
        dataPoints[j].append(contract.date[i])
        for z in range(lookBack):
            dataPoints[j].append(contract.pastYearReturnsNormalised[i+z])
            dataPoints[j].append(contract.past6MonthsReturnsNormalised[i+z])
            dataPoints[j].append(contract.past3MonthsReturnsNormalised[i+z])
            dataPoints[j].append(contract.pastMonthReturnsNormalised[i+z])
            dataPoints[j].append(contract.pastDayReturnsNormalised[i+z])
            dataPoints[j].append(contract.MACD8_24[i+z])
            dataPoints[j].append(contract.MACD16_48[i+z])
            dataPoints[j].append(contract.MACD32_96[i+z])

        # Future data:
        targets[j].append(contract.date[i - 1])
        targets[j].append(contract.pastDayReturnsNormalised[i - 1])

        j += 1

    return dataPoints, targets

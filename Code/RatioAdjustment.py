import AdjustedContracts
import FuturesContracts
import Features
import Export
import ConstructMLDataset


import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

def adjust(names):
    rootPath = "C:/Users/dream/Desktop/4th Year/Final Project/AdjustedData/"

    firstMonth, secondMonth = FuturesContracts.initialize(names)
    adjustedContracts = []

    i = 0

    for contract in firstMonth:
        contract.ratioAdjust(secondMonth[i])
        print(contract.name)
        adjustedContracts.append(AdjustedContracts.AdjustedContract.fromContract(contract))
        i += 1

    Export.exportCSV(adjustedContracts, rootPath)

    return adjustedContracts

def adjustmentStudy(names):
    firstMonth, secondMonth = FuturesContracts.initialize(names)

    adjustedContracts = adjust(names)

    pd.plotting.register_matplotlib_converters()

    for i in range(len(firstMonth)):
        plt.plot(firstMonth[i].date, firstMonth[i].open, label = "Front Month")
        plt.plot(secondMonth[i].date, secondMonth[i].open, label = "Next Month")
        plt.plot(adjustedContracts[i].date, adjustedContracts[i].open, label = "Continuous Contract")
        plt.title("Adjustment profiles on the " + adjustedContracts[i].name.replace(".csv", '') + " contract")

        leg = plt.legend();
        plt.show()

def pastReturnsStudy(names):
    adjustedContracts = []

    for name in names:
        adjustedContracts.append(AdjustedContracts.AdjustedContract.fromFileName("C:/Users/dream/Desktop/4th Year/Final Project/AdjustedData/" + name + ".csv"))

    for contract in adjustedContracts:
        plt.plot(contract.date, Features.pastReturns(contract, 365), label = "Returns")
        plt.title("Yearly return profile for the " + contract.name.replace(".csv", '') + " contract")

        leg = plt.legend();
        plt.show()


names = [
"KC",
"B",
"RR",
"JO",
"LN",
"CC",
"TF",
"Z",
"FGBL",
"FV",
"CL",
"AD",
"BO",
"BP",
"DX",
"EC",
"EMA",
"ES",
"FC",
"FCE",
"FDAX",
"GC",
"HG",
"HO",
"HSI",
"KW",
"LB",
"LC",
"MD",
"MW",
"NG",
"O",
"PA",
"PL",
"RB",
"S",
"SB",
"SI",
"SM",
"TU",
"TY",
"W",
"YM"
]

db = ConstructMLDataset.DatasetBuilder.fromContracts(names, 1, 1825)
for iteration in range(db.iterations):
    db.splitTrainingTest(0.9, iteration)
import AdjustedContracts
import Export
import Features

import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import pickle
import dateutil as du
import datetime as dt

class DatasetBuilder:
    # DatasetBuilder expects a flat representation of the contracts: DATA and TARGET along with the interval in time (days) for the splits.
    # DATA[i] holds all datapoints for contract.name = names[i], having a 2D lattice structure where len(DATA[i]) is the amount of data available for that contract and:
    #                   -> each row is a datapoint;
    #                   -> each column represents a feature, unless it is the first column, which gives the date of the recorded data.
    # DATA[i][0] is the first available datapoint for contract i, being the earliest available record. (DATA is expected to be in chronological order)
    # 
    # Also, there is a one-to-one correspondence between the indices of DATA[i] and TARGET[i], meaning a split for one, would imply a split for the other.
    def __init__(self, DATA, TARGETS, interval):
        self.DATA = DATA
        self.TARGETS = TARGETS

        # Scalars saved for consistency across multiple splitTrainingTest calls: whenever we ask for another interation's split, we get the same format.
        self.lookback = int(len(DATA[0][0]) / 8)
        self.interval = interval

        self.blueprint = [[] for i in range(len(DATA))]
        self.computeIntervals()

        self.iterations = len(self.blueprint[0])
        # Ratio between train and validation dataset, defaulted to 0.9:
        self.ratio = 0.9

    # Load the DATA and TARGET fields from previously cached files:
    @classmethod
    def fromFiles(cls, lookback, interval):
        # Load cached data:
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/nonsplitDataPoints_" + str(lookback) + ".txt", "rb") as fp:
            DATA = pickle.load(fp)

        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/nonsplitTargets_" + str(lookback) + ".txt", "rb") as fp:
            TARGETS = pickle.load(fp)

        return cls(DATA, TARGETS, interval)

    # From pipeline:
    @classmethod
    def fromContracts(cls, names, lookback, interval):
        adjustedContracts = AdjustedContracts.initialize(names)

        DATA = [[[]]]
        TARGETS = [[[]]]

        for adjustedContract in adjustedContracts:
            dataPoints, targets = Features.concatenateDataPoints(Features.computeFeatures(adjustedContract), lookback)
            DATA.append(dataPoints)
            TARGETS.append(targets)
            print(adjustedContract.name)

        DATA.pop(0)
        TARGETS.pop(0)

        Export.exportTXT(DATA, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/nonsplitDataPoints_" + str(lookback) + ".txt")

        Export.exportTXT(TARGETS, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/nonsplitTargets_" + str(lookback) + ".txt")

        return cls(DATA, TARGETS, interval)

    def getContracts(self):
        return len(self.DATA)

    # Performing a state update on the DatasetBuilder, by compiling the blueprint for our out-of-sample testing (to avoid look-ahead bias):
    #                   blueprint[i] holds all the indices for contract.name = names[i], at which a transition between training and test data occurs.
    #                   Its structure is prepending with 0s to have the same blueprint length across all contracts.
    def computeIntervals(self):
        # Artificially initialized to impossible values after visually inspecting the dataset:
        earliest_date = np.datetime64(pd.to_datetime("Nov 20, 2020"))
        latest_date = np.datetime64(pd.to_datetime("Nov 20, 1930"))

        for featurizedContract in self.DATA:
            if earliest_date > featurizedContract[0][0]:
                earliest_date = featurizedContract[0][0]
            if latest_date < featurizedContract[len(featurizedContract) - 1][0]:
                latest_date = featurizedContract[len(featurizedContract) - 1][0]

        earliest_date = earliest_date.astype('M8[D]')
        latest_date = latest_date.astype('M8[D]')

        # All dates at 5 year intervals between earliest_date and latest_date:
        thresholds = []

        while earliest_date < latest_date:
            thresholds.append(earliest_date)

            earliest_date += np.timedelta64(self.interval, 'D')

        # Building the blueprint for backtesting and retraining, done at an interval of 5 years.
        # Contracts that have less available data (more recent) are getting 0s prepended to their blueprint row until earlier contracts catch up with them on the splits.
        for ctr, featurizedContract in enumerate(self.DATA):
            for i in range(len(thresholds) - 1):
                for j, row in enumerate(featurizedContract):
                    # row[0] represents the date, appended first in the Features module in any dataPoints row.
                    if row[0] > thresholds[i + 1]:
                        self.blueprint[ctr].append(j)
                        break

    # For a specific interval in time, given by the particular iteration, we group all data int ML splits:
    def splitTrainingTest(self, ratio, iteration):
        self.ratio = ratio

        # Dummy row of 1 features used to avoid type conflicts when concatenating to an empy list later on in the function.
        # These get deleted later.
        train = [[1 for i in range(8 * self.lookback)]]
        target = [[1, 1]]
        validation = [[1 for i in range(8 * self.lookback)]]
        validation_target = [[1, 1]]
        test = [[1 for i in range(8 * self.lookback)]]
        test_target = [[1, 1]]

        # For every contract whose data is available in DATA:
        for contract_no in range(len(self.DATA)):
            # We pick the first blueprint[contract_no][training_iteration] days of each contract as train/validation data, separated by a factor of ratio:
            if self.blueprint[contract_no][iteration] != 0:
                # The delete statement is there to eliminate the date column.
                train = np.concatenate((train, np.delete(self.DATA[contract_no][0 : int(self.blueprint[contract_no][iteration] * ratio)], 0, 1)))
                validation = np.concatenate((validation, np.delete(self.DATA[contract_no][int(self.blueprint[contract_no][iteration] * ratio) : self.blueprint[contract_no][iteration]], 0, 1)))
                target = np.concatenate((target, self.TARGETS[contract_no][0 : int(self.blueprint[contract_no][iteration] * ratio)]))
                validation_target = np.concatenate((validation_target, self.TARGETS[contract_no][int(self.blueprint[contract_no][iteration] * ratio) : self.blueprint[contract_no][iteration]]))
            # The next immediate interval in the contract's self.blueprint is reserved for out-of-sample testing:
            if iteration + 1 == len(self.blueprint[contract_no]):
                test = np.concatenate((test, np.delete(self.DATA[contract_no][self.blueprint[contract_no][iteration] : len(self.DATA[contract_no])], 0, 1)))
                test_target = np.concatenate((test_target, self.TARGETS[contract_no][self.blueprint[contract_no][iteration] : len(self.DATA[contract_no])]))
            elif self.blueprint[contract_no][iteration] != self.blueprint[contract_no][iteration + 1]:
                test = np.concatenate((test, np.delete(self.DATA[contract_no][self.blueprint[contract_no][iteration] : self.blueprint[contract_no][iteration + 1]], 0, 1)))
                test_target = np.concatenate((test_target, self.TARGETS[contract_no][self.blueprint[contract_no][iteration] : self.blueprint[contract_no][iteration + 1]]))

        # Remove the hardcoded initialization row that was only used for type matching purposes:
        train = np.delete(train, 0, 0)
        test = np.delete(test, 0, 0)
        validation = np.delete(validation, 0, 0)
        target = np.delete(target, 0, 0)
        test_target = np.delete(test_target, 0, 0)
        validation_target = np.delete(validation_target, 0, 0)

        print("Iteration " + str(iteration) + ":")
        print("Train size: " + str(len(train)))
        print("Target size: " + str(len(target)))
        print("Test size: " + str(len(test)))
        print("Target size: " + str(len(test_target)))
        print("Validation size: " + str(len(validation)))
        print("Target size: " + str(len(validation_target)))
        print("\n")

        Export.exportTXT(train, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/dataPoints" + str(iteration) + "_" + str(ratio) + ".txt")

        Export.exportTXT(target, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/targets" + str(iteration) + "_" + str(ratio) + ".txt")

        Export.exportTXT(test, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/testDataPoints" + str(iteration) + "_" + str(ratio) + ".txt")

        Export.exportTXT(test_target, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/testTargets" + str(iteration) + "_" + str(ratio) + ".txt")

        Export.exportTXT(validation, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/validationDataPoints" + str(iteration) + "_" + str(ratio) + ".txt")

        Export.exportTXT(validation_target, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/validationTargets" + str(iteration) + "_" + str(ratio) + ".txt")

        return train, target, validation, validation_target, test, test_target

    def importTrainingTest(self, ratio, iteration):
        self.ratio = ratio

        if path.exists("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/"):
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/dataPoints" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                train = pickle.load(fp)
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/targets" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                target = pickle.load(fp)
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/testDataPoints" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                test = pickle.load(fp)
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/testTargets" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                test_target = pickle.load(fp)
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/validationDataPoints" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                validation = pickle.load(fp)
            with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/CachedDataSet_" + str(self.lookback) + "_" + str(self.interval) + "/validationTargets" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
                validation_target = pickle.load(fp)
        else:
            self.splitTrainingTest(ratio, iteration)

        return train, target, validation, validation_target, test, test_target


# # Uses the Feature module to compute all available datapoints and cache a flat representation of the contracts: DATA and TARGET.
# # DATA[i] holds all datapoints for contract.name = names[i], having a 2D lattice structure where len(DATA[i]) is the amount of data available for that contract and:
# #                   -> each row is a datapoint;
# #                   -> each column represents a feature, unless it is the first column, which gives the date of the recorded data.
# # DATA[i][0] is the first available datapoint for contract i, being the earliest available record.
# # 
# # Also, there is a one-to-one correspondence between the indices of DATA[i] and TARGET[i], meaning a split for one, would imply a split for the other.
# def featurizeContracts(names, lookback):
#     adjustedContracts = AdjustedContracts.initialize(names)

#     DATA = [[[]]]
#     TARGETS = [[]]

#     for adjustedContract in adjustedContracts:
#         dataPoints, targets = Features.concatenateDataPoints(Features.computeFeatures(adjustedContract), lookback)
#         DATA.append(dataPoints)
#         TARGETS.append(targets)
#         print(adjustedContract.name)

#     DATA.pop(0)
#     TARGETS.pop(0)

#     Export.exportTXT(DATA, "C:/Users/dream/Desktop/4th Year/Final Project/nonsplitDataPoints_" + str(lookback) + ".txt")

#     Export.exportTXT(TARGETS, "C:/Users/dream/Desktop/4th Year/Final Project/nonsplitTargets_" + str(lookback) + ".txt")

# def computeIntervals(interval, DATA):
#     # Artificially initialized to impossible values after visually inspecting the dataset:
#     earliest_date = np.datetime64(pd.to_datetime("Nov 20, 2020"))
#     latest_date = np.datetime64(pd.to_datetime("Nov 20, 1930"))

#     for featurizedContract in DATA:
#         if earliest_date > featurizedContract[0][0]:
#             earliest_date = featurizedContract[0][0]
#         if latest_date < featurizedContract[len(featurizedContract) - 1][0]:
#             latest_date = featurizedContract[len(featurizedContract) - 1][0]

#     earliest_date = earliest_date.astype('M8[D]')
#     latest_date = latest_date.astype('M8[D]')

#     # All dates at 5 year intervals between earliest_date and latest_date:
#     thresholds = []
#     blueprint = [[] for i in range(len(DATA))]

#     while earliest_date < latest_date:
#         thresholds.append(earliest_date)

#         earliest_date += np.timedelta64(interval, 'D')

#     for ctr, featurizedContract in enumerate(DATA):
#         for i in range(len(thresholds) - 1):
#             for j, row in enumerate(featurizedContract):
#                 if row[0] > thresholds[i + 1]:
#                     blueprint[ctr].append(j)
#                     break

#     return blueprint

# # Works with a cached version of the DATA and TARGETS lattices obtained using pickle to dump the outputs of the method above.
# def splitTrainingTest(lookback):
#     # Load cached data:
#     with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/nonsplitDataPoints_" + str(lookback) + ".txt", "rb") as fp:
#         DATA = pickle.load(fp)

#     with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/nonsplitTargets_" + str(lookback) + ".txt", "rb") as fp:
#         TARGETS = pickle.load(fp)

#     # 365 * 5 calendar days to approximate a 5 year duration.
#     blueprint = computeIntervals(1825, DATA)

#     # For all 5 year intervals:
#     for training_iteration in range(len(blueprint[0])):
#         train = [[1 for i in range(8 * lookback)]]
#         target = []
#         validation = [[1 for i in range(8 * lookback)]]
#         validation_target = []
#         test = [[1 for i in range(8 * lookback)]]
#         test_target = []

#         # For every contract whose data is available in DATA:
#         for contract_no in range(len(DATA)):
#             if blueprint[contract_no][training_iteration] != 0:
#                 # The delete statement is there to eliminate the date column:
#                 train = np.concatenate((train, np.delete(DATA[contract_no][0 : int(blueprint[contract_no][training_iteration] * 0.9)], 0, 1)))
#                 validation = np.concatenate((validation, np.delete(DATA[contract_no][int(blueprint[contract_no][training_iteration] * 0.9) : blueprint[contract_no][training_iteration]], 0, 1)))
#                 target = np.concatenate((target, TARGETS[contract_no][0 : int(blueprint[contract_no][training_iteration] * 0.9)]))
#                 validation_target = np.concatenate((validation_target, TARGETS[contract_no][int(blueprint[contract_no][training_iteration] * 0.9) : blueprint[contract_no][training_iteration]]))
#             if training_iteration + 1 == len(blueprint[contract_no]):
#                 test = np.concatenate((test, np.delete(DATA[contract_no][blueprint[contract_no][training_iteration] : len(DATA[contract_no])], 0, 1)))
#                 test_target = np.concatenate((test_target, TARGETS[contract_no][blueprint[contract_no][training_iteration] : len(DATA[contract_no])]))
#             elif blueprint[contract_no][training_iteration] != blueprint[contract_no][training_iteration + 1]:
#                 test = np.concatenate((test, np.delete(DATA[contract_no][blueprint[contract_no][training_iteration] : blueprint[contract_no][training_iteration + 1]], 0, 1)))
#                 test_target = np.concatenate((test_target, TARGETS[contract_no][blueprint[contract_no][training_iteration] : blueprint[contract_no][training_iteration + 1]]))

#         # Remove the hardcoded initialization row that was only used for type matching purposes:
#         train = np.delete(train, 0, 0)
#         test = np.delete(test, 0, 0)
#         validation = np.delete(validation, 0, 0)

#         print("Iteration " + str(training_iteration) + ":")
#         print("Train size: " + str(len(train)))
#         print("Target size: " + str(len(target)))
#         print("Test size: " + str(len(test)))
#         print("Target size: " + str(len(test_target)))
#         print("Validation size: " + str(len(validation)))
#         print("Target size: " + str(len(validation_target)))
#         print("\n")

#         Export.exportTXT(train, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/dataPoints" + str(training_iteration) + ".txt")

#         Export.exportTXT(target, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/targets" + str(training_iteration) + ".txt")

#         Export.exportTXT(test, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/testDataPoints" + str(training_iteration) + ".txt")

#         Export.exportTXT(test_target, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/testTargets" + str(training_iteration) + ".txt")

#         Export.exportTXT(validation, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/validationDataPoints" + str(training_iteration) + ".txt")

#         Export.exportTXT(validation_target, "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/validationTargets" + str(training_iteration) + ".txt")






# names = [
# "KC",
# "B",
# "RR",
# "JO",
# "LN",
# "CC",
# "TF",
# "Z",
# "FGBL",
# "FV",
# "CL",
# "AD",
# "BO",
# "BP",
# "DX",
# "EC",
# "EMA",
# "ES",
# "FC",
# "FCE",
# "FDAX",
# "GC",
# "HG",
# "HO",
# "HSI",
# "KW",
# "LB",
# "LC",
# "MD",
# "MW",
# "NG",
# "O",
# "PA",
# "PL",
# "RB",
# "S",
# "SB",
# "SI",
# "SM",
# "TU",
# "TY",
# "W",
# "YM"
# ]
# db = DatasetBuilder.fromFiles(lookback = 5, interval = 1825)
# for iteration in range(db.iterations):
#     db.splitTrainingTest(0.9, iteration)
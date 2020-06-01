import PredictionReconstruction
import ConstructMLDataset
import Export

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import statistics

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

import os.path
import pickle

class SplitManager:
    def __init__(self, lookback, interval):
        # We initialize from cached 
        self.db = ConstructMLDataset.DatasetBuilder.fromFiles(lookback, interval)

    def generateFullBatchSplit(self, ratio, iteration, seq_length):
        train, target, validation, validation_target, test, test_target = self.db.importTrainingTest(ratio, iteration)

        # There is a need to implement batching for more non-convex models.
        fullbatch_inputs = torch.from_numpy(np.array(train, dtype='float32')).cuda()
        fullbatch_targets = torch.Tensor(np.array(np.delete(target, 0, 1), dtype='float32')).cuda()

        fullbatch_valids = torch.from_numpy(np.array(validation, dtype='float32')).cuda()
        fullbatch_validtargets = torch.Tensor(np.array(np.delete(validation_target, 0, 1), dtype='float32')).cuda()

        fullbatch_tests = torch.Tensor(np.array(test, dtype='float32')).cuda()
        fullbatch_testtargets = torch.Tensor(np.array(np.delete(test_target, 0, 1), dtype='float32')).cuda()

        self.reconstructionDates = np.delete(test_target, 1, 1)

        return fullbatch_inputs, fullbatch_targets, fullbatch_valids, fullbatch_validtargets, fullbatch_tests, fullbatch_testtargets

    def minibatch(self, batch_size, fullbatch_inputs, fullbatch_targets):
        permutation = torch.randperm(fullbatch_inputs.size()[0])

        list_of_batched_inputs = []
        list_of_batched_targets = []

        for i in range(0, fullbatch_inputs.size()[0], batch_size):
            if (i + batch_size) > fullbatch_inputs.size()[0]:
                indices = permutation[i : fullbatch_inputs.size()[0]]
            else:
                indices = permutation[i : (i + batch_size)]

            list_of_batched_inputs.append(fullbatch_inputs[indices])
            list_of_batched_targets.append(fullbatch_targets[indices])

        return list_of_batched_inputs, list_of_batched_targets

    def getIterations(self):
        return self.db.iterations

class SequenceSplitManager(SplitManager):
    def __init__(self, lookback, interval):
        super().__init__(lookback, interval)

    def seqIndicesTrain(self, iteration, seq_length):
        blueprint = self.db.blueprint
        DATA = self.db.DATA

        menu = []

        offset = 0
        for contract in range(len(self.db.DATA)):
            # The only way these menu-building functions differ is through the bounds they implement for train/valid/test:
            for j in range(int(blueprint[contract][iteration] * self.db.ratio)):
                if (j+1) % seq_length == 0:
                    menu.append((offset + (j+1) - seq_length, offset + (j+1)))

            # When you are done with a contract's menu, shift to the right to the next contract:
            offset += int(blueprint[contract][iteration] * self.db.ratio)

        return menu

    def seqIndicesValid(self, iteration, seq_length):
        blueprint = self.db.blueprint
        DATA = self.db.DATA

        menu = []

        offset = 0
        for contract in range(len(DATA)):
            for j in range(blueprint[contract][iteration] - int(blueprint[contract][iteration] * self.db.ratio)):
                if (j+1) % seq_length == 0:
                    menu.append((offset + (j+1) - seq_length, offset + (j+1)))

            offset += blueprint[contract][iteration] - int(blueprint[contract][iteration] * self.db.ratio)

        return menu

    def seqIndicesTest(self, iteration, seq_length):
        blueprint = self.db.blueprint
        DATA = self.db.DATA

        menu = []

        offset = 0
        for contract in range(len(DATA)):
            if iteration + 1 != self.getIterations():
                for j in range(blueprint[contract][iteration + 1] - blueprint[contract][iteration]):
                    if (j+1) % seq_length == 0:
                        menu.append((offset + (j+1) - seq_length, offset + (j+1)))

                offset += blueprint[contract][iteration + 1] - blueprint[contract][iteration]
            else:
                for j in range(len(DATA[contract]) - blueprint[contract][iteration]):
                    if (j+1) % seq_length == 0:
                        menu.append((offset + (j+1) - seq_length, offset + (j+1)))

                offset += len(DATA[contract]) - blueprint[contract][iteration]

        return menu

    def importFullBatchSplit(self, ratio, iteration, seq_length):
        # Load cached data:
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/dataPoints_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_inputs = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/targets_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_targets = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/validationDataPoints_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_valids = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/validationTargets_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_validtargets = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/testDataPoints_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_tests = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/testTargets_" + str(iteration) + "_" + str(ratio) + ".txt", "rb") as fp:
            fullbatch_testtargets = pickle.load(fp)
        with open("C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/dates_" + str(iteration) + ".txt", "rb") as fp:
            self.reconstructionDates = pickle.load(fp)

        print("Training set consisting of " + str(fullbatch_inputs.size()) + " data points, " + str(fullbatch_targets.size()) + " training targets.")
        print("Validation set consisting of " + str(fullbatch_valids.size()) + " data points, " + str(fullbatch_validtargets.size()) + " validation targets.")
        print("Test set consisting of " + str(fullbatch_tests.size()) + " data points, " + str(fullbatch_testtargets.size()) + " test targets.")

        return fullbatch_inputs, fullbatch_targets, fullbatch_valids, fullbatch_validtargets, fullbatch_tests, fullbatch_testtargets

    def generateFullBatchSplit(self, ratio, iteration, seq_length):
        inputs, targets, valids, validtargets, tests, testtargets = super().generateFullBatchSplit(ratio, iteration, seq_length)

        menu = self.seqIndicesTrain(iteration, seq_length)
        for i in range(len(menu)):
            i1, i2 = menu[i]

            if i == 0:
                fullbatch_inputs = inputs[i1 : i2].view(seq_length, 1, self.db.lookback * 8)
                fullbatch_targets = targets[i1 : i2].view(seq_length, 1, 1)
            else:
                fullbatch_inputs = torch.cat((fullbatch_inputs, inputs[i1 : i2].view(seq_length, 1, self.db.lookback * 8)), 1)
                fullbatch_targets = torch.cat((fullbatch_targets, targets[i1 : i2].view(seq_length, 1, 1)), 1)

        # Batch data:
        # inputs = torch.split(fullbatch_inputs, batch_size, 1)
        # targets = torch.split(fullbatch_targets, batch_size, 1)

        print("Training set consisting of " + str(fullbatch_inputs.size()) + " data points, " + str(fullbatch_targets.size()) + " training targets.")

        menu = self.seqIndicesValid(iteration, seq_length)
        for i in range(len(menu)):
            i1, i2 = menu[i]

            if i == 0:
                fullbatch_valids = valids[i1 : i2].view(seq_length, 1, self.db.lookback * 8)
                fullbatch_validtargets = validtargets[i1 : i2].view(seq_length, 1, 1)
            else:
                fullbatch_valids = torch.cat((fullbatch_valids, valids[i1 : i2].view(seq_length, 1, self.db.lookback * 8)), 1)
                fullbatch_validtargets = torch.cat((fullbatch_validtargets, validtargets[i1 : i2].view(seq_length, 1, 1)), 1)

        print("Validation set consisting of " + str(fullbatch_valids.size()) + " data points, " + str(fullbatch_validtargets.size()) + " validation targets.")

        menu = self.seqIndicesTest(iteration, seq_length)
        for i in range(len(menu)):
            i1, i2 = menu[i]

            if i == 0:
                fullbatch_tests = tests[i1 : i2].view(seq_length, 1, self.db.lookback * 8)
                fullbatch_testtargets = testtargets[i1 : i2].view(seq_length, 1, 1)
                sequencedReconstructionDates = self.reconstructionDates[i1 : i2].reshape(seq_length, 1, 1)
            else:
                fullbatch_tests = torch.cat((fullbatch_tests, tests[i1 : i2].view(seq_length, 1, self.db.lookback * 8)), 1)
                fullbatch_testtargets = torch.cat((fullbatch_testtargets, testtargets[i1 : i2].view(seq_length, 1, 1)), 1)
                sequencedReconstructionDates = np.concatenate((sequencedReconstructionDates, self.reconstructionDates[i1 : i2].reshape(seq_length, 1, 1)), 1)

        self.reconstructionDates = sequencedReconstructionDates
        Export.exportTXT(self.reconstructionDates, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/dates_" + str(iteration) + ".txt")

        print("Test set consisting of " + str(fullbatch_tests.size()) + " data points, " + str(fullbatch_testtargets.size()) + " test targets.")

        Export.exportTXT(fullbatch_inputs, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/dataPoints_" + str(iteration) + "_" + str(ratio) + ".txt")
        Export.exportTXT(fullbatch_targets, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/targets_" + str(iteration) + "_" + str(ratio) + ".txt")
        Export.exportTXT(fullbatch_valids, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/validationDataPoints_" + str(iteration) + "_" + str(ratio) + ".txt")
        Export.exportTXT(fullbatch_validtargets, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/validationTargets_" + str(iteration) + "_" + str(ratio) + ".txt")
        Export.exportTXT(fullbatch_tests, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/testDataPoints_" + str(iteration) + "_" + str(ratio) + ".txt")
        Export.exportTXT(fullbatch_testtargets, "C:/Users/dream/Desktop/4th Year/Final Project/DATA BACKUP/MLDataSet/SeqDataSet_" + str(self.db.interval) + "_" + str(seq_length) + "/testTargets_" + str(iteration) + "_" + str(ratio) + ".txt")

        return fullbatch_inputs, fullbatch_targets, fullbatch_valids, fullbatch_validtargets, fullbatch_tests, fullbatch_testtargets

    def minibatch(self, batch_size, fullbatch_inputs, fullbatch_targets):
        permutation = torch.randperm(fullbatch_inputs.size()[1])

        list_of_batched_inputs = []
        list_of_batched_targets = []

        for i in range(0, fullbatch_inputs.size()[1], batch_size):
            if (i + batch_size) > fullbatch_inputs.size()[1]:
                indices = permutation[i : fullbatch_inputs.size()[1]]
            else:
                indices = permutation[i : (i + batch_size)]

            indices = permutation[i : i + batch_size]

            list_of_batched_inputs.append(fullbatch_inputs[:, indices])
            list_of_batched_targets.append(fullbatch_targets[:, indices])

        return list_of_batched_inputs, list_of_batched_targets

    def reconstruct(self, predictions):
        # dates = self.reconstructionDates.flatten()

        # # Contains groupings of indices in the flat array of predictions/returns that would correspond to the same date.
        # menu = []
        # # Summary of progress
        # found = []
        # j = 0

        # result = []

        # while len(found) < len(dates):
        #     date = dates[j]
        #     sameDay = []
            
        #     for i in range(j, len(dates), 1):
        #         if i in found:
        #             if i + 1 < len(dates) and sameDay == []:
        #                 date = dates[i + 1]
        #             continue

        #         if dates[i] == date:
        #             found.append(i)
        #             sameDay.append(i)

        #     menu.append(sameDay)
        #     j = j + 1

        # predictions = predictions.flatten()

        # for j in range(len(menu)):
        #     sameDayPredictions = [predictions[i] for i in menu[j]]

        #     result.append([sum(sameDayPredictions) / len(sameDayPredictions), dates[menu[j][0]]])

        # return result
         
        dates = self.reconstructionDates.flatten()
        predictions = predictions.flatten()

        result = []
        
        while len(dates) != 0:
            date = dates[0]
            stratReturn = 0
            dayIndex = []

            for i in range(len(dates)):
                if dates[i] == date:
                    stratReturn = stratReturn + predictions[i]
                    dayIndex.append(i)

            result.append([date, stratReturn / len(dayIndex)])

            dates = np.delete(dates, dayIndex)

        return np.array(sorted(result, key = lambda x: x[0]))


class LSTMPredictor(nn.Module):
    def __init__(self, lookback, interval, loss_fn):   
        # Model construction:
        # # # # # # # # # # #
        super(LSTMPredictor, self).__init__()

        # Peripherals:
        # # # # # # # # # # #
        self.splitManager = SequenceSplitManager(lookback, interval)
        self.loss_fn = loss_fn

    def loadDataset(self, ratio, iteration, seq_length):
        self.fullbatch_inputs, self.fullbatch_targets, self.fullbatch_valids, self.fullbatch_validtargets, self.fullbatch_tests, self.fullbatch_testtargets = self.splitManager.importFullBatchSplit(ratio, iteration, seq_length)

    def forward(self, sequences):
        hidden_states, _ = self.lstm(sequences)
        returns = torch.tanh(self.hidden2prediction(hidden_states))
        return returns

    def resetModel(self, learningRate, hidden_dim):
        self.hidden_dim = hidden_dim

        # The LSTM takes sequences of length seq_len = 63 as inputs, with every input element of the sequence having 8 features
        # (concatenate data with lookback = 1), and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(8, hidden_dim).cuda()

        # The linear layer that maps from hidden state space to actual trend estimators or position sizes.
        self.hidden2prediction = nn.Linear(hidden_dim, 1).cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr = learningRate)

    def fit(self, num_epochs, batch_size):
        validLoss = []

        list_of_batches_inputs, list_of_batches_targets = self.splitManager.minibatch(batch_size, self.fullbatch_inputs, self.fullbatch_targets)

        for epoch in range(num_epochs):     
            for batch in range(len(list_of_batches_inputs)):
                # Generate predictions
                pred = self(list_of_batches_inputs[batch])
                loss = self.loss_fn(pred, list_of_batches_targets[batch])
                
                # Perform gradient descent
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
        
            print('Training Sharpe: ', sharpe(self(self.fullbatch_inputs), self.fullbatch_targets))
            print('Training returns: ', - avgReturnLoss(self(self.fullbatch_inputs), self.fullbatch_targets))
            print('Validation Sharpe: ', sharpe(self(self.fullbatch_valids), self.fullbatch_validtargets))
            print('Validation returns: ', - avgReturnLoss(self(self.fullbatch_valids), self.fullbatch_validtargets))
            validLoss.append(float(self.loss_fn(self(self.fullbatch_valids), self.fullbatch_validtargets).to(torch.device("cpu"))))

            # Stop if no improvement in 25 epochs:
            if len(validLoss) > 25:
                validLoss.pop(0)
                if statistics.mean(validLoss[0:24]) < validLoss[24]:
                    break
        
        # backtestReturns = self(self.fullbatch_tests) * self.fullbatch_testtargets
        # perDateReturns = self.splitManager.reconstruct(backtestReturns)
        # TSMOM = np.delete(perDateReturns, 0, 1)
        # DATES = np.delete(perDateReturns, 1, 1)
        
        # plt.plot(DATES, np.cumsum(TSMOM))
        # plt.show()

        # TSMOM = torch.from_numpy(np.array(TSMOM, dtype='float32'))
        # vol = TSMOM.std()
        # som = torch.mean(TSMOM)
        
        # print('Test percentage of positive bets: ', posBets(TSMOM) * 100)
        # print('Test volatility: ', vol)
        # print('Test Sharpe: ', som / vol)

        print('Test expected return: ', - avgReturnLoss(self(self.fullbatch_tests), self.fullbatch_testtargets))
        print('Test Sharpe ratio: ', sharpe(self(self.fullbatch_tests), self.fullbatch_testtargets))

        print('\n')

# class TrainingModule:
#     def __init__(self, lookback, interval, model, optim, loss_fn):
#         self.splitManager = SequenceSplitManager(lookback, interval)

#         self.model = model
#         self.optim = optim
#         self.loss_fn = loss_fn

#     def fit(self, num_epochs, ratio, iteration, seq_length, batch_size):
#         validLoss = []

#         fullbatch_inputs, fullbatch_targets, fullbatch_valids, fullbatch_validtargets, fullbatch_tests, fullbatch_testtargets = self.splitManager.generateFullBatchSplit(ratio, iteration, seq_length)

#         list_of_batches_inputs, list_of_batches_targets = self.splitManager.minibatch(batch_size, fullbatch_inputs, fullbatch_targets)

#         for epoch in range(num_epochs):     
#             for batch in range(len(list_of_batches_inputs)):
#                 # Generate predictions
#                 pred = self.model(list_of_batches_inputs[batch])
#                 loss = self.loss_fn(pred, list_of_batches_targets[batch])
                
#                 # Perform gradient descent
#                 loss.backward()
#                 self.optim.step()
#                 self.optim.zero_grad()
        
#             print('Training loss: ', self.loss_fn(self.model(fullbatch_inputs), fullbatch_targets))
#             print('Validation loss: ', self.loss_fn(self.model(fullbatch_valids), fullbatch_validtargets))
#             validLoss.append(float(self.loss_fn(self.model(fullbatch_valids), fullbatch_validtargets).to(torch.device("cpu"))))

#             # Stop if no improvement in 25 epochs:
#             if len(validLoss) > 25:
#                 validLoss.pop(0)
#                 if statistics.mean(validLoss[0:24]) < validLoss[24]:
#                     break
#         print(self.model(fullbatch_tests))
#         print('Test expected return: ', avgReturnLoss(self.model(fullbatch_tests), fullbatch_testtargets))
#         print('Test percentage of positive bets: ', posBets(self.model(fullbatch_tests), fullbatch_testtargets)) * 100
#         print('\n')


def avgReturnLoss(positionSizes, actualReturns):
    return - torch.mean(positionSizes * actualReturns)

def posBets(returns):
    return (returns >= 0).sum().to(dtype = torch.float) / returns.numel()

def sharpeLoss(positionSizes, actualReturns):
    returns = positionSizes * actualReturns

    ret = torch.sum(returns)
    
    return - (ret / (torch.sum(returns * returns) - ret * ret))

def sharpe(positionSizes, actualReturns):
    returns = positionSizes * actualReturns

    ret = torch.sum(returns)

    return (ret /  torch.sqrt(torch.sum(returns * returns) * returns.numel() - ret * ret))




# Grid Search hyperparameters:
# dimension_space = [5, 10, 20, 40, 80]
# learningRate_space = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# batch_space = [32, 64, 128, 256, 512]

# NoDropoutLSTM = LSTMPredictor(lookback = 1, interval = 1825, loss_fn = nn.MSELoss())
# NoDropoutLSTM.loadDataset(ratio = 0.9, iteration = 10, seq_length = 63)

# for hidden_dim in dimension_space:
#     for learningRate in learningRate_space:
#         for batch_size in batch_space:
#             print("For hidden dimensions " + str(hidden_dim) + " learning rate " + str(learningRate) + " and batch size " + str(batch_size) + " and MSE loss:")
            
# NoDropoutLSTM.resetModel(learningRate = 0.05, hidden_dim = 40)
# NoDropoutLSTM.fit(num_epochs = 100, batch_size = 256)






# for iteration in range(11):
#     _, _, _, _, _, _ = NoDropoutLSTM.splitManager.generateFullBatchSplit(0.9, iteration, 63)
# print(NoDropoutLSTM.splitManager.reconstruct(targets))
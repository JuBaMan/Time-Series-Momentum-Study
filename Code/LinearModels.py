import PredictionReconstruction
import Export

import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import statistics


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt, inputs, targets, valids, validtargets, tests, testtargets):
    validLoss = []

    for epoch in range(num_epochs):     
        # Generate predictions
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        
        # Perform gradient descent
        loss.backward()
        opt.step()
        opt.zero_grad()
    
        print('Training loss: ', loss_fn(model(inputs), targets))
        print('Validation loss: ', loss_fn(model(valids), validtargets))
        validLoss.append(float(loss_fn(model(valids), validtargets).to(torch.device("cpu"))))

        # Stop if no improvement in 25 epochs:
        if len(validLoss) > 25:
            validLoss.pop(0)
            if statistics.mean(validLoss[0:24]) < validLoss[24]:
                break

    print("Test loss: ", loss_fn(model(tests), testtargets))
    print("\n")


def initializeData(iteration, lookback):
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/dataPoints" + str(iteration) + ".txt", "rb") as fp:
        train = pickle.load(fp)
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/targets" + str(iteration) + ".txt", "rb") as fp:
        target = pickle.load(fp)
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/testDataPoints" + str(iteration) + ".txt", "rb") as fp:
        test = pickle.load(fp)
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/testTargets" + str(iteration) + ".txt", "rb") as fp:
        testtarget = pickle.load(fp)
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/validationDataPoints" + str(iteration) + ".txt", "rb") as fp:
        validation = pickle.load(fp)
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/CachedDataSet_" + str(lookback) + "/validationTargets" + str(iteration) + ".txt", "rb") as fp:
        validationtarget = pickle.load(fp)

    # There is a need to implement batching for more non-convex models.
    inputs = torch.from_numpy(np.array(train, dtype='float32')).cuda()
    targets = torch.Tensor(target).unsqueeze(1).cuda()

    valids = torch.from_numpy(np.array(validation, dtype='float32')).cuda()
    validtargets = torch.Tensor(validationtarget).unsqueeze(1).cuda()

    tests = torch.Tensor(np.array(test, dtype='float32')).cuda()
    testtargets = torch.Tensor(testtarget).unsqueeze(1).cuda()

    return inputs, targets, valids, validtargets, tests, testtargets

def avgReturnLoss(pred, targets):
    return - torch.mean(pred * targets)

def sharpeLoss(pred, targets):
    returns = torch.sum(pred * targets)

    return - (returns / (torch.sum(pred * targets * pred * targets) - returns * returns))

def linearMSE():
    # Define model
    model = (torch.nn.Linear(40, 1)).cuda()

    # Define loss function
    loss_fn = avgReturnLoss

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, opt

def mainTrainingResults():
    for iteration in range(11):
        inputs, targets, valids, validtargets, tests, testtargets = initializeData(iteration, 5)

        model, loss_fn, opt = linearMSE()

        fit(100, model, loss_fn, opt, inputs, targets, valids, validtargets, tests, testtargets)

        # Export the predictions.
        Export.exportTXT(model(tests).to(torch.device("cpu")).squeeze(1), "C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/Predictions/pred" + str(iteration) + ".txt")

        # Compute performance metrics on the results

mainTrainingResults()
# contracts = PredictionReconstruction.reconstruct()



# int(loss_fn(model(torch.Tensor(test1).cuda()), torch.Tensor(targettest1).unsqueeze(1).cuda()))

# prediction = model(test).to(torch.device("cpu")).squeeze(1)[0:1260]
# prediction = prediction.detach().numpy()
# real = (torch.Tensor(targettest1))[0:1260]

# plt.plot(prediction, label = 'Pred')
# plt.plot(real, 'r-', alpha=0.6, label = 'Real')
# leg = plt.legend();
# plt.show()

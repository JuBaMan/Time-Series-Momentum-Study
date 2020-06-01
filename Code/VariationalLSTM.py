import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import statistics

import LSTMModels

class MyLSTM(nn.Module):
    def __init__(self, lookback, interval, loss_fn):   
        super(MyLSTM, self).__init__()

        self.splitManager = LSTMModels.SequenceSplitManager(lookback, interval)
        self.loss_fn = loss_fn

    def loadDataset(self, ratio, iteration, seq_length):
        self.fullbatch_inputs, self.fullbatch_targets, self.fullbatch_valids, self.fullbatch_validtargets, self.fullbatch_tests, self.fullbatch_testtargets = self.splitManager.importFullBatchSplit(ratio, iteration, seq_length)

    def forward(self, sequences, seqNo):
        hidden_states, _ = self.lstmDropout(sequences, (torch.zeros(seqNo, self.hidden_dim).cuda(), torch.zeros(seqNo, self.hidden_dim).cuda()))
        returns = torch.tanh(self.hidden2prediction(hidden_states))
        return returns

    def resetModel(self, learningRate, hidden_dim, dropout):
        self.hidden_dim = hidden_dim

        # The LSTM takes sequences of length seq_len = 63 as inputs, with every input element of the sequence having 8 features
        # (concatenate data with lookback = 1), and outputs hidden states with dimensionality hidden_dim.
        self.lstmDropout = LSTMDropout(8, hidden_dim, dropout).cuda()

        # The linear layer that maps from hidden state space to actual trend estimators or position sizes.
        self.hidden2prediction = nn.Linear(hidden_dim, 1).cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr = learningRate)

    def fit(self, num_epochs, batch_size):
        validLoss = []

        list_of_batches_inputs, list_of_batches_targets = self.splitManager.minibatch(batch_size, self.fullbatch_inputs, self.fullbatch_targets)

        for epoch in range(num_epochs):     
            for batch in range(len(list_of_batches_inputs)):
                # Reset masks for the number of sequences in this batch:
                self.lstmDropout.resetMasks(list_of_batches_inputs[batch].shape[1])

                # Generate predictions
                pred = self(list_of_batches_inputs[batch], list_of_batches_inputs[batch].shape[1])

                loss = self.loss_fn(pred, list_of_batches_targets[batch])
                
                # Perform gradient descent
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
        
            self.lstmDropout.setTesting()
            print('Training Sharpe: ', LSTMModels.sharpe(self(self.fullbatch_inputs, self.fullbatch_inputs.shape[1]), self.fullbatch_targets))
            print('Training returns: ', - LSTMModels.avgReturnLoss(self(self.fullbatch_inputs, self.fullbatch_inputs.shape[1]), self.fullbatch_targets))
            print('Validation Sharpe: ', LSTMModels.sharpe(self(self.fullbatch_valids, self.fullbatch_valids.shape[1]), self.fullbatch_validtargets))
            print('Validation returns: ', - LSTMModels.avgReturnLoss(self(self.fullbatch_valids, self.fullbatch_valids.shape[1]), self.fullbatch_validtargets))
            validLoss.append(float(self.loss_fn(self(self.fullbatch_valids, self.fullbatch_valids.shape[1]), self.fullbatch_validtargets).to(torch.device("cpu"))))

            # Stop if no improvement in 25 epochs:
            if len(validLoss) > 25:
                validLoss.pop(0)
                if statistics.mean(validLoss[0:24]) < validLoss[24]:
                    break

        print('Test expected return: ', - LSTMModels.avgReturnLoss(self(self.fullbatch_tests, self.fullbatch_tests.shape[1]), self.fullbatch_testtargets))
        print('Test Sharpe ratio: ', LSTMModels.sharpe(self(self.fullbatch_tests, self.fullbatch_tests.shape[1]), self.fullbatch_testtargets))

        print('\n')

# My custom written LSTM module.
class LSTMDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0):
        super().__init__()
        
        self.input_size = input_size # 8 features
        self.hidden_size = hidden_size # customizable
        self.dropout = dropout # customizable



        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size))
        self.V_f = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size))
        self.V_i = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size))
        self.V_c = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        self.V_o = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_i = nn.Parameter(torch.randn(hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))

        self.forgetMask = StaticDropout(dropout, hidden_size)
        self.inputModulationMask = StaticDropout(dropout, hidden_size)
        self.cellStateMask = StaticDropout(dropout, hidden_size)
        self.outputModulationMask = StaticDropout(dropout, hidden_size)

    def resetMasks(self, noSequences):
        self.forgetMask.resetMasks(noSequences)
        self.inputModulationMask.resetMasks(noSequences)
        self.cellStateMask.resetMasks(noSequences)
        self.outputModulationMask.resetMasks(noSequences)

        self.noSequences = noSequences

    def setTesting(self):
        self.forgetMask.setTesting()
        self.inputModulationMask.setTesting()
        self.cellStateMask.setTesting()
        self.outputModulationMask.setTesting()

    def lstm_step(self, x, h, c, W_f, V_f, W_i, V_i, W_c, V_c, W_o, V_o, b_f, b_i, b_c, b_o):
        
        xf = torch.matmul(W_f, x.t()).t()
        hf = torch.matmul(V_f, h.t()).t()
        forgetgate = torch.sigmoid(xf + hf + b_f.expand_as(xf))
        forgetgate = self.forgetMask(forgetgate)

        xi = torch.matmul(W_i, x.t()).t()
        hi = torch.matmul(V_i, h.t()).t()
        inputgate = torch.sigmoid(xi + hi + b_i.expand_as(xi))
        inputgate = self.inputModulationMask(inputgate)

        xc = torch.matmul(W_c, x.t()).t()
        hc = torch.matmul(V_c, h.t()).t()
        cellgate = torch.sigmoid(xc + hc + b_c.expand_as(xf))
        cellgate = self.cellStateMask(cellgate)

        xo = torch.matmul(W_o, x.t()).t()
        ho = torch.matmul(V_o, h.t()).t()
        outputgate = torch.sigmoid(xo + ho + b_o.expand_as(xf))
        outputgate = self.outputModulationMask(outputgate)

        c = forgetgate * c + inputgate * cellgate
        h = outputgate * torch.tanh(c)
        
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        
        h, c = states
        outputs = []
        # Split the 3D Tesnor into time-steps for every sequence in the batch:
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_f, self.V_f, self.W_i, self.V_i, self.W_c, self.V_c, self.W_o, self.V_o, self.b_f, self.b_i, self.b_c, self.b_o)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class StaticDropout(nn.Module):
    def __init__(self, p, hidden_dim):
        super(StaticDropout, self).__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.training = True

        if self.p < 1:
            self.dropoutScale = 1.0 / (1.0-p)
        else:
            self.dropoutScale = 0.0

    def resetMasks(self, noSequences):
        self.training = True

        mask = torch.Tensor(noSequences, self.hidden_dim).uniform_(0, 1) > self.p

        # One mask per sequence:
        # self.mask = torch.autograd.Variable(mask.type(torch.cuda.FloatTensor), requires_grad = False)
        self.mask = mask.type(torch.cuda.FloatTensor)

    def setTesting(self):
        self.training = False

    def forward(self, inp):
        if not self.training:
            return inp
            
        # Rescale when training according to inverse dropout framework:
        return self.mask * inp * self.dropoutScale




# dimension_space = [5, 10, 20, 40, 80]
# learningRate_space = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# batch_space = [32, 64, 128, 256, 512]
# 
NoDropoutLSTM = LSTMModels.LSTMPredictor(lookback = 1, interval = 1825, loss_fn = LSTMModels.avgReturnLoss)
NoDropoutLSTM.loadDataset(ratio = 0.9, iteration = 10, seq_length = 63)

NoDropoutLSTM.resetModel(learningRate = 0.01, hidden_dim = 40)
NoDropoutLSTM.fit(num_epochs = 100, batch_size = 256)

DropoutLSTM = MyLSTM(lookback = 1, interval = 1825, loss_fn = LSTMModels.avgReturnLoss)
DropoutLSTM.loadDataset(ratio = 0.9, iteration = 10, seq_length = 63)

# for hidden_dim in dimension_space:
#     for learningRate in learningRate_space:
#         for batch_size in batch_space:
            # print("For hidden dimensions " + str(hidden_dim) + " learning rate " + str(learningRate) + " and batch size " + str(batch_size) + " and MSE loss:")
            
DropoutLSTM.resetModel(learningRate = 0.01, hidden_dim = 40, dropout = 0.3)
DropoutLSTM.fit(num_epochs = 100, batch_size = 256)
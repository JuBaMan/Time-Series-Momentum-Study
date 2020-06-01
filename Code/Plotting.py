import csv
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

def plot(contract, labels):
    dictionary = contract.__dict__

    pd.plotting.register_matplotlib_converters()

    for label in labels:
        plt.plot(dictionary["date"], dictionary[label], label = label)

    leg = plt.legend();
    plt.show()
import csv
import numpy as np
import pickle

def exportCSV(contracts, rootPath):
    for contract in contracts:
        dictionary = contract.__dict__
        
        keys = []

        with open(rootPath + contract.name, 'w', newline = '') as file:
            wr = csv.writer(file)

            for key in dictionary:
                if isinstance(dictionary[key], np.ndarray):
                    keys.append(key)
            
            wr.writerow(keys)

            for i in range(len(dictionary[keys[1]])):
                row = []
                for key in keys:
                    row.append(dictionary[key][i])

                wr.writerow(row)

def exportTXT(array, fileName):
    with open(fileName, "wb") as fp:
        pickle.dump(array, fp)

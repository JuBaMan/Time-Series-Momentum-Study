import ConstructMLDataset

import pickle

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

def reconstruct():
    with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/nonsplitDataPoints.txt", "rb") as fp:
        DATA = pickle.load(fp)

    blueprint = ConstructMLDataset.computeIntervals(1825, DATA)

    contracts = [[] for i in range(len(DATA))]

    for iteration in range(len(blueprint[0])):
        offset = 0

        with open("C:/Users/dream/Desktop/4th Year/Final Project/MLDataSet/Predictions/pred" + str(iteration) + ".txt", "rb") as fp:
            predictions = pickle.load(fp)

        for contract in range(len(names)):
            if iteration + 1 != len(blueprint[contract]):
                if blueprint[contract][iteration] != blueprint[contract][iteration + 1]:
                    contracts[contract].extend(predictions[offset : (offset + blueprint[contract][iteration + 1] - blueprint[contract][iteration])])

                    offset += blueprint[contract][iteration + 1] - blueprint[contract][iteration]
                else:
                    continue
            else:
                contracts[contract].extend(predictions[offset : (offset + len(DATA[contract]) - blueprint[contract][iteration])])

                offset = len(DATA[contract]) - blueprint[contract][iteration]

    return contracts

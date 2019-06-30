# Combined Cycle Power Plant Data Set
# https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearRegression2 import LinearRegression2 as LR
from LeastSquare import LeastSquare as LS 
import matplotlib.pyplot as plt
import time

df = pd.read_csv("Folds5x2_pp.csv", sep=',')

data_in = np.array(df.iloc[: , 0: -1])
expected_out = np.array(df['PE'])

# scale input data
for row in data_in:
    row = row/np.amax(row)

# scale output data
y = expected_out/np.amax(expected_out)

# insert w0
x = np.insert(data_in, 0, values=1, axis=1)

max_iteration = 500000
learning_rate = 0.000000001
initial_params = np.ones(x.shape[1])

start_time = time.time()
lr = LR(x, y, initial_params=initial_params, 
            learning_rate=learning_rate, 
            max_iteration=max_iteration)
print("------ {} s. ------".format(time.time() - start_time))
print("{} iteration".format(max_iteration))
print("learning rate: {:.2e}".format(learning_rate))
print("initial cost: {}".format(lr.costs[0]))
print("final cost: {}".format(lr.costs[-1]))
print("Linear Regression parameters: \n{}".format(
        np.around(lr.parameters[-1], 4)
    ))

start_time = time.time()
ls = LS(x, y)
print("------ {} s. ------".format(time.time() - start_time))
print("Least Square parameters: \n{}".format(
        np.around(ls.parameters, 4)
    ))

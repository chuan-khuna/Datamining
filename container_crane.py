import pandas as pd
import numpy as np
from LinearRegression2 import LinearRegression2 as LR

csv_data = pd.read_csv('Container_Crane_Controller_Data_Set.csv')

x = np.array(csv_data[['Speed', 'Angle']])
y = np.array(csv_data['Power'])

max_iteration = 1000
learning_rate = 0.001
initial_params = np.ones(x.shape[1] + 1)

lr = LR(x, y, initial_params=initial_params, 
            learning_rate=learning_rate, 
            max_iteration=max_iteration)

print("Expected y:   {}".format(y))
print("Hypothesis y: {}".format(
        np.round(lr.predicted_y, 1)
    ))
print("Initial Cost: {}, Final Cost: {}".format(lr.costs[0], lr.costs[-1]))
print("Parameters: {}".format(lr.parameters[-1]))
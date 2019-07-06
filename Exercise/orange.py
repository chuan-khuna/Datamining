# orange dataset from R
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from LinearRegression import LinearRegression as LR
from VisualizationLR import Visualization as VLR
import matplotlib.pyplot as plt 

df = pd.read_csv("orange_5.csv", sep=',')

age = np.array(df['age'])
cir = np.array(df['circumference'])

max_iteration = 3
learning_rate = 0.00001
extend_axis = 1000
lr = LR(age, cir, initial_params=[5, 0.1], learning_rate=learning_rate, 
        max_iteration=max_iteration)
print("without standardization")
print("cost: {}".format(lr.costs[-1]))
vlr = VLR(lr, extend_axis=extend_axis)
# vlr.expected_predicted()
# vlr.cost_iteration()
print(lr.costs)
vlr.contour_cost(grid_res=50, color_res=5)
vlr.show()

s_age = (age - np.mean(age))/np.std(age)
s_cir = (cir - np.mean(cir))/np.std(cir)
max_iteration = 100
learning_rate = 0.2
s_lr = LR(s_age, s_cir, learning_rate=learning_rate, max_iteration=max_iteration)
print("with standardization")
print("cost: {}".format(s_lr.costs[-1]))
s_vlr = VLR(s_lr, extend_axis=1)
# s_vlr.animation_expected_predicted()
# s_vlr.cost_iteration()
# s_vlr.animation_contour_cost(level_round=0.1, delay=0.2)
s_vlr.contour_cost(grid_res=50, color_res=10, level_round=8)
s_vlr.show()
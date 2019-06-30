# dataset from https://archive.ics.uci.edu/ml/datasets/Wine+Quality

import pandas as pd
import numpy as np
from LinearRegression2 import LinearRegression2 as LR
from LeastSquare import LeastSquare as LS

red_wine = pd.read_csv("winequality-red.csv", sep=';')
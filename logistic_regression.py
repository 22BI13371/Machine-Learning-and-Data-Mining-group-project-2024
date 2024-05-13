import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

train_data = "dataset/train.csv"
valid_data = "dataset/valid.csv"

data = pd.read_csv(train_data)
validate_data = pd.read_csv(valid_data)
print(validate_data)

data.shape()


def initialize_theta(dim):
    theta = np.random.rand(dim)
    return theta

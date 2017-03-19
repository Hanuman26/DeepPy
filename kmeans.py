from time import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed()
digits = load_digits()
data = scale(digits.data)

n_sample, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("

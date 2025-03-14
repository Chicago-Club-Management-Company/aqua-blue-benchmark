import numpy as np
import matplotlib.pyplot as plt

import reservoirpy
from reservoirpy.datasets import mackey_glass

import aqua_blue

from aqua_blue_benchmark.benchmark import Stat, Result

from typing import List

# Dataset Generation
TAU = 17
DATA = mackey_glass(n_timesteps=2000, tau=TAU)

# Benchmark Settings
NUM_TRIALS = 30 # Number of times the predictions are carried out for each model. Benchmark results are averaged over this number of trials. 
CRITERIA = Stat.MSE # The benchmark will plot the best and worst performances of each of the models. "Best" and "Worst" are evaluated based on this criteria.

aqua_blue_results: List[Result] = []
respy_results : List[Result] = []

# Model Parameters 
TRAIN_LEN = 1500
TEST_LEN = 500
LEAKING_RATE = 0.3
SPECTRAL_RADIUS = 1.25
RIDGE_COEFFICIENT = 1e-6
RESERVOIR_DIMENSIONALITY = 100 
W = np.random.uniform(
    low=-1.0, 
    high = 1.0, 
    size=(RESERVOIR_DIMENSIONALITY, RESERVOIR_DIMENSIONALITY)
    )
W_IN = np.random.uniform(
    low=-1.0, 
    high = 1.0, 
    size=(RESERVOIR_DIMENSIONALITY, RESERVOIR_DIMENSIONALITY)
    )

# ResevoirPy Setup
input_data = DATA[:-1].reshape(-1, 1)
target_data = DATA[1:].reshape(-1, 1)

# Split into train and test sets
X_train, y_train = input_data[:TRAIN_LEN], target_data[:TRAIN_LEN]
X_test, y_test = input_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN], target_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN]

# Build reservoir
reservoir = reservoirpy.nodes.Reservoir(RESERVOIR_DIMENSIONALITY, lr=0.3, sr=1.25, W=W, Win=W_IN)
readout = reservoirpy.nodes.Ridge(ridge=1e-6)

# Train the model
reservoir >>= readout
readout.fit(X_train, y_train)

# Predict
y_pred_rp = readout.run(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True Mackey-Glass')
plt.plot(y_pred_rp, label='Predicted Mackey-Glass', linestyle='--')
plt.legend()
plt.title('Mackey-Glass Time Series Prediction with ReservoirPy')
plt.show()

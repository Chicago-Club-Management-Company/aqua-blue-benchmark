import numpy as np
import matplotlib.pyplot as plt

import reservoirpy # type: ignore[import-untyped]
from reservoirpy.datasets import mackey_glass # type: ignore[import-untyped]

import aqua_blue # type: ignore[import-untyped]

from typing import List


# Dataset Generation
TAU = 17
DATA = mackey_glass(n_timesteps=2000, tau=TAU)

# Model Parameters 
TRAIN_LEN = 1500
TEST_LEN = 500
LEAKING_RATE = 0.35
SPECTRAL_RADIUS = 1.15
RIDGE_COEFFICIENT = 1e-4
RESERVOIR_DIMENSIONALITY = 100 

GENERATOR = np.random.default_rng(seed=1928301923)


W = GENERATOR.uniform(
    low=-1.0, 
    high=1.0,
    size=(RESERVOIR_DIMENSIONALITY, RESERVOIR_DIMENSIONALITY)
)
    
W_IN = GENERATOR.uniform(
    low=-1.0, 
    high = 1.0, 
    size=(RESERVOIR_DIMENSIONALITY, 1)
    )

# ResevoirPy Setup
input_data = DATA[:-1].reshape(-1, 1)
target_data = DATA[1:].reshape(-1, 1)

# Split into train and test sets
X_train, y_train = input_data[:TRAIN_LEN], target_data[:TRAIN_LEN]
X_test, y_test = input_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN], target_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN]
    
# Build reservoirpy reservoir
reservoir = reservoirpy.nodes.Reservoir(RESERVOIR_DIMENSIONALITY, lr=LEAKING_RATE, sr=SPECTRAL_RADIUS, W=W, Win=W_IN)
readout = reservoirpy.nodes.Ridge(ridge=RIDGE_COEFFICIENT)

# Train the model
reservoir >>= readout
readout.fit(X_train, y_train)

# Predict
y_pred_rp = readout.run(X_test)

# Aqua-Blue Setup 
time_series = aqua_blue.time_series.TimeSeries(dependent_variable=DATA[:1500], times=np.arange(1500))
normalizer = aqua_blue.utilities.Normalizer()
time_series = normalizer.normalize(time_series)


model = aqua_blue.models.Model(
    reservoir=aqua_blue.reservoirs.DynamicalReservoir(
        input_dimensionality=1,
        reservoir_dimensionality=RESERVOIR_DIMENSIONALITY,
        leaking_rate=LEAKING_RATE,
        w_in=W_IN,
        w_res=W,
        spectral_radius=SPECTRAL_RADIUS,
    ),
    readout=aqua_blue.readouts.LinearReadout(rcond=RIDGE_COEFFICIENT)
)

model.train(time_series)
prediction = model.predict(horizon=TEST_LEN)
prediction = normalizer.denormalize(prediction)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(TRAIN_LEN+TEST_LEN-1), np.concatenate((y_train, y_test)), label='True Mackey-Glass')
# plt.plot(np.arange(TRAIN_LEN, TRAIN_LEN+TEST_LEN-1), y_pred_rp, label='ReservoirPy Predicted Mackey-Glass', linestyle='--')
plt.plot(np.arange(TRAIN_LEN, TRAIN_LEN+TEST_LEN), prediction.dependent_variable, label='Aqua-Blue Predicted Mackey-Glass', linestyle='--')
plt.legend()
plt.title('Mackey-Glass Time Series Benchmark')
plt.show()

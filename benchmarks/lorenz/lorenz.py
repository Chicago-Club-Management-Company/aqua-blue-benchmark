import numpy as np
import matplotlib.pyplot as plt

import reservoirpy 
import aqua_blue 
from reservoirpy.datasets import lorenz


# Dataset Generation
DATA = lorenz(n_timesteps=2000)

# Model Parameters 
TRAIN_LEN = 1500
TEST_LEN = 250
LEAKING_RATE = 0.35
SPECTRAL_RADIUS = 1.2
RIDGE_COEFFICIENT = 1e-2
RESERVOIR_DIMENSIONALITY = 100 

GENERATOR = np.random.default_rng(seed=12312)


W = GENERATOR.uniform(
    low=-1.0, 
    high=1.0,
    size=(RESERVOIR_DIMENSIONALITY, RESERVOIR_DIMENSIONALITY)
)
    
W_IN = GENERATOR.uniform(
    low=-1.0, 
    high = 1.0, 
    size=(RESERVOIR_DIMENSIONALITY, 3)
    )

# ResevoirPy Setup
input_data = DATA[:-1]
target_data = DATA[1:]

# Split into train and test sets
X_train, y_train = input_data[:TRAIN_LEN], target_data[:TRAIN_LEN]
X_test, y_test = input_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN], target_data[TRAIN_LEN:TRAIN_LEN+TEST_LEN]


# Build reservoirpy reservoir
reservoir = reservoirpy.nodes.Reservoir(
    units=100,
)
readout = reservoirpy.nodes.Ridge(ridge=1e-2)

readout.fit(reservoir.run(X_train), y_train)

# Predict
y_pred_rp = []
input_step = X_test[0].reshape(1, -1)

for _ in range(len(X_test)):
    res_state = reservoir.run(input_step) 
    pred = readout.run(res_state)  
    y_pred_rp.append(pred)  
    input_step = pred

# Aqua-Blue Setup 
time_series = aqua_blue.time_series.TimeSeries(dependent_variable=DATA[:1500, :], times=np.arange(1500))
normalizer = aqua_blue.utilities.Normalizer()
time_series = normalizer.normalize(time_series)


model = aqua_blue.models.Model(
    reservoir=aqua_blue.reservoirs.DynamicalReservoir(
        input_dimensionality=3,
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

rmse = np.sqrt(np.mean((DATA[TRAIN_LEN:TRAIN_LEN+TEST_LEN, :] - prediction.dependent_variable) ** 2))

print(rmse)
# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(TEST_LEN), y_test[:, 0], label='True Lorenz')
plt.plot(np.arange(TEST_LEN),np.array(y_pred_rp).squeeze()[:, 0], label='ReservoirPy Predicted Lorenz', linestyle='--')
plt.plot(np.arange(TEST_LEN), prediction.dependent_variable[:, 0], label='Aqua-Blue Predicted Lorenz', linestyle='--')
plt.legend()
plt.title('Lorenz System Time Series Benchmark')
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(TEST_LEN), y_test[:, 1], label='True Lorenz')
plt.plot(np.arange(TEST_LEN),np.array(y_pred_rp).squeeze()[:, 1], label='ReservoirPy Predicted Lorenz', linestyle='--')
plt.plot(np.arange(TEST_LEN), prediction.dependent_variable[:, 1], label='Aqua-Blue Predicted Lorenz', linestyle='--')
plt.legend()
plt.title('Lorenz System Time Series Benchmark')
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(TEST_LEN), y_test[:, 2], label='True Lorenz')
plt.plot(np.arange(TEST_LEN),np.array(y_pred_rp).squeeze()[:, 2], label='ReservoirPy Predicted Lorenz', linestyle='--')
plt.plot(np.arange(TEST_LEN), prediction.dependent_variable[:, 2], label='Aqua-Blue Predicted Lorenz', linestyle='--')
plt.legend()
plt.title('Lorenz System Time Series Benchmark')
plt.show()
plt.close()
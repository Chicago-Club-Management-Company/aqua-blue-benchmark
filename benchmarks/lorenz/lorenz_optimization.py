import sys
sys.path.append('../../aqua_blue_benchmark')

from aqua_blue_benchmark import annealer

import numpy as np
from reservoirpy.datasets import lorenz
import aqua_blue
    
# Dataset Generation
DATA = lorenz(n_timesteps=50000)
TRAIN_LEN = 1500 
reservoir_dimensionality=100

GENERATOR = np.random.default_rng(seed=12312)

def f(
    spectral_radius_: float, 
    leaking_rate_: float,
      ): 
    
    W = GENERATOR.uniform(
        low=-1.0, 
        high=1.0,
        size=(reservoir_dimensionality, reservoir_dimensionality)
    )
        
    W_IN = GENERATOR.uniform(
        low=-1.0, 
        high = 1.0, 
        size=(reservoir_dimensionality, 3)
        )
    
    rmse = 12908390128390123
    horizon = 1000
    
    rmses = []
    horizons = []
    
    while rmse > 5: 
        horizon -= 10
    # Aqua-Blue Setup 
        time_series = aqua_blue.time_series.TimeSeries(dependent_variable=DATA[:TRAIN_LEN, :], times=np.arange(TRAIN_LEN))
        normalizer = aqua_blue.utilities.Normalizer()
        time_series = normalizer.normalize(time_series)
        
        
        model = aqua_blue.models.Model(
            reservoir=aqua_blue.reservoirs.DynamicalReservoir(
                input_dimensionality=3,
                reservoir_dimensionality=reservoir_dimensionality,
                leaking_rate=leaking_rate_,
                w_in=W_IN,
                w_res=W,
                spectral_radius=spectral_radius_,
            ),
            readout=aqua_blue.readouts.LinearReadout(rcond=1e-2)
        )
        
        model.train(time_series)
        prediction = model.predict(horizon=horizon)
        prediction = normalizer.denormalize(prediction)
        # print(DATA[TRAIN_LEN:TRAIN_LEN+horizon].shape, prediction.dependent_variable.shape)
        rmse = (np.mean((DATA[TRAIN_LEN:TRAIN_LEN+horizon, :] - prediction.dependent_variable) ** 2))

    return -horizon

out = annealer.optimize(f=f)

annealer.plot_result(out)
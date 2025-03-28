import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable
import progressbar

@dataclass
class Result:
    radii: np.typing.NDArray[np.floating]
    rates: np.typing.NDArray[np.floating]
    horizons: np.typing.NDArray[np.number]
    accepts: np.typing.NDArray[np.number]
    """
    Container class for storing annealing output

    radii : NDArray[float]
        Series of spectral radii throughout annealing 
    
    rates : NDArray[float]
        Series of leaking rates throughout annealing

    horizons : NDArray[int]
        Series of max horizons throughout annealing (within a given RMSE cutoff)

    accepts : NDArray[int]
        Series of accepts throughout annealing
    """


def optimize(
    f : Callable[[float, float], int],
    iterations: int = 10_000, 
    T_i : float = 10, 
    T_f : float = 1e-3,
    sr_init : float = 0.95, 
    lr_init : float = 0.35
) -> Result: 
    
    """ 
    Simulated Annealing for Aqua-Blue Hyperparameter Optimization. 
    Modified to output the optimal values, rather than the final values of the annealing sequence.
    
    f : Callable
        An objective function that takes spectral radius and leaking rate as parameters 
        and returns the negative of the maximum possible prediction horizon within a given RMSE cutoff

        f(sr, lr) -> -horizon_max
    
    iterations : int
        Number of iterations to run the annealer for. Defaults to 10_000 

    T_i : float
        Initial temperature of the annealing run. Defaults to 10 
    
    T_f : float
        Final temperature of the annealing run. Defaults to 0.001
    
    sr_init : float
        Initial spectral radius. Defaults to 0.95
    
    lr_init : float
        Initial leaking rate. Defaults to 0.35
    """


    betas = 1/np.linspace(T_i, T_f, iterations)
    
    # Pregenerate perturbations and random numbers to improve efficiency
    pert_radii = np.random.uniform(low=-0.01, high=0.01, size=(iterations,))
    pert_leaks = np.random.uniform(low=-0.01, high=0.01, size=(iterations,))

    randns = np.random.uniform(size=(iterations))
    
    # Data
    radii = np.zeros(iterations)
    leaks = np.zeros(iterations)
    horizons = -np.zeros(iterations)
    accepts = np.zeros(iterations)
    
    # Initial Values 
    sr = sr_init
    lr = lr_init

    # Optimal Values
    optimal_leak = lr_init
    optimal_radius = sr_init

    # Progress
    bar = progressbar.ProgressBar(max_value=iterations)

    for step in range(iterations): 

        radii[step] = sr
        leaks[step] = lr
        
        horizons[step] = -f(sr, lr)
        
        sr_new = sr + pert_radii[step]
        lr_new = lr + pert_leaks[step]
        
        if sr_new < 0:
            sr_new = sr - pert_radii[step]
        
        if lr_new < 0:
            lr_new = lr - pert_leaks[step]
            
        diff = f(sr_new, lr_new) - horizons[step]
        
        if np.exp(-diff*betas[step]) >= randns[step]:
            sr=sr_new
            lr=lr_new
            accepts[step] = 1
        
        if diff < 0:
            optimal_leak = lr_new
            optimal_radius = sr_new

        bar.update(step)

    print(f'\nOptimal Spectral Radius: {optimal_radius}')
    print(f'Optimal Leaking Rate: {optimal_leak}')
    print(f'Cutoff Horizon: {-f(optimal_radius, optimal_leak)}')

    return Result(radii, leaks, horizons, accepts)


def plot_result(out : Result):

    plt.plot(np.arange(out.radii.size), out.radii)
    plt.plot(np.arange(out.rates.size), out.rates)
    plt.plot(np.arange(out.horizons.size), out.horizons)
    plt.legend(['Spectral Radius', 'Leaking Rate', 'Max Horizon'])
    plt.show()
    plt.close() 

    plt.plot(np.arange(out.accepts.size), np.cumsum(out.accepts))
    plt.show()


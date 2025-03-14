from dataclasses import dataclass

from typing import Callable, Union
import numpy as np

from tabulate import tabulate

@dataclass
class Result: 
    """
    A container object for the results of a time series prediction 
    """

    model_name: str
    """
    The model that produced the output 
    """

    times: np.typing.NDArray[np.floating]
    """
    times: The independent variable of the timeseries 
    """

    prediction: np.typing.NDArray[np.floating]
    """ 
    The prediction made by the reservoir
    """
    
    mse: float 
    """
    Mean-Square Error of a prediction 
    """
    
    rmse: float
    """
    Root Mean Square error of a prediction 
    """

    nrmse_mean: float 
    """
    Normalized Root Mean Square error of a  prediction
    """

    nrmse_minmax: float
    """
    Normalized Root Mean Square error of a prediction 
    """



def compare_results(bench1: Result, bench2: Result):
    # Compare the results of two benchmarks 
    titles = ["\033[94m Model \033[0m", "\033[94m MSE \033[0m", "\033[94m RMSE \033[0m", "\033[94m NRMSE_MEAN \033[0m", "\033[94m NRMSE_MINMAX \033[0m"]
    
    data1 = [bench1.model_name]
    data2 = [bench2.model_name]

    # Compare the stats and color accordingly
    
    # MSE
    if bench1.mse < bench2.mse: 
        data1.append(f"\033[92m{bench1.mse:.4f}\033[0m") 
        data2.append(f'{bench2.mse:.4f}')
    elif bench1.mse > bench2.mse:
        data2.append(f"\033[92m {bench2.mse:.4f} \033[0m")
        data1.append(f'{bench1.mse:.4f}')
    else: 
        data1.append(f'{bench1.mse:.4f}')
        data2.append(f'{bench2.mse:.4f}')
    
    if bench1.rmse < bench2.rmse: 
        data1.append(f"\033[92m{bench1.rmse:.4f}\033[0m") 
        data2.append(f'{bench2.rmse:.4f}')
    elif bench1.rmse > bench2.rmse:
        data2.append(f"\033[92m {bench2.rmse:.4f} \033[0m")
        data1.append(f'{bench1.rmse:.4f}')
    else: 
        data1.append(f'{bench1.rmse:.4f}')
        data2.append(f'{bench2.rmse:.4f}')
    
    if bench1.nrmse_mean < bench2.nrmse_mean: 
        data1.append(f"\033[92m{bench1.nrmse_mean:.4f}\033[0m") 
        data2.append(f'{bench2.nrmse_mean:.4f}')
    elif bench1.nrmse_mean > bench2.nrmse_mean:
        data2.append(f"\033[92m {bench2.nrmse_mean:.4f} \033[0m")
        data1.append(f'{bench1.nrmse_mean:.4f}')
    else: 
        data1.append(f'{bench1.nrmse_mean:.4f}')
        data2.append(f'{bench2.nrmse_mean:.4f}')
    
    if bench1.nrmse_minmax < bench2.nrmse_minmax: 
        data1.append(f"\033[92m{bench1.nrmse_minmax:.4f}\033[0m") 
        data2.append(f'{bench2.nrmse_minmax:.4f}')
    elif bench1.nrmse_minmax > bench2.nrmse_minmax:
        data2.append(f"\033[92m {bench2.nrmse_minmax:.4f} \033[0m")
        data1.append(f'{bench1.nrmse_minmax:.4f}')
    else: 
        data1.append(f'{bench1.nrmse_minmax:.4f}')
        data2.append(f'{bench2.nrmse_minmax:.4f}')
    

    data = [data1, data2]
    print(tabulate(data, headers=titles, tablefmt='grid'))

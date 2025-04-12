import numpy as np
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass 
class SpaceParameters: 
    L: float = 64 
    N: int = 128

@dataclass
class TimeParameters: 
    dt: float = 0.05 
    tmax: float = 100.0


def kuramoto(space_: SpaceParameters, time_: TimeParameters) -> np.typing.NDArray:
    """ Approximate solution to the Kuramoto-Sivashinsky equation using ETDRK4 """
    N = space_.N              
    L = space_.L         
    
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    k = 1j * k

    dt = time_.dt           
    T = time_.tmax        
    nsteps = int(T / dt)
    
    u = 0.1 * np.random.randn(N)
    v = np.fft.fft(u)
    
    Lhat = -k**2 - k**4
    E = np.exp(Lhat * dt)
    E2 = np.exp(Lhat * dt / 2)
    
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * Lhat[:, None] + r
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR)*(-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3, axis=1)
    
    t_output = np.linspace(0, T, 400)
    output_idx = np.linspace(0, nsteps, len(t_output), dtype=int)
    usave = []
    j = 0
    
    # Dealiasing mask (2/3 rule)
    kcut = N // 3
    dealias = np.zeros(N)
    dealias[:kcut] = 1
    dealias[-kcut:] = 1

    # Time stepping
    for i in range(nsteps):
        u_phys = np.fft.ifft(v).real
        Nv = -0.5j * k * np.fft.fft(u_phys ** 2)
        Nv *= dealias

        a = E2 * v + Q * Nv
        Na = -0.5j * k * np.fft.fft(np.fft.ifft(a).real ** 2)
        Na *= dealias

        b = E2 * v + Q * Na
        Nb = -0.5j * k * np.fft.fft(np.fft.ifft(b).real ** 2)
        Nb *= dealias

        c = E2 * a + Q * (2*Nb - Nv)
        Nc = -0.5j * k * np.fft.fft(np.fft.ifft(c).real ** 2)
        Nc *= dealias

        v = E * v + Nv * f1 + 2*(Na + Nb) * f2 + Nc * f3

        if i == output_idx[j]:
            usave.append(np.fft.ifft(v).real.copy())
            j += 1
    
    return np.array(usave)

def main():
    # Space Params
    L = 64
    N = 128  
    
    # Time Params 
    DT = 0.05 
    T_MAX = 100
    
    x = np.arange(0, L, L/N)
    t = np.arange(0, T_MAX, DT)
    
    X,Y = np.meshgrid(x, t)
    
    snapshots = kuramoto(SpaceParameters(), TimeParameters())

    # Plot the results
    plt.imshow(snapshots.T, aspect='auto', extent=[0, T_MAX, 0, L], cmap='inferno', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Kuramoto-Sivashinsky Equation: Spatiotemporal Chaos')
    plt.colorbar(label='u(x,t)')
    plt.show()


if __name__ == "__main__": 
    main()

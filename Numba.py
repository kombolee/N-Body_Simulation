import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import timeit
from numba import jit, njit, vectorize, guvectorize, float64
from multiprocessing.pool import ThreadPool
from dask.distributed import Client
client = Client(processes=False, n_workers=1, threads_per_worker=4) 
import dask
#dask.config.set(scheduler='threads')
from dask import delayed, compute 
#dask.config.set(scheduler='threads', pool=ThreadPool(4))

G = 10.0
m = 1.0
N = 500
dt = 0.001

position = np.random.rand(N, 2)
velocity = np.random.rand(N, 2) * 100
F = np.zeros([N, 2])

@njit
def jit_compute_force_threads(position, F, threads=4): 
    for t in range(threads):
        for i in range(t*threads, (t+1)*threads): 
            for j in range(len(position)):
                if i != j:
                    diff = position[i] - position[j]
                    r = math.sqrt(diff[0]**2+diff[1]**2) 
                    f = -G * (1.0/r)**3 * diff
                    F[i] += f

@njit
def jit_compute_force(position, F): 
    for i in range(len(position)):
        for j in range(len(position)): 
            if i != j:
                diff = position[i] - position[j]
                r = math.sqrt(diff[0]**2+diff[1]**2)
                f = -G * (1.0/r)**3 * diff
                F[i] += f

@njit
def jit_update(position, velocity, F): 
    for i in range(len(position)): 
        velocity[i] += dt * F[i]/m
        position[i] += dt * velocity[i]

def run_par(steps):
    for i in range(steps):
        F.fill(0) 
        #jit_compute_force_threads(position, F) 
        jit_compute_force(position, F) 
        jit_update(position, velocity, F)



run_par(10)
plt.scatter(position[:,0],position[:,1])

import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.dists import Uniform
from nengo.processes import WhiteSignal
from nengo.utils.ensemble import tuning_curves
from nengo.utils.ipython import hide_input
from nengo.utils.matplotlib import rasterplot

"""
Notes: 
In this code, we create a simple input node. The Node takes in 
a lambda t value, representing time, and produces an output 
proportional to time. We can graph this input node over the course of a second. 
"""

def aligned(n_neurons, radius=0.9):
    intercepts = np.linspace(-radius, radius, n_neurons)
    encoders = np.tile([[1], [-1]], (n_neurons // 2, 1))
    intercepts *= encoders[:, 0]
    return intercepts, encoders

model = nengo.Network(label="NEF summary")
with model:
    input = nengo.Node(lambda t: t * 2 - 1)
    input_probe = nengo.Probe(input)

with nengo.Simulator(model) as sim:
    sim.run(1.0)

plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], lw=2)
plt.title("Input signal")
plt.xlabel("Time (s)")
plt.xlim(0, 1)
plt.show()
hide_input()

"""
Notes: 
In this code, we create a simple input node. The Node takes in 
a lambda t value, representing time, and produces an output 
proportional to time. We can graph this input node over the course of a second. 
"""

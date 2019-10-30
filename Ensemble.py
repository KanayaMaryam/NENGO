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
In this code, we will use out input node on an ensemble. 
To create a semi-random ensemble of neurons, we take a 
spread of intercepts and alternatively flip the sign of
each one. These are our encoders and intercepts for each
neuron. As you can see, there are some inputs which stop
because they are negative, with their intercept at the stop point. 
Encoders:
[[ 1]
 [-1]
 [ 1]
 [-1]
 [ 1]
 [-1]
 [ 1]
 [-1]] 
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
plt.xlim(0, 1);
hide_input()

intercepts, encoders = aligned(8)  # Makes evenly spaced intercepts
with model:
    A = nengo.Ensemble(
        8,
        dimensions=1,
        intercepts=intercepts,
        max_rates=Uniform(80, 100),
        encoders=encoders)

with model:
    nengo.Connection(input, A)
    A_spikes = nengo.Probe(A.neurons)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.figure()
ax = plt.subplot(1, 1, 1)
rasterplot(sim.trange(), sim.data[A_spikes], ax)
ax.set_xlim(0, 1)
ax.set_ylabel('Neuron')
ax.set_xlabel('Time (s)');
plt.show()
print (encoders)
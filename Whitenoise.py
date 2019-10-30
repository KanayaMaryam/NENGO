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
Any smooth function can be decoded. 

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

model = nengo.Network(label="NEF summary")
with model:
    input = nengo.Node(lambda t: t * 2 - 1)
    input_probe = nengo.Node(WhiteSignal(1, high=5), size_out=1)
    intercepts, encoders = aligned(8)  # Makes evenly spaced intercepts
    A = nengo.Ensemble(30, dimensions=1,
                       max_rates=Uniform(80, 100))
    nengo.Connection(input, A)
    A_spikes = nengo.Probe(A.neurons, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

scale = 180
plt.figure()
for i in range(A.n_neurons):
    plt.plot(sim.trange(), sim.data[A_spikes][:, i] - i * scale)
plt.xlim(0, 1)
plt.ylim(scale * (-A.n_neurons + 1), scale)
plt.ylabel("Neuron")
plt.yticks(
    np.arange(scale / 1.8, (-A.n_neurons + 1) * scale, -scale),
    np.arange(A.n_neurons))

model = nengo.Network(label="NEF summary")
with model:
    input = nengo.Node(WhiteSignal(1, high=5), size_out=1)
    input_probe = nengo.Probe(input)
    A = nengo.Ensemble(30, dimensions=1, max_rates=Uniform(80, 100))
    nengo.Connection(input, A)
    A_spikes = nengo.Probe(A.neurons)
    A_probe = nengo.Probe(A, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.plot(sim.trange(), sim.data[input_probe], label="Input signal")
plt.plot(sim.trange(), sim.data[A_probe], label="Decoded esimate")
plt.legend(loc="best")
plt.xlabel("Time (s)")
plt.xlim(0, 1)

ax = plt.subplot(1, 2, 2)
rasterplot(sim.trange(), sim.data[A_spikes], ax)
plt.xlim(0, 1)
plt.xlabel("Time (s)")
plt.ylabel("Neuron")
hide_input()

plt.show()



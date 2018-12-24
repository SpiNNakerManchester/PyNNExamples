import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1)
runtime=100

spike_times=[10, 15]
pop_input = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': spike_times}, label="input")

neuron_params = {
    'tau_m': 20.0,
    'cm': 20, # Updated to suit tau_m of 20 and make membrane resistance 1
    'v_rest': 0.0,
    'v': 0,
    "i_offset": 0, # dc current
    'B': 10.0,
    'small_b_0': 10,
    'tau_a': 200,
    'beta': 10,
    'tau_refrac':3
    }

pop_hidden = p.Population(2,
                       p.extra_models.IFCurrDeltaGrazAdaptive(**neuron_params),
                       label='adaptive neuron')

# Input to Hidden Projection (connect to first neuron)
synapse = p.Projection(
    pop_input, pop_hidden, p.FromListConnector([[0, 0, 1000, 10]]),
    p.StaticSynapse(weight=1, delay=1), receptor_type="excitatory")

# Recurrent Hidden 1 to Hidden 2 projection
synapse = p.Projection(
    pop_hidden, pop_hidden, p.FromListConnector([[0, 1, 1000, 7]]),
    p.StaticSynapse(weight=1, delay=1), receptor_type="excitatory")



pop_hidden.record("all")
p.run(runtime)

hidden_data = pop_hidden.get_data()

# Plot
Figure(
    Panel(hidden_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="z(t)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="Threshold B(t)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].spiketrains,
          xlabel='Time (ms)',
          yticks=True, xticks=True, markersize=2, xlim=(0, runtime)),
)


n=0

for i in hidden_data.segments[0].spiketrains[0]:
    print i.magnitude

print "\n\n\n\n\n"
print "*************************************"

for i in hidden_data.segments[0].filter(name='gsyn_inh')[0]:
    print i.magnitude[n]

print "\n\n\n\n\n"
print "*************************************"

for i in hidden_data.segments[0].filter(name='v')[0]:
    print i.magnitude[n]

print "\n\n\n\n\n"
print "*************************************"

for i in hidden_data.segments[0].filter(name='gsyn_exc')[0]:
    print i.magnitude[n]

plt.show()
p.end()



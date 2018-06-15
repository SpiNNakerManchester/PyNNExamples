import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1)
runtime=1000


neuron_params = {
    'tau_m': 20.0,
    'cm': 20, # Updated to suit tau_m of 20 and make membrane resistance 1
    'v_rest': 0.0,
    "i_offset": 200, # dc current
    'thresh_B': 10.0,
    'thresh_b_0': 10,
    'thresh_tau_a': 200,
    'thresh_beta': 10,
    'tau_refrac':3
    }

pop_exc = p.Population(1,
                       p.extra_models.IFCurrDeltaGrazAdaptive(**neuron_params),
                       label='adaptive neuron')

pop_exc.record("all")
p.run(runtime)

exc_data = pop_exc.get_data()

# Plot
Figure(
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="z(t)",
          data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="Threshold B(t)",
          data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].spiketrains,
          xlabel='Time (ms)',
          yticks=True, xticks=True, markersize=2, xlim=(0, runtime)),
)

for i in exc_data.segments[0].spiketrains[0]:
    print i.magnitude

print "\n\n\n\n\n"
print "*************************************"

for i in exc_data.segments[0].filter(name='gsyn_inh')[0]:
    print i.magnitude[0]

print "\n\n\n\n\n"
print "*************************************"

for i in exc_data.segments[0].filter(name='v')[0]:
    print i.magnitude[0]

print "\n\n\n\n\n"
print "*************************************"

for i in exc_data.segments[0].filter(name='gsyn_exc')[0]:
    print i.magnitude[0]

plt.show()
p.end()



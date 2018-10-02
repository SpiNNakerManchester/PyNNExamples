import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from signal_prep import *
p.setup(1.)
runtime = 1000
num_repeats = 2.#1.#
population_size = 10

input_spikes = [i for i in range(10,runtime,100)]
exc_src = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': input_spikes}, label="src1")

# Post-synapse population
pop_exc = p.Population(population_size, p.IF_cond_exp,  label="test")
synapse_exc = p.Projection(
    exc_src, pop_exc, p.OneToOneConnector(),
    p.StaticSynapse(weight=0.1, delay=1), receptor_type="excitatory")

pop_exc.record("all")
for _ in range(int(num_repeats)):
    p.run(runtime/num_repeats)

exc_data = pop_exc.get_data()

print "Post-synaptic neuron firing frequency: {} Hz".format(
    len(exc_data.segments[0].spiketrains[0]))

p.end()
# Plot
Figure(
    # plot data for postsynaptic neuron
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",legend=False,
          yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",legend=False,
           yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",legend=False,
           yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].spiketrains,marker='.',
          yticks=True,markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0,xticks=True, xlim=(0, runtime)),
    annotations="Post-synaptic neuron firing frequency: {} Hz".format(
        len(exc_data.segments[0].spiketrains[0]))
)

plt.show()
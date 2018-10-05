import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from signal_prep import *

p.setup(1.)
runtime = 500
num_repeats = 1.
column_size = 1

prediction_spikes = [i for i in range(50,runtime,100)]
# Spike source to send spike via plastic synapse
exc_src = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': prediction_spikes}, label="src1")

# Post-synapse population
pop_exc = p.Population(column_size, p.IF_cond_exp,{},  label="test")

pred_list = [(0,0)]
synapse_exc = p.Projection(
    exc_src, pop_exc, p.FromListConnector(pred_list),
    p.StaticSynapse(weight=0.1, delay=1), receptor_type="excitatory")

pop_exc.record("all")
weights = []
for _ in range(int(num_repeats)):
    p.run(runtime/num_repeats)
    # runtime = runtime/0.1 # temporary scaling to account for new recording
    weights.append(synapse_exc.get('weight', 'list',
                                       with_address=True))

exc_data = pop_exc.get_data()

print "weights",weights
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
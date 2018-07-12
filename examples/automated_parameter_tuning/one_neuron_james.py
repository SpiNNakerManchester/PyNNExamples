import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1)
runtime = 200

spike_times = [10]# ,11,12,13,14,15,16,17, 18, 19,20]

pop_src1 = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': spike_times}, label="src1")


# Example Distributions
delay_dist = p.RandomDistribution(distribution='exponential', beta=[5.0])

tau_m_dist = p.RandomDistribution(
    distribution='normal_clipped', mu=1, sigma=0.5, high=1.5, low=0.5)

# Neuron parameters
default_parameters = {
        'tau_m': 20.0,
        'cm': tau_m_dist,
        'v_rest': -65.0,
        'v_reset': -70.0,
        'v_thresh': -50.0,
        'tau_syn_E': 5.0,
        'tau_syn_I': 5.0,
        'tau_refrac': 1,
        'i_offset': 0}

# Post-synapse population
pop_exc = p.Population(1, p.IF_curr_exp(**default_parameters),  label="test")

# Create projections
synapse = p.Projection(
    pop_src1, pop_exc, p.OneToOneConnector(),
    p.StaticSynapse(weight=5, delay=delay_dist), receptor_type="excitatory")

pop_src1.record('all')
pop_exc.record("all")
p.run(runtime)

pre_spikes_slow = pop_src1.get_data('spikes')
exc_data = pop_exc.get_data()


# Plot
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(pre_spikes_slow.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # plot data for postsynaptic neuron
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop_src1.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[pop_src1.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[pop_src1.label], yticks=True, xlim=(0, runtime)),
#     Panel(exc_data.segments[0].spiketrains,
#           yticks=True, markersize=0.2, xlim=(0, runtime)),
    annotations="Post-synaptic neuron firing frequency: {} Hz".format(
    len(exc_data.segments[0].spiketrains[0]))
)
plt.show()
p.end()



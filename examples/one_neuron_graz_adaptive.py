import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1)
runtime =1100

spike_times = [10]

pop_src1 = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': spike_times}, label="src1")

neuron_params = {
    'tau_m': 20.0,
    'cm': 1.0,
    'v_rest': -65.0,
    'v_reset': -65.0,
    "i_offset":0, # dc current
    'v_thresh': -50.0,
    'v_thresh_resting': -50,
    'v_thresh_tau': 700,
    'v_thresh_adaptation': 10,
    }

pop_exc = p.Population(1, p.extra_models.IFCurrExpGrazAdaptive(**neuron_params), label='test')

# Create projections
synapse = p.Projection(
    pop_src1, pop_exc, p.OneToOneConnector(),
    p.StaticSynapse(weight=7, delay=1), receptor_type="excitatory")

pop_src1.record('all')
pop_exc.record("all")
p.run(runtime)
weights = []

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
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
)
plt.show()
p.end()



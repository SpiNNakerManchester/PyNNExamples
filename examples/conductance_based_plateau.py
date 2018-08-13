import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1) # simulation timestep (ms)
runtime = 200

# Post-synapse population
neuron_params = {
    "v_thresh": 100,
    "v_reset": 0,
    "v_rest": 0,
    "i_offset": 0,
    "e_rev_E": 80,
#     "tau_syn_E":50,
    "e_rev_I": 0 # DC input
                 }

pop_exc = p.Population(5, # number of neurons
                       p.IF_cond_exp(**neuron_params),  # Neuron model
                       label="LIF Neuron" # identifier
                       )


# Spike source to send spike via synapse
spike_times = [10, 20, 30, 35, 40 , 50, 55, 60, 66, 68, 80]
pop_src1 = p.Population(2, # number of sources
                        p.SpikeSourceArray, # source type
                        {'spike_times': spike_times}, # source spike times
                        label="src1" # identifier
                        )

# Spike source to send spike via synapse
spike_times = [ 80]
pop_src2 = p.Population(1, # number of sources
                        p.SpikeSourceArray, # source type
                        {'spike_times': spike_times}, # source spike times
                        label="src1" # identifier
                        )




# Create projection from source to LIF neuron
synapse = p.Projection(
    pop_src1, pop_exc, p.AllToAllConnector(),
    p.StaticSynapse(weight=0.5, delay=1), receptor_type="excitatory")


# Create projection from source to LIF neuron
synapse = p.Projection(
    pop_src2, pop_exc, p.AllToAllConnector(),
    p.StaticSynapse(weight=5, delay=1), receptor_type="excitatory")

pop_src1.record('spikes')
pop_exc.record("all")

pop_exc.set(i_offset= 0)
p.run(runtime/2)
pop_exc.set(i_offset= 2)
p.run(runtime/4)
pop_exc.set(i_offset= 0)
p.run(runtime/4)


pre_spikes = pop_src1.get_data('spikes')
test = pop_exc.spinnaker_get_data('spikes')
test_v = pop_exc.spinnaker_get_data('v')
# import numpy as np
# np.savetxt("~/test.csv", test_v, delimiter=", ")
exc_data = pop_exc.get_data()

# Plot
F = Figure(
    # plot data for postsynaptic neuron
    Panel(pre_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),
    )

# F.fig.set_adjustable()
# F.fig.subplots_adjust(hspace=2)

plt.show()
p.end()

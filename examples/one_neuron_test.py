import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(0.1) # simulation timestep (ms)
runtime = 100

# Post-synapse population
neuron_params = {
    "v_thresh": -50,
    "v_reset": -70,
    "v_rest": -65,
    "i_offset": 0, # DC input
    "v": -60
                 }

spike_times = [10]
# delays = 250
delays = 2


# pop_src = p.Population(1, p.SpikeSourceArray, {'spike_times': spike_times}, label="src")


pop_exc = p.Population(64, # number of neurons
                       p.IF_curr_exp(**neuron_params),  # Neuron model
                       label="LIF Neuron" # identifier
                       )


# proj_exc = p.Projection(pop_exc, pop_exc, p.FixedProbabilityConnector(0.99),
#                     p.StaticSynapse(weight=0.01, delay=delays), receptor_type="excitatory")
#
# proj_inh = p.Projection(pop_src, pop_exc, p.FixedProbabilityConnector(0.99),
#                     p.StaticSynapse(weight=0.1, delay=delays), receptor_type="inhibitory")

# pop_exc.record("all")

p.run(runtime)

# exc_data = pop_exc.get_data()

# syn_weight_exc = proj_exc.get('weight', 'list', with_address=False)
# syn_weight_inh = proj_inh.get('weight', 'list', with_address=False)

# print syn_weight_exc
# , syn_weight_inh

# Plot
# F = Figure(
#     # plot data for postsynaptic neuron
#     Panel(exc_data.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",
#           data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)
#           ),
#     Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",
#           data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)
#           ),
#     Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
#           ylabel="gsyn inhibitory (mV)",
#           data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)
#           ),
#     Panel(exc_data.segments[0].spiketrains,
#           yticks=True, markersize=2, xlim=(0, runtime)
#           ),
#     )

# plt.show()

p.end()
print 'job done'


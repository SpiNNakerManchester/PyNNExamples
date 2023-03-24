# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A single LIF neuron with two exponential, current-based synapses,
and two delta, current-based synapses, fed by two spike sources.
"""

import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

sim.setup(timestep=1.0, min_delay=1.0)

delta_cell = sim.Population(1, sim.IF_curr_delta(**{
    'i_offset': 0.1,
    'tau_refrac': 3.0,
    'v_thresh': -51.0,
    'v_reset': -70.0}))

exp_cell = sim.Population(1, sim.IF_curr_exp(**{
    'i_offset': 0.1,
    'tau_refrac': 3.0,
    'v_thresh': -51.0,
    'v_reset': -70.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 5.0}))


spike_sourceE = sim.Population(1, sim.SpikeSourceArray(**{
    'spike_times': [float(i) for i in range(5, 105, 10)]}))
spike_sourceI = sim.Population(1, sim.SpikeSourceArray(**{
    'spike_times': [float(i) for i in range(155, 255, 10)]}))

sim.Projection(spike_sourceE, exp_cell,
               sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=1.5, delay=2.0),
               receptor_type='excitatory')
sim.Projection(spike_sourceI, exp_cell,
               sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=-1.5, delay=4.0),
               receptor_type='inhibitory')
sim.Projection(spike_sourceE, delta_cell,
               sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=1.5, delay=2.0),
               receptor_type='excitatory')
sim.Projection(spike_sourceI, delta_cell,
               sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=-1.5, delay=4.0),
               receptor_type='inhibitory')

delta_cell.record('all')
exp_cell.record('all')

runtime = 200.0

sim.run(runtime)

stoc_data = delta_cell.get_data()
exp_data = exp_cell.get_data()

# Plot
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(stoc_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(exp_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(stoc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[delta_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(stoc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[delta_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(stoc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[delta_cell.label], yticks=True, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(exp_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[exp_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(exp_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[exp_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(exp_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[exp_cell.label], yticks=True, xlim=(0, runtime)),
    title="IF_curr_delta example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()

sim.end()

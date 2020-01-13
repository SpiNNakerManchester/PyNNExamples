# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
A single LIF neuron with two exponential, current-based synapses,
and two delta, current-based synapses, fed by two spike sources.
"""

import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

sim.setup(timestep=1.0, min_delay=1.0, max_delay=4.0)

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

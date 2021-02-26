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
A population of integrate-and-firing neurons with different input firing rates
(example used in the HBP Neuromorphic Computing Guidebook)
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import pyNN.spiNNaker as sim
except ImportError:
    import spynnaker8 as sim

sim.setup(timestep=1.0, min_delay=1.0)

# create cells
cell_params = {
    'cm': 0.25, 'tau_m': 10.0, 'tau_refrac': 2.0,
    'tau_syn_E': 2.5, 'tau_syn_I': 2.5,
    'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -55.0}

neurons = sim.Population(100, sim.IF_cond_exp(**cell_params))
inputs = sim.Population(100, sim.SpikeSourcePoisson(rate=0.0))

# set input firing rates as a linear function of cell index
input_firing_rates = np.linspace(0.0, 1000.0, num=inputs.size)
inputs.set(rate=input_firing_rates)

# create one-to-one connections
wiring = sim.OneToOneConnector()
static_synapse = sim.StaticSynapse(weight=0.1, delay=2.0)
connections = sim.Projection(inputs, neurons, wiring, static_synapse)

# configure recording
neurons.record('spikes')

# run simulation
sim_duration = 10.0  # seconds
sim.run(sim_duration*1000.0)

# retrieve recorded data
spike_counts = neurons.get_spike_counts()
print(spike_counts)
output_firing_rates = np.array(
    [value for (key, value) in sorted(spike_counts.items())])/sim_duration

sim.end()

# plot graph
plt.plot(input_firing_rates, output_firing_rates)
plt.xlabel("Input firing rate (spikes/second)")
plt.ylabel("Output firing rate (spikes/second)")
plt.savefig("simple_example.png")
plt.show()

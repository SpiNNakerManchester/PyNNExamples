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
Simple test for Structural Plasticity modified from a simple test for STDP.

The current file contains no STDP (all weights are static). The only plastic
element in the network is the connectivity.

To note:
1. There is no initial connectivity
2. Over time, at a rate of 10 kHz, rewiring is attempted, connectivity the
    source and target layers
3. The sigma for feed-forward formation is quite small
4. In the final testing phase enough connections have been formed to see
    a duplication of pre-synaptic spikes in the post-synaptic spiking activity

author: Petrut Bogdan
date  : March, 2018
"""

import numpy as np
import pylab as plt
import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
# +---------------------------------------------------------------------------+
# | General Parameters                                                        |
# +---------------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }

# Other simulation parameters
e_rate = 80
in_rate = 300

n_stim_test = 5
n_stim_pairing = 20
dur_stim = 20

# pop_size = 14**2
pop_size = 7 ** 2

isi = 90.
start_test_pre_pairing = 200.
start_pairing = 1500.
start_test_post_pairing = 700.

sim_time = (start_pairing + start_test_post_pairing
            + isi * (n_stim_pairing + n_stim_test) + 550.)

# Initialisations of the different types of populations
i_add_pre = []
i_add_post = []

# +---------------------------------------------------------------------------+
# | Creation of neuron populations                                            |
# +---------------------------------------------------------------------------+

# Neuron populations
pre_pop = sim.Population(pop_size, model(**cell_params),
                         label="PRE_SYN_POP")
post_pop = sim.Population(pop_size, model(**cell_params),
                          label="STRUCTURAL_POP")

# Test of the effect of activity of the pre_pop population on the post_pop
# population prior to the "pairing" protocol : only pre_pop is stimulated
for i in range(n_stim_test):
    i_add_pre.append(sim.Population(
        pop_size,
        sim.SpikeSourcePoisson,
        {'rate': in_rate,
         'start': start_test_pre_pairing + isi * (i),
         'duration': dur_stim
         }))

# Pairing protocol : pre_pop and post_pop are stimulated with a 10 ms
# difference
for i in range(n_stim_pairing):
    i_add_pre.append(sim.Population(
        pop_size,
        sim.SpikeSourcePoisson,
        {'rate': in_rate,
         'start': start_pairing + isi * (i),
         'duration': dur_stim
         }))
    i_add_post.append(sim.Population(
        pop_size,
        sim.SpikeSourcePoisson,
        {'rate': in_rate,
         'start': start_pairing + isi * (i) + 10.,
         'duration': dur_stim
         }))

# Test post pairing : only pre_pop is stimulated (and should trigger activity
# in Post)
for i in range(n_stim_test):
    i_add_pre.append(sim.Population(
        pop_size,
        sim.SpikeSourcePoisson,
        {'rate': in_rate,
         'start': (start_pairing
                   + isi * (n_stim_pairing)
                   + start_test_post_pairing
                   + isi * (i)),
         'duration': dur_stim
         })
    )

# Noise inputs
i_noise_pre = sim.Population(
    pop_size,
    sim.SpikeSourcePoisson, {'rate': e_rate, 'start': 0, 'duration': sim_time},
    label="expoisson")
i_noise_post = sim.Population(
    pop_size,
    sim.SpikeSourcePoisson, {'rate': e_rate, 'start': 0, 'duration': sim_time},
    label="inpoisson")

# +---------------------------------------------------------------------------+
# | Creation of connections                                                   |
# +---------------------------------------------------------------------------+

# Connection parameters
jee = 3.

# Noise projections
sim.Projection(
    i_noise_pre, pre_pop, sim.OneToOneConnector(),
    receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=jee * 0.05))
sim.Projection(
    i_noise_post, post_pop, sim.OneToOneConnector(),
    receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=jee * 0.05))

# Additional Inputs projections
for i in range(len(i_add_pre)):
    sim.Projection(
        i_add_pre[i], pre_pop, sim.OneToOneConnector(),
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=jee * 0.05))
for i in range(len(i_add_post)):
    sim.Projection(
        i_add_post[i], post_pop, sim.OneToOneConnector(),
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=jee * 0.05))

# Structurally plastic connection between pre_pop and post_pop
partner_selection_last_neuron = sim.RandomSelection()
formation_distance = sim.DistanceDependentFormation(
    grid=[np.sqrt(pop_size), np.sqrt(pop_size)],  # spatial org of neurons
    sigma_form_forward=.5  # spread of feed-forward connections
)
elimination_weight = sim.RandomByWeightElimination(
    threshold=.2  # Use same weight as initial weight for static connections
)
structure_model_without_stdp = sim.StructuralMechanismStatic(
    # Partner selection, formation and elimination rules from above
    partner_selection_last_neuron, formation_distance, elimination_weight,
    # Use this weight when creating a new synapse
    initial_weight=.2,
    # Use this weight for synapses at start of simulation
    weight=.2,
    # Use this delay when creating a new synapse
    initial_delay=10,
    # Use this weight for synapses at the start of simulation
    delay=10,
    # Maximum allowed fan-in per target-layer neuron
    s_max=32,
    # Frequency of rewiring in Hz
    f_rew=10 ** 4
)

plastic_projection = sim.Projection(
    pre_pop, post_pop,
    sim.FixedProbabilityConnector(0.),  # No initial connections
    synapse_type=structure_model_without_stdp,
    label="structurally_plastic_projection"
)

# +---------------------------------------------------------------------------+
# | Simulation and results                                                    |
# +---------------------------------------------------------------------------+

# Record neurons' potentials
pre_pop.record(['v', 'spikes'])
post_pop.record(['v', 'spikes'])

# Run simulation
sim.run(sim_time)

# Retrieve connectivity information from SpiNNaker
pre_weights = []
pre_weights.append(
    plastic_projection.get(["weight", "delay"], format="list")
)
pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')

# End simulation on SpiNNaker
sim.end()

# Plotting spikes
Figure(
    # raster plot of the neuron spike times
    Panel(pre_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time)),
    Panel(post_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time)),
    title="Structural plasticity without STDP example",
)

# Plotting connectivity
final_ff_conn_network = np.ones((pop_size, pop_size)) * np.nan
final_ff_weight_network = np.ones((pop_size, pop_size)) * np.nan
for source, target, weight, delay in pre_weights[-1]:
    if np.isnan(final_ff_weight_network[int(source), int(target)]):
        final_ff_weight_network[int(source), int(target)] = weight
    else:
        final_ff_weight_network[int(source), int(target)] += weight
    assert delay == 10

for source, target, weight, delay in pre_weights[-1]:
    if np.isnan(final_ff_conn_network[int(source), int(target)]):
        final_ff_conn_network[int(source), int(target)] = 1
    else:
        final_ff_conn_network[int(source), int(target)] += 1
    assert delay == 10

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
i = ax1.matshow(np.nan_to_num(final_ff_weight_network))
i2 = ax2.matshow(np.nan_to_num(final_ff_conn_network))
ax1.grid(visible=False)
ax1.set_title("Feed-forward weighted connectivity matrix", fontsize=16)
ax2.set_title("Feed-forward connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i2, cax=cbar_ax)
cbar.set_label("Number of connections", fontsize=16)
plt.show()

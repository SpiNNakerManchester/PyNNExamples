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

import spynnaker8 as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp
cell_params = {'cm': 0.25, 'i_offset': 0.0, 'tau_m': 10.0,
               'tau_refrac': 2.0, 'tau_syn_E': 2.5, 'tau_syn_I': 2.5,
               'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -55.4}

delta_t = 10
time_between_pairs = 150
num_pre_pairs = 10
num_pairs = 100
num_post_pairs = 10
pop_size = 1

pairing_start_time = (num_pre_pairs * time_between_pairs) + delta_t
pairing_end_time = pairing_start_time + (num_pairs * time_between_pairs)
sim_time = pairing_end_time + (num_post_pairs * time_between_pairs)

# +-------------------------------------------------------------------+
# | Creation of neuron populations                                    |
# +-------------------------------------------------------------------+
# Neuron populations
pre_pop = sim.Population(pop_size, model, cell_params)
post_pop = sim.Population(pop_size, model, cell_params)

# Stimulating populations
pre_stim = sim.Population(pop_size, sim.SpikeSourceArray,
                          {'spike_times':
                           [[i for i in range(0, sim_time,
                                              time_between_pairs)], ]})
post_stim = sim.Population(pop_size, sim.SpikeSourceArray,
                           {'spike_times':
                            [[i for i in range(pairing_start_time,
                                               pairing_end_time,
                                               time_between_pairs)], ]})

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+
# Connection type between noise poisson generator and
# excitatory populations
ee_connector = sim.OneToOneConnector()
synapse_type = sim.StaticSynapse(weight=2)

sim.Projection(pre_stim, pre_pop, ee_connector, synapse_type=synapse_type,
               receptor_type='excitatory')
sim.Projection(post_stim, post_pop, ee_connector,
               synapse_type=synapse_type, receptor_type='excitatory')

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
  timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=50.0,
                                      A_plus=0.02, A_minus=0.02),
  weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=1))

sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(),
               synapse_type=stdp_model)

# Record spikes
pre_pop.record("spikes")
post_pop.record("spikes")

# Run simulation
sim.run(sim_time)

pre_neo = pre_pop.get_data("spikes")
post_neo = post_pop.get_data("spikes")

# End simulation on SpiNNaker
sim.end()

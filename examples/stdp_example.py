# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple test for STDP :

   Reproduces a classical plasticity experiment of plasticity induction by
pre/post synaptic pairing specifically :

 * At the beginning of the simulation, "n_stim_test" external stimulations of
   the "pre_pop" (presynaptic) population do not trigger activity in the
   "post_pop" (postsynaptic) population.

 * Then the presynaptic and postsynaptic populations are stimulated together
   "n_stim_pairing" times by an external source so that the "post_pop"
   population spikes 10ms after the "pre_pop" population.

 * After that period, only the "pre_pop" population is externally stimulated
   "n_stim_test" times, but now it should trigger activity in the "post_pop"
   population (due to STDP learning)

Run as :

   $ ./stdp_example

This example requires that the NeuroTools package is installed
(https://neuralensemble.org/trac/NeuroTools)

Authors : Catherine Wacongne < catherine.waco@gmail.com >
          Xavier Lagorce < Xavier.Lagorce@crans.org >

April 2013
"""
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0)

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp

cell_params = {
    'cm': 0.25, 'i_offset': 0.0, 'tau_m': 20.0,
    'tau_refrac': 2.0, 'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
    'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -50.0}


# Other simulation parameters
e_rate = 80
in_rate = 300

n_stim_test = 5
n_stim_pairing = 20
dur_stim = 20

pop_size = 40

isi = 90.
start_test_pre_pairing = 200.
start_pairing = 1500.
start_test_post_pairing = 700.

sim_time = (start_pairing + start_test_post_pairing
            + isi * (n_stim_pairing + n_stim_test) + 550.)

# Initialisations of the different types of populations
i_add_pre = []
i_add_post = []

# +-------------------------------------------------------------------+
# | Creation of neuron populations                                    |
# +-------------------------------------------------------------------+

# Neuron populations
pre_pop = sim.Population(pop_size, model(**cell_params))
post_pop = sim.Population(pop_size, model(**cell_params))

# Test of the effect of activity of the pre_pop population on the post_pop
# population prior to the "pairing" protocol : only pre_pop is stimulated
for i in range(n_stim_test):
    i_add_pre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=start_test_pre_pairing + isi * i,
            duration=dur_stim)))

# Pairing protocol : pre_pop and post_pop are stimulated with a 10 ms
# difference
for i in range(n_stim_pairing):
    i_add_pre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
           rate=in_rate, start=start_pairing + isi * i, duration=dur_stim)))
    i_add_post.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=start_pairing + isi * i + 10.,
            duration=dur_stim)))

# Test post pairing : only pre_pop is stimulated (and should trigger activity
# in Post)
for i in range(n_stim_test):
    i_add_pre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=(
                    start_pairing + isi * n_stim_pairing +
                    start_test_post_pairing + isi * i),
            duration=dur_stim)))

# Noise inputs
i_noise_pre = sim.Population(
    pop_size, sim.SpikeSourcePoisson(
        rate=e_rate, start=0, duration=sim_time), label="expoisson")
i_noise_post = sim.Population(
    pop_size, sim.SpikeSourcePoisson(
        rate=e_rate, start=0, duration=sim_time), label="expoisson")

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+

# Connection parameters
jee = 3.

# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector()

# Noise projections
sim.Projection(
    i_noise_pre, pre_pop, ee_connector, receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=jee * 0.05))
sim.Projection(
    i_noise_post, post_pop, ee_connector, receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=jee * 0.05))

# Additional Inputs projections
for _i_add_pre in i_add_pre:
    sim.Projection(
        _i_add_pre, pre_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=jee * 0.05))
for _i_add_post in i_add_post:
    sim.Projection(
        _i_add_post, post_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=jee * 0.05))

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9))

plastic_projection = sim.Projection(
    pre_pop, post_pop, sim.FixedProbabilityConnector(p_connect=0.5),
    synapse_type=stdp_model)

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
pre_pop.record(['v', 'spikes'])
post_pop.record(['v', 'spikes'])

# Run simulation
sim.run(sim_time)

print(f"Weights:{plastic_projection.get('weight', 'list')}")

pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(pre_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.5, xlim=(0, sim_time)),
    Panel(post_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.5, xlim=(0, sim_time)),
    title="stdp example curr",
    annotations=f"Simulated with {sim.name()}")
plt.show()

# End simulation on SpiNNaker
sim.end()

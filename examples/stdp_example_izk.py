# Copyright (c) 2020 The University of Manchester
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

Modified for Izhikevich's Current neurons by Dr. Basabdatta Sen-Bhattacharya
and Ishita Mediratta  (BITS Pilani K.K. Birla Goa Campus, India) in June 2020

************************************************************************

Simple test for STDP :

   Reproduces a classical plasticity experiment of plasticity induction by
pre/post synaptic pairing specifically :

 * At the begining of the simulation, "n_stim_test" external stimulations of
   the "pre_pop" (presynaptic) population do not trigger activity in the
   "post_pop" (postsynaptic) population.

 * Then the presynaptic and postsynaptic populations are stimulated together
   "n_stim_pairing" times by an external source so that the "post_pop"
   population spikes 10ms after the "pre_pop" population.

 * Ater that period, only the "pre_pop" population is externally stimulated
   "n_stim_test" times, but now it should trigger activity in the "post_pop"
   population (due to STDP learning)

Run as :

   $ ./stdp_example

This example requires that the NeuroTools package is installed
(http://neuralensemble.org/trac/NeuroTools)

Authors : Catherine Wacongne < catherine.waco@gmail.com >
          Xavier Lagorce < Xavier.Lagorce@crans.org >

April 2013
"""

import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# SpiNNaker setup
sim.setup(timestep=0.1, min_delay=1.0)

model = sim.Izhikevich
a_tonic = 0.02
b_tonic = 0.2
c_tonic = -65.0
d_tonic = 6.0
v_init_tonic = -65.0
u_init_tonic = b_tonic * v_init_tonic
current_Pulse = 0
tau_ex = 1.7
tau_inh = 2.5
cell_params = {'a': a_tonic, 'b': b_tonic, 'c': c_tonic, 'd': d_tonic,
               'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
               'i_offset': current_Pulse
               }

initial_values = {'v': v_init_tonic, 'u': u_init_tonic}

e_rate = 60
in_rate = 300

n_stim_test = 5
n_stim_pairing = 20
dur_stim = 20

pop_size = 40

ISI = 90.
start_test_pre_pairing = 200.
start_pairing = 1500.
start_test_post_pairing = 700.

simtime = (start_pairing + start_test_post_pairing
           + ISI * (n_stim_pairing + n_stim_test) + 550.)

# Initialisations of the different types of populations
IAddPre = []
IAddPost = []

# Neuron populations
pre_pop = sim.Population(pop_size, model(**cell_params),
                         initial_values=initial_values)
post_pop = sim.Population(pop_size, model(**cell_params),
                          initial_values=initial_values)

# Test of the effect of activity of the pre_pop population on the post_pop
# population prior to the "pairing" protocol : only pre_pop is stimulated
for i in range(n_stim_test):
    IAddPre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=start_test_pre_pairing + ISI * i,
            duration=dur_stim)))

# Pairing protocol : pre_pop and post_pop are stimulated with a 10 ms
# difference
for i in range(n_stim_pairing):
    IAddPre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
           rate=in_rate, start=start_pairing + ISI * i, duration=dur_stim)))
    IAddPost.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=start_pairing + ISI * i + 10,
            duration=dur_stim)))
    # Pre spikes 10ms after Post so that Hebb can cause LTP

# Test post pairing : only pre_pop is stimulated (and should trigger activity
# in Post)
for i in range(n_stim_test):
    IAddPre.append(sim.Population(
        pop_size, sim.SpikeSourcePoisson(
            rate=in_rate, start=(
                start_pairing + ISI * n_stim_pairing +
                start_test_post_pairing + ISI * i),
            duration=dur_stim)))

# Noise inputs
INoisePre = sim.Population(
    pop_size, sim.SpikeSourcePoisson(
        rate=e_rate, start=0, duration=simtime), label="expoisson")
INoisePost = sim.Population(
    pop_size, sim.SpikeSourcePoisson(
        rate=e_rate, start=0, duration=simtime), label="expoisson")

# Connection parameters
JEE = 5

# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector()

# Noise projections
sim.Projection(
    INoisePre, pre_pop, ee_connector, receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=JEE))
sim.Projection(
    INoisePost, post_pop, ee_connector, receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=JEE))

# Additional Inputs projections
for i in range(len(IAddPre)):
    sim.Projection(
        IAddPre[i], pre_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=JEE))
for i in range(len(IAddPost)):
    sim.Projection(
        IAddPost[i], post_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=JEE))

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=20.0, tau_minus=20.0, A_plus=0.045, A_minus=0.045),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=10))

plastic_projection = sim.Projection(
    pre_pop, post_pop, sim.FixedProbabilityConnector(p_connect=0.5),
    synapse_type=stdp_model)

# Record neurons' potentials
pre_pop.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])
post_pop.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

# Run simulation
sim.run(simtime)

print("Weights:{}".format(plastic_projection.get('weight', 'list')))

# Get data
pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')
pre_gsyn_e = pre_pop.get_data("gsyn_exc")
post_gsyn_e = post_pop.get_data("gsyn_exc")
pre_gsyn_i = pre_pop.get_data("gsyn_inh")
post_gsyn_i = post_pop.get_data("gsyn_inh")

# Plot data
Figure(
    Panel(pre_gsyn_e.segments[0].filter(name="gsyn_exc")[0],
          ylabel="pre_gsyn_e", yticks=True, markersize=0.2, xlim=(0, simtime),
          legend=False),
    Panel(post_gsyn_e.segments[0].filter(name="gsyn_exc")[0],
          ylabel="post_gsyn_e", yticks=True, xticks=True, markersize=0.2,
          xlim=(0, simtime), legend=False),
    Panel(pre_gsyn_i.segments[0].filter(name="gsyn_inh")[0],
          ylabel="pre_gsyn_i", yticks=True, markersize=0.2, xlim=(0, simtime),
          legend=False),
    Panel(post_gsyn_i.segments[0].filter(name="gsyn_inh")[0],
          ylabel="post_gsyn_i", yticks=True, xticks=True, markersize=0.2,
          xlim=(0, simtime), legend=False),
    # raster plot of the pre- and post-synaptic neuron spike times
    Panel(pre_spikes.segments[0].spiketrains,
          yticks=True, markersize=2.0, xlim=(0, simtime), color="red",
          ylabel="Pre-Population"),
    Panel(post_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=2.0,
          xlim=(0, simtime), color="blue", ylabel="Post-Population"),
    title="Excitatory synaptic conductance (mS) and spike raster plots",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()

# End simulation on SpiNNaker
sim.end()

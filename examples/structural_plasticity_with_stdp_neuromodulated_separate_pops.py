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
Simple test for neuromodulated-STDP incorporating structural plasticity

We take 10 populations of 49 stimuli neurons and connect to each 10
populations of 49 neurons using  a straightforward creation/elimination
structural method (similar to that used in the struct_pl examples).  Then we
connect each of these 10 populations (via one-to-one connections) to
10 post-synaptic populations of 49 neurons.

The spiking of stimuli causes some spikes in post-synaptic neurons initially.
We then inject reward signals from dopaminergic neurons
periodically to reinforce synapses that are active. This
is followed by increased weights of some synapses and thus
increased response to the stimuli.
We then proceed to inject punishment signals from dopaminergic
neurons which causes an inverse effect to reduce response of
post-synaptic neurons to the same stimuli.
"""

import pyNN.spiNNaker as sim
import pylab
import numpy as np

timestep = 1.0
stim_rate = 100
duration = 12000
plastic_weights = 3.5
struct_pl_weights = 0.5
n_neurons = 7**2
n_pops = 10

# Times of rewards and punishments
rewards = [x for x in range(2000, 2010)] + \
          [x for x in range(3000, 3020)] + \
          [x for x in range(4000, 4100)]
punishments = [x for x in range(6000, 6010)] + \
              [x for x in range(7000, 7020)] + \
              [x for x in range(8000, 8100)]

cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 1.0,
               'tau_syn_I': 1.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }

sim.setup(timestep=timestep)

# Create a population of dopaminergic neurons for reward and punishment
reward_pop = sim.Population(n_neurons, sim.SpikeSourceArray,
                            {'spike_times': rewards}, label='reward')
punishment_pop = sim.Population(n_neurons, sim.SpikeSourceArray,
                                {'spike_times': punishments},
                                label='punishment')

pre_pops = []
stimulation = []
mid_pops = []
post_pops = []
reward_projections = []
punishment_projections = []
plastic_projections = []
mid_projections = []
stim_projections = []

# Create synapse dynamics with neuromodulated STDP and structural plasticity
# Structurally plastic connection between pre_pop and post_pop
partner_selection_last_neuron = sim.RandomSelection()
formation_distance = sim.DistanceDependentFormation(
    grid=[np.sqrt(n_neurons), np.sqrt(n_neurons)],  # spatial neurons
    sigma_form_forward=1.0  # spread of feed-forward connections
)
elimination_weight = sim.RandomByWeightElimination(
    prob_elim_potentiated=0.4,
    prob_elim_depressed=0.4,
    threshold=struct_pl_weights  # Use same weight as initial weight for static
)

structure_model_with_stdp = sim.StructuralMechanismStatic(
    # Partner selection, formation and elimination rules from above
    partner_selection_last_neuron, formation_distance, elimination_weight,
    # Use this weight when creating a new synapse
    initial_weight=struct_pl_weights,
    # Use this weight for synapses at start of simulation
    weight=struct_pl_weights,
    # Use this delay when creating a new synapse
    initial_delay=10,
    # Use this weight for synapses at the start of simulation
    delay=10,
    # Maximum allowed fan-in per target-layer neuron
    s_max=32,
    # Frequency of rewiring in Hz
    f_rew=10 ** 4
)

synapse_dynamics_neuromod = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=2, tau_minus=1,
        A_plus=1, A_minus=1),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=5.0),
    weight=plastic_weights)


for i in range(n_pops):
    stimulation.append(sim.Population(n_neurons, sim.SpikeSourcePoisson,
                       {'rate': stim_rate, 'duration': duration}, label="pre"))
    mid_pops.append(sim.Population(
        n_neurons, sim.IF_curr_exp, cell_params, label="mid"))
    post_pops.append(sim.Population(
        n_neurons, sim.IF_curr_exp,
        cell_params, label='post'))
    mid_projections.append(sim.Projection(
        stimulation[i], mid_pops[i],
        # small number of initial connections
        sim.FixedProbabilityConnector(0.0),
        synapse_type=structure_model_with_stdp,
        label="stim-mid projection"))
    plastic_projections.append(
        sim.Projection(mid_pops[i], post_pops[i],
                       sim.OneToOneConnector(),
                       synapse_type=synapse_dynamics_neuromod,
                       receptor_type='excitatory',
                       label='Mid-post projection'))
    mid_pops[i].record('spikes')
    post_pops[i].record('spikes')
    reward_projections.append(sim.Projection(
        reward_pop, post_pops[i], sim.OneToOneConnector(),
        synapse_type=sim.extra_models.Neuromodulation(
            weight=0.05, tau_c=100.0, tau_d=5.0, w_max=5.0),
        receptor_type='reward', label='reward synapses'))
    punishment_projections.append(sim.Projection(
        punishment_pop, post_pops[i], sim.OneToOneConnector(),
        synapse_type=sim.extra_models.Neuromodulation(
            weight=0.05, tau_c=100.0, tau_d=5.0, w_max=5.0),
        receptor_type='punishment', label='punishment synapses'))

sim.run(duration)

# Graphical diagnostics


def plot_spikes(_mid_spikes, _spikes, _title, _n_pops, _n_neurons):
    """
    Creates a plot of the spikes if any received.

    :param list([int, quantity]) _mid_spikes:
    :param list([int, quantity]) _spikes:
    :param str _title:
    :param iny _n_pops:
    :param int _n_neurons:
    """
    if _spikes is not None:
        pylab.figure(figsize=(15, 5))
        pylab.xlim((0, duration))
        pylab.ylim((0, (_n_pops * _n_neurons) + 1))
        pylab.plot(
            [i[1] for i in _mid_spikes], [i[0] for i in _mid_spikes], "y.")
        pylab.plot([i[1] for i in _spikes], [i[0] for i in _spikes], "b.")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(_title)
    else:
        print("No spikes received")


post_spikes = []
mid_spikes = []
weights = []

for i in range(n_pops):
    weights.append(plastic_projections[i].get('weight', 'list'))
    spikes = post_pops[i].get_data('spikes').segments[0].spiketrains
    spikes_mid = mid_pops[i].get_data('spikes').segments[0].spiketrains
    for j in range(n_neurons):
        for x in spikes[j]:
            post_spikes.append([(i*n_neurons)+j+1, x])
        for x in spikes_mid[j]:
            mid_spikes.append([(i*n_neurons)+j+1, x])

plot_spikes(mid_spikes, post_spikes, "mid- and post-synaptic neuron activity",
            n_pops, n_neurons)
pylab.plot(rewards, [0.5 for x in rewards], 'g^')
pylab.plot(punishments, [0.5 for x in punishments], 'r^')
pylab.show()

print("Weights(Initial {plastic_weights})")
for x in weights:
    print(x)

sim.end()

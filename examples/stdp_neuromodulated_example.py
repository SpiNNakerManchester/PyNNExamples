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
Simple test for neuromodulated-STDP
We take 10 populations of 5 stimuli neurons and connect to each
10 post-synaptic populations of 5 neurons. The spiking of stimuli causes some
spikes in post-synaptic neurons initially.
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

timestep = 1.0
stim_rate = 50
duration = 12000
plastic_weights = 1.5
n_neurons = 5
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
post_pops = []
reward_projections = []
punishment_projections = []
plastic_projections = []
stim_projections = []

# Create synapse dynamics with neuromodulated STDP.
synapse_dynamics = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=2, tau_minus=1,
        A_plus=1, A_minus=1),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=20),
    weight=plastic_weights)

for i in range(n_pops):
    stimulation.append(sim.Population(n_neurons, sim.SpikeSourcePoisson,
                       {'rate': stim_rate, 'duration': duration}, label="pre"))
    post_pops.append(sim.Population(
        n_neurons, sim.IF_curr_exp, cell_params, label='post'))
    plastic_projections.append(
        sim.Projection(stimulation[i], post_pops[i],
                       sim.OneToOneConnector(),
                       synapse_type=synapse_dynamics,
                       receptor_type='excitatory',
                       label='Pre-post projection'))
    post_pops[i].record('spikes')
    reward_projections.append(sim.Projection(
        reward_pop, post_pops[i], sim.OneToOneConnector(),
        synapse_type=sim.extra_models.Neuromodulation(
            weight=0.05, tau_c=100.0, tau_d=5.0, w_max=20.0),
        receptor_type='reward', label='reward synapses'))
    punishment_projections.append(sim.Projection(
        punishment_pop, post_pops[i], sim.OneToOneConnector(),
        synapse_type=sim.extra_models.Neuromodulation(
            weight=0.05, tau_c=100.0, tau_d=5.0, w_max=20.0),
        receptor_type='punishment', label='punishment synapses'))

sim.run(duration)

# Graphical diagnostics


def plot_spikes(_spikes, _title, _n_neurons):
    """
    Plots the spikes if there are any.

    :param list([int, quantity]) _spikes:
    :param str _title:
    :param int _n_neurons:
    """
    if _spikes is not None:
        pylab.figure(figsize=(15, 5))
        pylab.xlim((0, duration))
        pylab.ylim((0, (10 * _n_neurons) + 1))
        pylab.plot([i[1] for i in _spikes], [i[0] for i in _spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(_title)
    else:
        print("No spikes received")


post_spikes = []
weights = []

for i in range(n_pops):
    weights.append(plastic_projections[i].get('weight', 'list'))
    spikes = post_pops[i].get_data('spikes').segments[0].spiketrains
    for j in range(n_neurons):
        for x in spikes[j]:
            post_spikes.append([(i*n_neurons)+j+1, x])

plot_spikes(post_spikes, "post-synaptic neuron activity", n_neurons)
pylab.plot(rewards, [0.5 for x in rewards], 'g^')
pylab.plot(punishments, [0.5 for x in punishments], 'r^')
pylab.show()

print("Weights(Initial {plastic_weights})")
for x in weights:
    print(x)

sim.end()

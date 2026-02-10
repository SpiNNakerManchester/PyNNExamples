# Copyright (c) 2026 The University of Manchester
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
stdp_mad_recurrent_pre_stochastic_multiplicative
"""
import matplotlib.pyplot as plt
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel
# pylint: disable=wrong-spelling-in-comment

p.setup(timestep=1.0, min_delay=1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 100)

nSourceNeurons = 1  # number of input (excitatory) neurons
nExcitNeurons = 1   # number of excitatory neurons in the recurrent memory
nInhibNeurons = 10  # number of inhibitory neurons in the recurrent memory
nTeachNeurons = 1
runTime = 3200

cell_params_lif = {
    'cm': 0.25,        # nF was 0.25
    'i_offset': 0.0,
    'tau_m': 10.0,
    'tau_refrac': 2.0,
    'tau_syn_E': 0.5,
    'tau_syn_I': 0.5,
    'v_reset': -70.0,
    'v_rest': -70.0,
    'v_thresh': -50.0}

populations = list()
projections = list()

stimulus = 0
inhib = 1
excit = 2
teacher = 3

weight_to_force_firing = 15.0
baseline_excit_weight = 2.0

spikes0 = list()
teachingSpikes = list()
for i in range(runTime//40):
    spikes0.append(i*40)
for i in range(runTime//80):
    teachingSpikes.append(i*40+5+120)

arrayEntries = []
for i in range(nSourceNeurons):
    newEntry = []
    for spike in spikes0:
        newEntry.append(spike + i*40.0/100.0)
    arrayEntries.append(newEntry)
spikeArray = {'spike_times': arrayEntries}

teachlist = list()
for i in range(nSourceNeurons):
    teachlist.append(teachingSpikes)
teachingSpikeArray = {'spike_times': teachlist}
populations.append(p.Population(nSourceNeurons,
                                p.SpikeSourceArray(**spikeArray),
                                label='excit_pop_ss_array'))       # 0
populations.append(p.Population(nInhibNeurons,
                                p.IF_curr_exp(**cell_params_lif),
                                label='inhib_pop'))                # 1
populations.append(p.Population(nExcitNeurons,
                                p.IF_curr_exp(**cell_params_lif),
                                label='excit_pop'))                # 2
populations.append(p.Population(nTeachNeurons,
                                p.SpikeSourceArray(**teachingSpikeArray),
                                label='teaching_ss_array'))        # 3

stdp_model = p.STDPMechanism(
    timing_dependence=p.extra_models.RecurrentRule(
        accumulator_depression=-6, accumulator_potentiation=3,
        mean_pre_window=10.0, mean_post_window=10.0, dual_fsm=False,
        A_plus=0.2, A_minus=0.2),
    weight_dependence=p.MultiplicativeWeightDependence(w_min=0.0, w_max=16.0),
    weight=baseline_excit_weight, delay=1)

projections.append(
    p.Projection(populations[stimulus], populations[excit],
                 p.AllToAllConnector(), synapse_type=stdp_model))

projections.append(
    p.Projection(populations[teacher], populations[excit],
                 p.OneToOneConnector(), receptor_type='excitatory',
                 synapse_type=p.StaticSynapse(
                     weight=weight_to_force_firing, delay=1)))

populations[inhib].record(['v', 'spikes'])
populations[excit].record(['v', 'spikes'])

p.run(runTime)

final_weights = projections[0].get('weight', 'list', with_address=False)
print(f"Final weights: {final_weights}")

v = populations[excit].get_data('v')
spikes = populations[excit].get_data('spikes')
vInhib = populations[inhib].get_data('v')
spikesInhib = populations[inhib].get_data('spikes')

Figure(
    # plot of the neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runTime)),
    # membrane potential of the neurons
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[populations[excit].label], yticks=True,
          xlim=(0, runTime), xticks=True),
    title="Simple associative memory: spikes and membrane potential",
    annotations=f"Simulated with {p.name()}"
)
plt.show()

p.end()

# combined binaries [
# IF_curr_exp_stdp_mad_recurrent_pre_stochastic_multiplicative.aplx]

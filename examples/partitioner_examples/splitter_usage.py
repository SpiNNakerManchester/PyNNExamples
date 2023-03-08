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
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from pacman.model.partitioner_splitters import (
    SplitterOneToOneLegacy as OneToOneSplitter)
from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexFixed,
    SpynnakerSplitterFixedLegacy as LegacySplitter)

runtime = 1000
n_neurons = 100  # number of neurons in each population
weight_to_spike = 2.0  # weight to spike
delay = 17  # delay (above delay extension point)

p.setup(timestep=1.0, min_delay=1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, int(n_neurons / 2))

neuron = p.Population(
    n_neurons, p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterAbstractPopulationVertexFixed()})
neuron.record("all")
neuron2 = p.Population(
    int(n_neurons / 2), p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterAbstractPopulationVertexFixed()})
neuron3 = p.Population(
    n_neurons * 2, p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterAbstractPopulationVertexFixed()})
neuron4 = p.Population(
    int(n_neurons / 3), p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterAbstractPopulationVertexFixed()})
input1 = p.Population(
    n_neurons, p.SpikeSourcePoisson(), label='inputSpikes_1',
    additional_parameters={
        "splitter_object": LegacySplitter()})
input2 = p.Population(
    n_neurons, p.SpikeSourcePoisson(), label='inputSpikes_2',
    additional_parameters={
        "splitter_object": OneToOneSplitter()})

for pop in [neuron, neuron2, neuron3, neuron4]:
    for pop2 in [neuron, neuron2, neuron3, neuron4]:
        p.Projection(
            pop, pop2, p.FixedProbabilityConnector(p_connect=0.1),
            p.StaticSynapse(weight=weight_to_spike, delay=delay))
p.Projection(
    input1, neuron, p.OneToOneConnector(),
    p.StaticSynapse(weight=weight_to_spike, delay=delay))
p.Projection(
    input2, neuron, p.OneToOneConnector(),
    p.StaticSynapse(weight=weight_to_spike, delay=delay))

p.run(runtime)

# get data (could be done as one, but can be done bit by bit as well)
v = neuron.get_data('v')
gsyn_exc = neuron.get_data('gsyn_exc')
gsyn_inh = neuron.get_data('gsyn_inh')
spikes = neuron.get_data('spikes')
p.end()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[neuron.label], yticks=True, xlim=(0, runtime)),
    Panel(gsyn_exc.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[neuron.label], yticks=True, xlim=(0, runtime)),
    Panel(gsyn_inh.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=[neuron.label], yticks=True, xlim=(0, runtime)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

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
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

from examples.partitioner_examples.splitter_objects.\
    splitter_by_compartments_abstract_pop_vertex import \
    SplitterByCompartmentsAbstractPopVertex
from examples.partitioner_examples.splitter_objects.\
    splitter_by_fixed_atoms import SplitterByFixedAtom
from examples.partitioner_examples.splitter_objects.\
    splitter_by_slice_abstract_pop_vertex import \
    SplitterBySliceAbstractPopVertex
from examples.partitioner_examples.splitter_objects.\
    splitter_one_to_one import SplitterOneToOne

runtime = 1000
n_neurons = 100  # number of neurons in each population
weight_to_spike = 2.0  # weight to spike
delay = 17  # delay (above delay extension point)

p.setup(timestep=1.0, min_delay=1.0, max_delay=1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, int(n_neurons / 2))

neuron = p.Population(
    n_neurons, p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterBySliceAbstractPopVertex()})
neuron2 = p.Population(
    int(n_neurons / 2), p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterByCompartmentsAbstractPopVertex()})
neuron3 = p.Population(
    n_neurons * 2, p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterByFixedAtom()})
neuron4 = p.Population(
    int(n_neurons / 3), p.IF_curr_exp(), label='pop_1',
    additional_parameters={
        "splitter_object": SplitterOneToOne()})
input = p.Population(
    n_neurons, p.SpikeSourcePoisson(), label='inputSpikes_1')

for pop in [neuron, neuron2, neuron3, neuron4]:
    for pop2 in [neuron, neuron2, neuron3, neuron4]:
        p.Projection(
            pop, pop2, p.FixedProbabilityConnector(p_connect=0.1),
            p.StaticSynapse(weight=weight_to_spike, delay=delay))
p.Projection(
    input, neuron, p.OneToOneConnector(),
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

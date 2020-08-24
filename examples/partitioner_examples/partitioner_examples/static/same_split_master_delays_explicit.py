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

from examples.partitioner_examples.splitter_objects.\
    splitter_by_slice_abstract_pop_vertex import (
        SplitterBySliceAbstractPopVertex)


def main(plot):
    runtime = 1000
    n_neurons = 100  # number of neurons in each population
    weight_to_spike = 2.0  # weight to spike
    delay = 17  # delay (above delay extension point)

    p.setup(timestep=1.0, min_delay=1.0, max_delay=1.0)

    loop_connections = list()
    for i in range(0, n_neurons):
        single_connection = (i, (i + 1) % n_neurons, weight_to_spike, delay)
        loop_connections.append(single_connection)

    injection_connection = [(0, 0)]
    spike_array = {'spike_times': [[0]]}
    neuron = p.Population(
        n_neurons, p.IF_curr_exp(), label='pop_1',
        additional_parameters={
            "splitter_object": SplitterBySliceAbstractPopVertex()})
    input = p.Population(
        1, p.SpikeSourceArray(**spike_array), label='inputSpikes_1')

    p.Projection(
        neuron, neuron, p.FromListConnector(loop_connections),
        p.StaticSynapse(weight=weight_to_spike, delay=delay))
    p.Projection(
        input, neuron, p.FromListConnector(injection_connection),
        p.StaticSynapse(weight=weight_to_spike, delay=delay))

    neuron.record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])

    p.run(runtime)

    # get data (could be done as one, but can be done bit by bit as well)
    neuron.get_data('v')
    neuron.get_data('gsyn_exc')
    neuron.get_data('gsyn_inh')
    neuron.get_data('spikes')
    p.end()


if __name__ == '__main__':
    main()

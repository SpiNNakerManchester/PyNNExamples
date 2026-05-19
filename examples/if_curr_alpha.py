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
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt


def run_script(*, split: bool = True) -> None:
    """
    Runs the example script

    The default setting cause this script to split.

    :param split: If True will split the Populations that receive data
        into synapse and neuron cores.
        This requires more cores but allows more spikes to be received.
    """
    p.setup(0.1)
    runtime = 50
    populations = []
    title = "PyNN alpha synapse testing"

    pop_src1 = p.Population(1, p.SpikeSourceArray,
                            {'spike_times': [[5, 15, 20, 30]]}, label="src1")
    if split:
        # Due to the timestep of 0.1 by default
        # this splits into synapses and neuron cores
        populations.append(p.Population(1, p.IF_curr_alpha, {}, label="test"))
    else:
        # can be forced to not split this way
        populations.append(p.Population(1, p.IF_curr_alpha, {}, label="test",
                                        n_synapse_cores=0))

    populations[0].set(tau_syn_E=2)
    populations[0].set(tau_syn_I=4)

    # define the projections
    p.Projection(
        pop_src1, populations[0], p.OneToOneConnector(),
        p.StaticSynapse(weight=1, delay=1), receptor_type="excitatory")
    p.Projection(
        pop_src1, populations[0], p.OneToOneConnector(),
        p.StaticSynapse(weight=1, delay=10),  receptor_type="inhibitory")

    populations[0].record("all")
    p.run(runtime)

    v = populations[0].get_data("v")
    gsyn_exc = populations[0].get_data("gsyn_exc")
    gsyn_inh = populations[0].get_data("gsyn_inh")
    spikes = populations[0].get_data("spikes")
    print(spikes)

    plot.Figure(
        plot.Panel(v.segments[0].filter(name='v')[0],
                   ylabel="Membrane potential (mV)",
                   data_labels=[populations[0].label],
                   yticks=True, xlim=(0, runtime)),
        plot.Panel(gsyn_exc.segments[0].filter(name='gsyn_exc')[0],
                   ylabel="gsyn excitatory (mV)",
                   data_labels=[populations[0].label],
                   yticks=True, xlim=(0, runtime)),
        plot.Panel(gsyn_inh.segments[0].filter(name='gsyn_inh')[0],
                   ylabel="gsyn inhibitory (mV)",
                   data_labels=[populations[0].label],
                   yticks=True, xlim=(0, runtime)),
        title=title,
        annotations=f"Simulated with {p.name()}"
    )
    plt.show()
    p.end()

# combined binaries [IF_curr_alpha.aplx]
# split binaries [IF_curr_alpha_neuron.aplx, synapses.aplx]


if __name__ == "__main__":
    run_script()

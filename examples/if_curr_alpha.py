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

p.setup(0.1)
runtime = 50
populations = []
title = "PyNN0.8 alpha synapse testing"

pop_src1 = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': [[5, 15, 20, 30]]}, label="src1")

populations.append(p.Population(1, p.IF_curr_alpha, {}, label="test"))

populations[0].set(tau_syn_E=2)
populations[0].set(tau_syn_I=4)

# define the projections
exc_proj = p.Projection(pop_src1, populations[0],
                        p.OneToOneConnector(),
                        p.StaticSynapse(weight=1, delay=1),
                        receptor_type="excitatory")
inh_proj = p.Projection(pop_src1, populations[0],
                        p.OneToOneConnector(),
                        p.StaticSynapse(weight=1, delay=10),
                        receptor_type="inhibitory")

populations[0].record("all")
p.run(runtime)

v = populations[0].get_data("v")
gsyn_exc = populations[0].get_data("gsyn_exc")
gsyn_inh = populations[0].get_data("gsyn_inh")
spikes = populations[0].get_data("spikes")

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

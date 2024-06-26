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

# A simple script that compares the spikes from two inputs to determine if
# one spiked just before or after the other.

import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt

sim.setup(timestep=1.0)

# A population that will only spike if it receives exactly two spikes at
# exactly the same time.
# NOTE: Weird parameters tweaked to get the desired result
# this set of parameters require very high weight so are NOT ideal
pop_1 = sim.Population(11, sim.IF_curr_exp(
    tau_syn_E=1, tau_refrac=0,  tau_m=1), label="pop_1")

# Two population that spike at slightly different times
input_1 = sim.Population(
    1, sim.SpikeSourceArray(spike_times=[1, 21, 42, 61, 84]), label="input")
input_2 = sim.Population(
    1, sim.SpikeSourceArray(spike_times=[1, 22, 41, 57, 81]), label="input")

# One projection which sends the spikes to different neurons
# with a range of delays
input_proj = sim.Projection(
    input_1, pop_1, sim.FromListConnector(
        [(0, i, 20, i + 1) for i in range(pop_1.size)]),
    synapse_type=sim.StaticSynapse())
# One projection that always sends to all neurons with the average delay
input_proj = sim.Projection(input_2, pop_1, sim.AllToAllConnector(),
                            synapse_type=sim.StaticSynapse(weight=20, delay=6))

# Request to record data
pop_1.record(["spikes", "v"])

# run
simtime = 100
sim.run(simtime)

# get the Data out in PyNN's Neo format
neo = pop_1.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
print(spikes)
v = neo.segments[0].filter(name='v')[0]
print(v)

# End the simulation
sim.end()

# Plot
plot.Figure(
    # plot voltage for first ([0]) neuron
    plot.Panel(v, ylabel="Membrane potential (mV)",
               data_labels=[pop_1.label], yticks=True, xlim=(0, simtime)),
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
    title="Simple Example",
    annotations=f"Simulated with {sim.name()}"
)
plt.show()

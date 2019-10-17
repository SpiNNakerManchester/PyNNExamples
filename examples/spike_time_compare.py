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

# A simple script that compares the spikes from two inputs to determine if
# one spiked just before or after the other.

import spynnaker8 as sim
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
    input_1, pop_1, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(
        weight=20, delay=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
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
    annotations="Simulated with {}".format(sim.name())
)
plt.show()

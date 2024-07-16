# Copyright (c) 2016 The University of Manchester
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

"""
Synfire chain example
"""
import matplotlib.pyplot as plt
import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot

# number of neurons in each population
n_neurons = 10
simtime = 1000
pops = 5

sim.setup(timestep=1.0, min_delay=1.0)

spikeArray = {'spike_times': [[0]]}
stimulus = sim.Population(1, sim.SpikeSourceArray, spikeArray,
                          label='stimulus')

pop = sim.Population(n_neurons, sim.IF_curr_exp, {}, label='chain')
pop.record("spikes")
#pop.record(["spikes", "v"])
sim.Projection(stimulus, pop,
               sim.OneToOneConnector(),
               sim.StaticSynapse(weight=5, delay=1))
sim.Projection(pop[n_neurons - 1], pop[0], sim.OneToOneConnector(),
               sim.StaticSynapse(weight=5, delay=1))
for i in range(n_neurons - 1):
    print(i, i+1)
    sim.Projection(pop[i], pop[i], sim.OneToOneConnector(),
                   sim.StaticSynapse(weight=5, delay=1))

sim.run(simtime)
neo = pop.get_data(variables=["spikes"])
#neo = pop.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
print(spikes)
# v = neo.segments[0].filter(name='v')[0]
# print(v)
sim.end()

plot.Figure(
    # plot voltage for first ([0]) neuron
    # plot.Panel(v, ylabel="Membrane potential (mV)",
    #           data_labels=[pop.label], yticks=True, xlim=(0, simtime)),
    # plot spikes
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
    title="Self Synfire Example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()
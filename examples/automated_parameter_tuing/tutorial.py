import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from pyNN.random import RandomDistribution


def plot_spiketrains(spiketrains):
    for spiketrain in spiketrains[spiketrain]:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel('Spiketrain %d' %spiketrain)
        plt.setp(plt.gca().get_xticklabels(), visible=False)

sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,100)

pop_1 = sim.Population(100, sim.IF_curr_exp(), label="pop_1")
input = sim.Population(1, sim.SpikeSourceArray(spike_times=[0]), label="input")
input_proj = sim.Projection(input, pop_1, sim.FromListConnector([(0, 0)]), 
                            synapse_type=sim.StaticSynapse(weight=5, delay=1))

size = 99
connection_list =[(size, 0)]
for i in range(0, size+1):
    new_connection = (i,i+1)
    connection_list.append(new_connection)
    
#print(connection_list)
randomise = RandomDistribution('normal_clipped_to_boundary', mu=i, sigma=size/5, low=0, high=size)
syn_fire_proj = sim.Projection(pop_1, pop_1, sim.FromListConnector(connection_list), synapse_type=sim.StaticSynapse(weight=5, delay=1))
pop_1.record(["spikes"])
simtime = 2000
sim.run(simtime)
neo = pop_1.get_data(variables=["spikes"])
spikes = neo.segments[0].spiketrains
print spikes
'''v = neo.segments[0].filter(name='v')[0]
print v
'''
sim.end()

plot.Figure(
# plot voltage for first ([0]) neuron
#plot.Panel(v, ylabel="Membrane potential (mV)",
#data_labels=[pop_1.label], yticks=True, xlim=(0, simtime)),
# plot spikes (or in this case spike)
plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
title="Simple Example",
annotations="Simulated with {}".format(sim.name())
)
plt.show()

import spynnaker8 as sim
from pyNN.connectors import OneToOneConnector, AllToAllConnector,\
    FromListConnector
import matplotlib.pyplot as plt
import pyNN.utility.plotting as plot
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
from elephant.statistics import mean_firing_rate, instantaneous_rate, kernels
import quantities as pq 
from max_pooler import plot_spiketrains
        
input_parameters = [10, 20, 10]
number_inputs = len(input_parameters)

#0.092 gives good average
standard_weight = 0.092
normalised_weight = standard_weight - (standard_weight/number_inputs)

delay = 1
simtime = 10000

responsive_neuron = sim.IF_cond_exp()#tau_syn_E=0.2, tau_syn_I=0.2)

inputs = []

sim.setup()

# setting up inputs
for i in range(number_inputs):
    inputs.append(sim.Population(1, sim.SpikeSourcePoisson(rate=input_parameters[i], ), label=("input_"+str(i))))
    inputs[i].record(["spikes"])

# setting up filter layer (repeats inputs with some inhibition)
filter_layer = sim.Population(number_inputs, responsive_neuron, label="filter_layer")

#connecting inputs to filter layer
for i in range(number_inputs):
    input_connection = [(0, i, standard_weight, delay)]
    input_proj = sim.Projection(inputs[i], filter_layer, sim.FromListConnector(input_connection))

# output neuron
OR_neuron = sim.Population(1, responsive_neuron, label="OR_neuron")

filter_proj = sim.Projection(filter_layer, OR_neuron, sim.AllToAllConnector(), sim.StaticSynapse(weight=normalised_weight, delay=delay))

OR_neuron.record(["spikes"])
filter_layer.record(["spikes"])

sim.run(simtime)

#getting data
data = OR_neuron.get_data(variables=["spikes"])
for i in range(number_inputs):
    data.segments[0].spiketrains.extend(inputs[i].get_data(variables=["spikes"]).segments[0].spiketrains)    
data.segments[0].spiketrains.extend(filter_layer.get_data(variables=["spikes"]).segments[0].spiketrains)

sim.end()

plot_spiketrains(data.segments[0])

plt.show()
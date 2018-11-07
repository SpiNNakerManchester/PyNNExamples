import spynnaker8 as sim
from pyNN.connectors import OneToOneConnector, AllToAllConnector,\
    FromListConnector
import matplotlib.pyplot as plt
import pyNN.utility.plotting as plot
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
from elephant.statistics import mean_firing_rate
import quantities as pq 

def plot_spiketrains(segment):
    #Adapted from http://neuralensemble.org/docs/PyNN/data_handling.html
    uniq = list(set([spiketrain.annotations['source_population'] for spiketrain in segment.spiketrains]))
    hot = plt.get_cmap('hot')
    cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    
    def which_colour(source_population):
        colours = {"input": "red", "output": "blue", "filter": "green"}
        
        if source_population.find('input') != -1:
            return colours["input"]
        
        if source_population.find('filter') != -1:
            return colours["filter"]
        
        if source_population.find('OR') != -1:
            return colours["output"]
            
    
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.', c=which_colour(spiketrain.annotations['source_population']))
        firing_rate = mean_firing_rate(spiketrain)
        firing_rate.units = pq.hertz
        print(firing_rate)
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)

input_parameters = [20, 20, 40, 20]
number_inputs = len(input_parameters)

inputs = []

simtime = 1000

sim.setup()

# setting up inputs
for i in range(number_inputs):
    inputs.append(sim.Population(1, sim.SpikeSourcePoisson(rate=input_parameters[i]), label=("input_"+str(i))))
    inputs[i].record(["spikes"])

# setting up filter layer (repeats inputs with some inhibition)
filter_layer = sim.Population(number_inputs, sim.IF_curr_exp, label="filter_layer")
forward_inh_pop = sim.Population(number_inputs, sim.IF_curr_exp, label="forward_inh_pop")
#connecting inputs to filter layer
for i in range(number_inputs):
    input_connection = [(0, i, 5, 2)]
    input_proj = sim.Projection(inputs[i], filter_layer, sim.FromListConnector(input_connection))
    input_forward_inh_connection = [(0, i, 5, 1)]
    input_forward_inh_proj = sim.Projection(inputs[i], forward_inh_pop, sim.FromListConnector(input_forward_inh_connection))

#feedforward inhibition
forward_filter_inh_connections = [(i, j, 5, 1) for i in range(number_inputs) for j in range(number_inputs) if i!=j]
forward_filter_inh_proj = sim.Projection(forward_inh_pop, filter_layer, sim.FromListConnector(forward_filter_inh_connections), receptor_type='inhibitory')

#feedback inhibition    
back_inh_proj = sim.Projection(filter_layer, forward_inh_pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=10, delay=1))


#lateral inhibition
lat_inh_connection = [(i, j, 5, 1) for i in range(number_inputs) for j in range(number_inputs) if i!=j]
lat_inh_proj = sim.Projection(forward_inh_pop, forward_inh_pop, sim.FromListConnector(lat_inh_connection), receptor_type='inhibitory')



# output neuron
OR_neuron = sim.Population(1, sim.IF_curr_exp, label="OR_neuron")

#recurrent_connection = [(i,i,10,1) for i in range(number_inputs)]
#recurrent_proj = sim.Projection(filter_layer, filter_layer, sim.FromListConnector(recurrent_connection))


#filter_reccurrent_proj = sim.Projection(filter_layer, filter_layer, sim.OneToOneConnector(), sim.StaticSynapse(weight=10, delay=10))
filter_proj = sim.Projection(filter_layer, OR_neuron, sim.AllToAllConnector(), sim.StaticSynapse(weight=5, delay=1))
#filter_to_inh_proj = sim.Projection(filter_layer, lat_inh_pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=5, delay=1))
#getting lateral inhibition set up

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
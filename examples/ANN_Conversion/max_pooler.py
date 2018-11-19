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
        print(("Neuron ID: %s, Number of Spikes: %s, Average: %s") % (spiketrain.annotations['source_id'], len(spiketrain), firing_rate))
        #print(spiketrain)
        '''if firing_rate != 0:
            inst_firing_rate = instantaneous_rate(spiketrain, 100*pq.ms)
            inst_firing_rate = pq.hertz
            print(inst_firing_rate)
        '''        
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        
input_parameters = [15, 100, 15, 15]
number_inputs = len(input_parameters)


# At 4 it works well
standard_weight = 0.075
strong_weight = 0.14
normalised_weight = standard_weight/number_inputs
strong_normalised_weight = strong_weight/number_inputs

delay = 1

simtime = 1000

responsive_neuron = sim.IF_cond_exp()#tau_syn_E=0.2, tau_syn_I=0.2)


inputs = []



sim.setup()

# setting up inputs
for i in range(number_inputs):
    inputs.append(sim.Population(1, sim.SpikeSourcePoisson(rate=input_parameters[i], ), label=("input_"+str(i))))
    inputs[i].record(["spikes"])

# setting up filter layer (repeats inputs with some inhibition)
filter_layer = sim.Population(number_inputs, responsive_neuron, label="filter_layer")
forward_inh_pop = sim.Population(number_inputs, responsive_neuron, label="forward_inh_pop")
#connecting inputs to filter layer
for i in range(number_inputs):
    input_connection = [(0, i, standard_weight, delay)]
    input_proj = sim.Projection(inputs[i], filter_layer, sim.FromListConnector(input_connection))
    #feedforward inhibition
    input_forward_inh_connection = [(0, i, normalised_weight, delay)]
    #input_forward_inh_proj = sim.Projection(inputs[i], forward_inh_pop, sim.FromListConnector(input_forward_inh_connection))

#feedback inhibition    
back_inh_proj = sim.Projection(filter_layer, forward_inh_pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=standard_weight, delay=delay))

#inhibition connection
inh_connections = [(i, j, strong_weight, 1) for i in range(number_inputs) for j in range(number_inputs) if i!=j]
inh_filter_proj = sim.Projection(forward_inh_pop, filter_layer, sim.FromListConnector(inh_connections), receptor_type='inhibitory')

# output neuron
OR_neuron = sim.Population(1, responsive_neuron, label="OR_neuron")

#filter_reccurrent_proj = sim.Projection(filter_layer, filter_layer, sim.OneToOneConnector(), sim.StaticSynapse(weight=normalised_weight, delay=delay))
filter_proj = sim.Projection(filter_layer, OR_neuron, sim.AllToAllConnector(), sim.StaticSynapse(weight=standard_weight, delay=delay))

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
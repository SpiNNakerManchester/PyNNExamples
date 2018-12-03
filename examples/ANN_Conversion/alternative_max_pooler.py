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

    
def make_input(input_parameters, neuron=sim.IF_cond_exp(), passthrough_weight=0.071, delay = 1):
    number_inputs = len(input_parameters)
    inputs = []
    #setting up input populations
    for i in range(number_inputs):
        inputs.append(sim.Population(1, sim.SpikeSourcePoisson(rate=input_parameters[i], ), label=("input_"+str(i))))
        inputs[i].record(["spikes"])
    
    filter_layer = sim.Population(number_inputs, neuron, label="filter_layer")
    
    #connecting inputs to filter layer
    for i in range(number_inputs):
        input_connection = [(0, i, passthrough_weight, delay)]
        input_proj = sim.Projection(inputs[i], filter_layer, sim.FromListConnector(input_connection))
    
    return filter_layer, inputs

def make_output(input_layer, neuron=sim.IF_cond_exp()):
    # output neuron
    OR_neuron = sim.Population(1, neuron, label="OR_neuron")
    
    return OR_neuron

def create_layers(input_layer, output_layer, neuron = sim.IF_cond_exp(), passthrough_weight = 0.071, delay = 1):
    number_inputs = input_layer.size
    #connecting input to output
    #modify for multiple outputs
    filter_proj = sim.Projection(input_layer, output_layer, sim.AllToAllConnector(), sim.StaticSynapse(weight=passthrough_weight, delay=delay*2))
    inh_pop = sim.Population(number_inputs, neuron, label="inh_pop")
    
    #forward inhibitory connections
    inh_connections = [(i, j, passthrough_weight, delay) for j in range(number_inputs) for i in range(number_inputs) if i != j]
    input_inh_proj = sim.Projection(input_layer, inh_pop, sim.FromListConnector(inh_connections), sim.StaticSynapse())
    inh_output_proj = sim.Projection(inh_pop, output_layer, sim.OneToOneConnector(), sim.StaticSynapse(weight=passthrough_weight/number_inputs, delay=delay), receptor_type = "inhibitory")
        
    
    new_layers = [inh_pop]
    new_connections = [filter_proj, input_inh_proj, inh_output_proj]
    
    return new_layers, new_connections
    
def get_output_data(data, inputs, filter_layer):
    number_inputs = len(inputs)
    #setting up data record
    
    for i in range(number_inputs):
        data.segments[0].spiketrains.extend(inputs[i].get_data(variables=["spikes"]).segments[0].spiketrains)    
    
    data.segments[0].spiketrains.extend(filter_layer.get_data(variables=["spikes"]).segments[0].spiketrains)

    return

def main():        
    layers = []
    simtime = 10000
       
    sim.setup()
    input, inputs = make_input([15, 100, 15, 15])
    output = make_output(input)
    layers.append(input)
    layers.append(output)
    
    new_layers, new_connections = create_layers(input, output)
    
    output.record(["spikes"])
    input.record(["spikes"])
    
    sim.run(simtime)

    data = output.get_data(variables=["spikes"])
    get_output_data(data, inputs, input)
    
    sim.end()
    
    plot_spiketrains(data.segments[0])
    
    plt.show()
    
if __name__ == '__main__':
    main()
from __future__ import division
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
import math

def plot_error(errors):
    #plot the errors 
    return
    
    
def get_error(stats):
    firing_rates = stats[0] 
    spike_numbers = stats[1]
    rate_error = abs(firing_rates["input"]-firing_rates["pop"])
    spike_number_error =  abs(spike_numbers["input"]-spike_numbers["pop"])
    total_spikes = spike_numbers["input"]+spike_numbers["pop"]
    relative_spike_error = spike_number_error/total_spikes
    print("Absolute Error: %.4f, Relative Error: %.6f" % (spike_number_error, relative_spike_error))
    return relative_spike_error

def get_stats(segment):
    firing_rates = {}
    spike_numbers = {}
    for spiketrain in segment.spiketrains:      
        firing_rate = mean_firing_rate(spiketrain)
        firing_rate.units = pq.hertz
        firing_rates[spiketrain.annotations['source_population']] = firing_rate
        spike_numbers[spiketrain.annotations['source_population']] = len(spiketrain)
    return firing_rates, spike_numbers

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
    plt.show()

def run_sim(weight, plot=False):

    input_parameters = [100]
    standard_weight = weight
    delay = 1
    responsive_neuron = sim.IF_cond_exp()#tau_syn_E=0.2, tau_syn_I=0.2)
   
    simtime = 10000
    sim.setup()
    input = sim.Population(1, sim.SpikeSourcePoisson(rate=input_parameters[0]), label="input")
    pop = sim.Population(1, responsive_neuron, label="pop")
    proj = sim.Projection(input, pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=standard_weight, delay=delay))
        
    input.record(["spikes"])
    pop.record(["spikes"])

    sim.run(simtime)
    
    #getting data
    data = input.get_data(variables=["spikes"])
    data.segments[0].spiketrains.extend(pop.get_data(variables=["spikes"]).segments[0].spiketrains)
    
    sim.end()
    if plot:
        plot_spiketrains(data.segments[0])
        
    stats = get_stats(data.segments[0])
    error = get_error(stats)
        
    return error

def gs_search():
    #adapted from https://en.wikipedia.org/wiki/Golden-section_search
    
    a = 0.06
    b = 0.1
    tol = 0.00001
    gr = (math.sqrt(5) + 1) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr 
    while abs(c - d) > tol:
        if run_sim(c) < run_sim(d):
            b = d
            print("Lower region selected. Minimum between %s and %s" % (a,b))
        else:
            a = c
            print("Upper region selected. Minimum between %s and %s" % (a,b))
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    point = (b + a) / 2
    print(point)

    return 

print(gs_search())



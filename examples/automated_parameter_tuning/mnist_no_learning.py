'''This code sets up a network that has no learning but allows weights to be inputted. Unfinished'''

import focal as focal
import poisson as poisson

import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution

import scipy
import numpy
import pylab
from pyNN.connectors import AllToAllConnector
from spynnaker.pyNN.models.spike_source import spike_source_array

#functions

def generate_input_st(train_time, rest_time):
    '''Generating the spiketrain from the stock image'''
   
    filename  = NE15_path + "/t10k-images-idx3-ubyte__idx_000__lbl_7_.png"
    img = pylab.imread(filename)
    height, width = img.shape
    
    max_freq = 1000 #Hz
    
    spikes = poisson.mnist_poisson_gen(numpy.array([img.reshape(height*width)]), height, width, max_freq, train_time, rest_time)
    
    return spikes;
 

def plot_spiketrains(spiketrains):
    '''plotting spike trains'''
    for spiketrain in spiketrains[spiketrain]:
        y = numpy.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel('Spiketrain %d' %spiketrain)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
    return;


NE15_path = '/home/edwardjones/git/NE15'
filename  = NE15_path + '/t10k-images-idx3-ubyte__idx_000__lbl_7_.png'
img = pylab.imread(filename)
height, width = img.shape
train_time = 200
rest_time = 100
test_time = 200
simtime = 2000 #train_time + rest_time + test_time
timestep = 0.1



training_spikes = generate_input_st(train_time, rest_time)
input_size = len(training_spikes)
#input_labels = generate_label_st(7, train_time)
test_spikes = generate_input_st(test_time, 0)
input_spikes = [[]]
'''
for i in range(len(training_spikes)):
    input_spikes[i] = numpy.append(training_spikes[i], (test_spikes[i]))
'''

sim.setup(timestep)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,80)

input_pop = sim.Population(input_size, sim.SpikeSourceArray(input_spikes), label="input")
pop_1 = sim.Population(10, sim.IF_curr_exp(), label="pop_1")
output_pop = sim.Population(10, sim.IF_curr_exp(), label="output_pop")



input_proj = sim.Projection(input_pop, pop_1, sim.AllToAllConnector(allow_self_connections=False), 
                            synapse_type=sim.StaticSynapse(weight = 5, delay=1))

output_proj = sim.Projection(pop_1, output_pop, sim.AllToAllConnector(),
                                 synapse_type=sim.StaticSynapse(weight = 5, delay=1))



input_pop.record(["spikes"])
output_pop.record(["spikes"])


sim.run(simtime)

input_neo = input_pop.get_data(variables=["spikes"])
pop_1_neo = pop_1.get_data(variables=["spikes"])

input_spikes = input_neo.segments[0].spiketrains
pop_1_spikes = pop_1_neo.segments[0].spiketrains

sim.end()

plot.Figure(
plot.Panel(input_spikes, yticks=True, markersize=5, xlim=(0, simtime)),
#plot.Panel(input_labels, yticks=True, markersize=5, xlim=(0, simtime)),
plot.Panel(pop_1_spikes, yticks=True, markersize=5, xlim=(0, simtime)),
title="Simple Example",
annotations="Simulated with {}".format(sim.name())
)
plt.show()
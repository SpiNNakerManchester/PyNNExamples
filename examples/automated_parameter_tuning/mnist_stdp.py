import focal as focal
import poisson as poisson

import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution
import random

import scipy
import numpy
import pylab
from pyNN.connectors import AllToAllConnector
from spynnaker.pyNN.models.spike_source import spike_source_array

#functions

def generate_input_st(train_time, rest_time):
    '''Generating the spiketrain from image'''
   
    filename  = NE15_path + "/t10k-images-idx3-ubyte__idx_000__lbl_7_.png"
    img = pylab.imread(filename)
    height, width = img.shape
    
    max_freq = 1000 #Hz
    
    spikes = poisson.mnist_poisson_gen(numpy.array([img.reshape(height*width)]), height, width, max_freq, train_time, rest_time)
    
    return spikes;

def generate_label_st(number, duration):
    '''Generating training label spiketrain for given number'''

    output = [[],[],[],[],[],[],[],[],[],[]]
    output[number] = numpy.repeat(100, duration*timestep)
    return output;
    

def plot_spiketrains(spiketrains):
    for spiketrain in spiketrains[spiketrain]:
        y = numpy.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel('Spiketrain %d' %spiketrain)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
    return;
        
# variable declaration
NE15_path = '/home/edwardjones/git/NE15'
img = pylab.imread(filename)
height, width = img.shape
train_time = 200
rest_time = 100
test_time = 200
simtime = train_time + rest_time + test_time
timestep = 0.1
#main code



training_spikes = generate_input_st(train_time, rest_time)
input_size = len(training_spikes)
input_labels = generate_label_st(7, train_time)
test_spikes = generate_input_st(test_time, 0)
input_spikes = [[]]

for i in range(len(training_spikes)):
    input_spikes[i] = numpy.append(training_spikes[i], (test_spikes[i]))



sim.setup(timestep)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,80)

input_pop = sim.Population(input_size, sim.SpikeSourceArray(input_spikes), label="input")
pop_1 = sim.Population(10, sim.IF_curr_exp(), label="pop_1")
output_pop = sim.Population(10, sim.IF_curr_exp(), label="output_pop")
teach_pop = sim.Population(10, sim.IF_curr_exp(), sim.SpikeSourceArray(input_labels), label="teach_pop")

random_weights = RandomDistribution('normal_clipped', mu=0.1, sigma=0.1, low=0, high=10)

input_proj = sim.Projection(input_pop, pop_1, sim.AllToAllConnector(allow_self_connections=False), 
                            synapse_type=sim.StaticSynapse(weight = 5, delay=1))

timing_rule = sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                A_plus=0.5, A_minus=0.5)
weight_rule = sim.AdditiveWeightDependence(w_max=5.0, w_min=0.0)

stdp_model = sim.STDPMechanism(timing_dependence=timing_rule,
                               weight_dependence=weight_rule,
                               weight=0.0, delay=5.0)

output_proj = sim.Projection(pop_1, output_pop, sim.AllToAllConnector(),
                                 synapse_type=stdp_model)

teach_proj = sim.Projection(teach_pop, output_proj, sim.OneToOneConnector(),
                            synapse_type=sim.StaticSynapse(weight=100, delay=1))


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
plot.Panel(input_labels, yticks=True, markersize=5, xlim=(0, simtime)),
plot.Panel(pop_1_spikes, yticks=True, markersize=5, xlim=(0, simtime)),
title="Simple Example",
annotations="Simulated with {}".format(sim.name())
)
plt.show()
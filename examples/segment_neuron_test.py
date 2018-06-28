import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

#================================================================================================
# Simulation parameters
#================================================================================================
model = sim.IF_curr_exp
target_cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 1.0,#2.5,
               'tau_syn_I': 1.0,#2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

w2s_target = 5.

connection_weight = w2s_target#/2.
number_of_inputs = 1

input_spikes =[]

for neuron in range(number_of_inputs):
    input_spikes.append([i*20 for i in range(10)])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,64)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes))
cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params,label="fixed_weight_scale")

cd_pop.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))

duration = max(input_spikes[0])

sim.run(duration)

cd_data = cd_pop.get_data(["spikes"])

sim.end()

spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="cd pop activity")
plt.show()
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

cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 5.,#10.0,#2.,#3.,#
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 3.0,#2.5,#
               'tau_syn_I': 2.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

w2s =2.
winh = 1.0#0.5

input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes/"
stim_times = numpy.load(input_directory+'ic_spikes_interleaved_sweep.npy')
max_time = 0
for times in stim_times:
    for time in times:
        if time>max_time:
            max_time=time.item()

input_size = len(stim_times)
hidden_size = 10*input_size
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=14.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,32)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(input_size,sim.SpikeSourceArray(spike_times=stim_times))
hidden_pop = sim.Population(hidden_size,sim.IF_curr_exp,cell_params,label="hidden_pop")
output_pop = sim.Population(input_size,sim.IF_curr_exp,cell_params,label="output_pop")

input_pop.record(["spikes"])
hidden_pop.record(["spikes"])
output_pop.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
av_n_inputs_hidden = 10.
p_connect_exc = av_n_inputs_hidden/input_size
weight_dist_exc = RandomDistribution('uniform',(0.,w2s/2))
input_hidden_projection = sim.Projection(input_pop,hidden_pop,sim.FixedProbabilityConnector(p_connect=p_connect_exc),
                                  synapse_type=sim.StaticSynapse(weight=weight_dist_exc,delay=1.))

p_connect_inh = 0.1
weight_dist_inh = RandomDistribution('uniform',(0.,winh))
hidden_hidden_inh_projection = sim.Projection(hidden_pop,hidden_pop,sim.FixedProbabilityConnector(p_connect=p_connect_inh),
                                              synapse_type=sim.StaticSynapse(weight=weight_dist_inh),receptor_type='inhibitory')

p_connect_exc_output = av_n_inputs_hidden/hidden_size
weight_dist_exc_output = RandomDistribution('uniform',(0.,w2s))
hidden_output_projection = sim.Projection(hidden_pop,output_pop,sim.FixedProbabilityConnector(p_connect=p_connect_exc_output),
                                  synapse_type=sim.StaticSynapse(weight=weight_dist_exc_output,delay=1.))

#TODO: add hard wta inh between output layer neurons

#================================================================================================
#  Run simuluation
#================================================================================================
duration = 5000#max_time
sim.run(duration)
hidden_data = hidden_pop.get_data(["spikes"])
output_data = output_pop.get_data(["spikes"])

sim.end()

#================================================================================================
# Analysis and result export
#================================================================================================
spike_raster_plot_8(stim_times,plt,duration/1000.,input_size+1,0.001,title="input pop activity")
spike_raster_plot_8(hidden_data.segments[0].spiketrains,plt,duration/1000.,hidden_size+1,0.001,title="hidden pop activity")
spike_raster_plot_8(output_data.segments[0].spiketrains,plt,duration/1000.,input_size+1,0.001,title="output pop activity")

plt.show()
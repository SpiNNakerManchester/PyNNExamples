import spynnaker8 as sim
import numpy as np
import pylab as plt
import sys
sys.path.append("../") 
from signal_prep import spike_raster_plot_8,get_sub_pop_spikes,sub_pop_builder_auto,normal_dist_connection_builder
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint
import time as local_time

#================================================================================================
# Simulation parameters
#================================================================================================

duration = 1000.
n_input = 10000.
n_per_core = 255

input_size = n_per_core*np.ceil(n_input/n_per_core)
target_pop_size = n_per_core*np.ceil((n_input * 2./3)/n_per_core)
inh_pop_size = n_per_core*np.ceil((n_input * 1./3)/n_per_core)

input_spikes=np.load('./input_spikes.npy').tolist()

spike_raster_plot_8(input_spikes,plt,duration/1000.,ylim=input_size+1,title="input_activity")

list_file_name = './conn_list_{}input_scaled_id.npz'.format(input_size)

connection_list_file = np.load(list_file_name)
source_target_list=connection_list_file['source_target_list']
source_inh_list=connection_list_file['source_inh_list']
target_target_list=connection_list_file['target_target_list']
target_inh_list=connection_list_file['target_inh_list']
inh_inh_list=connection_list_file['inh_inh_list']
inh_target_list=connection_list_file['inh_target_list']

timestep = 1.

time_start = local_time.time()
sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,n_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,n_per_core)

pop_size = max([input_size,target_pop_size,inh_pop_size])
source_pop = sim.Population(pop_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
target_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='target_pop_fixed_weight_scale_cond')
inh_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='inh_pop_fixed_weight_scale_cond')

target_pop.record('spikes')
inh_pop.record('spikes')
source_target_proj = sim.Projection(source_pop,target_pop,sim.FromListConnector(source_target_list),synapse_type=sim.StaticSynapse())
source_inh_proj = sim.Projection(source_pop,inh_pop,sim.FromListConnector(source_inh_list),synapse_type=sim.StaticSynapse())
target_lat_proj = sim.Projection(target_pop,target_pop,sim.FromListConnector(target_target_list),synapse_type=sim.StaticSynapse())
target_inh_proj = sim.Projection(target_pop,inh_pop,sim.FromListConnector(target_inh_list),synapse_type=sim.StaticSynapse())
inh_lat_proj = sim.Projection(inh_pop,inh_pop,sim.FromListConnector(inh_inh_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
inh_target_proj = sim.Projection(inh_pop,target_pop,sim.FromListConnector(inh_target_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')

max_period = 5000.
num_recordings =int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

target_data = target_pop.get_data(['spikes'])
output_spikes = target_data.segments[0].spiketrains

sim.end()

print "simulation of {}s complete in {}s".format(duration/1000.,local_time.time()-time_start)

spike_raster_plot_8(output_spikes,plt,duration/1000.,pop_size+1,0.001,title="output pop activity")

non_zero_spikes = [train for train in output_spikes if len(train)>0]
print "non zero output spikes length:{} target n_neurons:{}".format(len(non_zero_spikes),target_pop_size)

plt.show()

import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import spike_raster_plot_8,get_sub_pop_spikes,sub_pop_builder_auto,normal_dist_connection_builder,spatial_normal_dist_connection_builder
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint

def from_list_fixed_pre(pre_size,post_size,n_pre):
    conn_list = []
    for post in range(int(post_size)):
        pres = np.random.choice(range(int(pre_size)),n_pre,replace=False)
        for pre in pres:
            conn_list.append((pre,post))
    return conn_list

#================================================================================================
# Simulation parameters
#================================================================================================
duration = 1000.
using_fixed_n = False

input_spikes =[]
# for neuron in an_spikes[0]:
#     input_spikes.append([spike_time for spike_time in neuron if spike_time <= duration])
isi = 50
n_pres = int(duration/isi)
n_input = 16000.
n_per_core = 255
input_size = n_input#n_per_core*np.ceil(n_input/n_per_core)#len(input_spikes)#
n_total = input_size
target_pop_size = int(n_total*2./3)#n_per_core*np.ceil((n_total* 2./3)/n_per_core)#input_size#100.#int(np.round(n_total*10./89.)) #200
inh_pop_size = int(n_total*1./3)#n_per_core*np.ceil((n_total* 1./3)/n_per_core)

try:
    input_spikes=np.load("scale_test_fixed_prob_input_n={}.npy".format(int(n_input)))
except:
    source_firing_rate = 10. #Hz
    isi_ms = 1000./source_firing_rate
    n_repeats = int((duration/1000.)*source_firing_rate)

    for _ in range(int(input_size)):
        input_spikes.append([isi_ms*i + (np.random.rand())*(isi_ms/1.) for i in range(n_repeats)])
    np.save("scale_test_fixed_prob_input_n={}.npy".format(int(n_input)),input_spikes)

# input_spikes=[[1.],[10.,15.]]
spike_raster_plot_8(input_spikes,plt,duration/1000.,ylim=input_size+1,title="input_activity")

n_connections = 10.#input_size/100.#int(0.01*input_size)#76#10#
# n_connections = RandomDistribution('uniform',[30.,120.])

w2s_target = .1#7.#
# av_weight =(w2s_target/(0.5*n_connections))#(w2s_target/50.)#w2s_target#(w2s_target/25.)#
# connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])
# av_weight =(w2s_target/25.)#/(n_connections*0.005))#(w2s_target/50.)#w2s_target#(w2s_target/25.)#

total_n_neurons =input_size + target_pop_size + inh_pop_size
from_i_n_connections = int(n_connections * (float(inh_pop_size)/total_n_neurons))
from_input_n_connections = int(n_connections * (float(input_size)/total_n_neurons))
from_target_n_connections = int(n_connections * (float(target_pop_size)/total_n_neurons))

exc_ratio = 1.
inh_ratio = 0.1#0.05
av_lat_exc_w =(w2s_target/(from_target_n_connections*exc_ratio))
# av_lat_exc_w =(w2s_target/(from_target_n_connections))
lat_exc_w = RandomDistribution('uniform',[0,av_lat_exc_w*2.])
# lat_exc_w = RandomDistribution('normal_clipped',[av_lat_exc_w,av_lat_exc_w/10.,0,av_lat_exc_w*2.])
av_lat_inh_w = (w2s_target/(from_i_n_connections*inh_ratio))
lat_inh_w = RandomDistribution('uniform',[0,av_lat_inh_w*2.])
# lat_inh_w = RandomDistribution('normal_clipped',[av_lat_inh_w,av_lat_inh_w/10.,0,av_lat_inh_w*2.])
av_weight =(w2s_target/(from_input_n_connections*exc_ratio))
connection_weight = RandomDistribution('uniform',[0,av_weight*2.])
# connection_weight =RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])

timestep = 1.#0.1

sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,n_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,n_per_core)

source_pop = sim.Population(input_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
target_pop = sim.Population(target_pop_size,sim.IF_cond_exp,{},label='target_pop_fixed_weight_scale_cond')
inh_pop = sim.Population(inh_pop_size,sim.IF_cond_exp,{},label='inh_pop_fixed_weight_scale_cond')
target_pop.record('spikes')
inh_pop.record('spikes')

if using_fixed_n:
    source_target_proj = sim.Projection(source_pop,target_pop,sim.FixedNumberPreConnector(from_input_n_connections),synapse_type=sim.StaticSynapse(weight=connection_weight))
    source_inh_proj = sim.Projection(source_pop,inh_pop,sim.FixedNumberPreConnector(from_input_n_connections),synapse_type=sim.StaticSynapse(weight=connection_weight))
    target_lat_proj = sim.Projection(target_pop,target_pop,sim.FixedNumberPreConnector(from_target_n_connections),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    target_inh_proj = sim.Projection(target_pop,inh_pop,sim.FixedNumberPreConnector(from_target_n_connections),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    inh_lat_proj = sim.Projection(inh_pop,inh_pop,sim.FixedNumberPreConnector(from_i_n_connections),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')
    inh_target_proj = sim.Projection(inh_pop,target_pop,sim.FixedNumberPreConnector(from_i_n_connections),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')

else:
    source_target_proj = sim.Projection(source_pop,target_pop,sim.FromListConnector(from_list_fixed_pre(input_size,target_pop_size,from_input_n_connections)),synapse_type=sim.StaticSynapse(weight=connection_weight))
    source_inh_proj = sim.Projection(source_pop,inh_pop,sim.FromListConnector(from_list_fixed_pre(input_size,inh_pop_size,from_input_n_connections)),synapse_type=sim.StaticSynapse(weight=connection_weight))
    target_lat_proj = sim.Projection(target_pop,target_pop,sim.FromListConnector(from_list_fixed_pre(target_pop_size,target_pop_size,from_target_n_connections)),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    target_inh_proj = sim.Projection(target_pop,inh_pop,sim.FromListConnector(from_list_fixed_pre(target_pop_size,inh_pop_size,from_target_n_connections)),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    inh_lat_proj = sim.Projection(inh_pop,inh_pop,sim.FromListConnector(from_list_fixed_pre(inh_pop_size,inh_pop_size,from_i_n_connections)),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')
    inh_target_proj = sim.Projection(inh_pop,target_pop,sim.FromListConnector(from_list_fixed_pre(inh_pop_size,target_pop_size,from_i_n_connections)),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')


max_period = 5000.
num_recordings =int((duration/max_period)+1)
# np.savez('./master_profiles.npz',spike_processing=np.zeros(0),dma_processing=np.zeros(0),n_calls=np.zeros(0))

for i in range(num_recordings):
    sim.run(duration/num_recordings)

target_data = target_pop.get_data(['spikes'])
output_spikes = target_data.segments[0].spiketrains
inh_data = inh_pop.get_data(['spikes'])
inh_spikes = inh_data.segments[0].spiketrains

sim.end()
# mem_v = output_data.segments[0].filter(name='v')
# cell_voltage_plot_8(mem_v, plt, duration / timestep, [], scale_factor=timestep / 1000., title='output pop')

plot_spikes = output_spikes
plot_inh_spikes = inh_spikes

spike_raster_plot_8(plot_spikes, plt, duration / 1000., target_pop_size + 1, 0.001, title="output pop activity")
spike_raster_plot_8(plot_inh_spikes, plt, duration / 1000., inh_pop_size + 1, 0.001, title="inh pop activity")

# spike_raster_plot_8(inh_spikes,plt,duration/1000.,pop_size+1,0.001,title="inh pop activity")
# non_zero_spikes = [train for train in output_spikes if len(train)>0]
# print "non zero output spikes length:{} target n_neurons:{}".format(len(non_zero_spikes),target_pop_size)
# spike_raster_plot_8(input_pop_spikes,plt,duration/1000.,input_size+1,0.001,title="input pop activity")
# spike_raster_plot_8(input_spikes,plt,duration/1000.,input_size+1,0.001,title="input an activity")

# print "average pre pop range in a projection:{}".format(np.mean(m_pre))
# plt.figure("number of sub pre pops each sub post pop forms connections with")
# plt.plot(np.asarray(m_pre)/float(len(input_pops)))
# plt.xlabel("sub post population")
# plt.ylabel("n connections to sub pre populations / n sub pre pops ")
plt.show()
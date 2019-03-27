import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint

#================================================================================================
# Simulation parameters
#================================================================================================
duration = 500.
n_input = 1000.
n_per_core = 255
n_rbn = 1
input_size = n_input#n_per_core*np.ceil(n_input/n_per_core)#len(input_spikes)#
n_total = input_size
target_pop_size = int(n_total*2./3)#n_per_core*np.ceil((n_total* 2./3)/n_per_core)#input_size#100.#int(np.round(n_total*10./89.)) #200
inh_pop_size = int(n_total*1./3)#n_per_core*np.ceil((n_total* 1./3)/n_per_core)
source_firing_rate = 10. #Hz
try:
    input_spikes=np.load("scale_test_fixed_prob_input_n_{}Hz={}.npy".format(int(n_input),int(source_firing_rate)))
except:
    input_spikes = []
    isi_ms = 1000./source_firing_rate
    n_repeats = int((duration/1000.)*source_firing_rate)

    for _ in range(int(input_size)):
        input_spikes.append([isi_ms*i + (np.random.rand())*(isi_ms/1.) for i in range(n_repeats)])
    np.save("scale_test_fixed_prob_input_n={}.npy".format(int(n_input)),input_spikes)



# input_spikes=[[1.],[10.,15.]]
# spike_raster_plot_8(input_spikes,plt,duration/1000.,ylim=input_size+1,title="input_activity")

p_intrinsic = 0.01#input_size/100.#int(0.01*input_size)#76#10#
p_extrinsic = 0.001
# n_connections = RandomDistribution('uniform',[30.,120.])

w2s_target = .1#7.#
# av_weight =(w2s_target/(0.5*n_connections))#(w2s_target/50.)#w2s_target#(w2s_target/25.)#
# connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])
# av_weight =(w2s_target/25.)#/(n_connections*0.005))#(w2s_target/50.)#w2s_target#(w2s_target/25.)#

total_n_neurons =input_size + target_pop_size + inh_pop_size
from_i_n_connections = int(p_intrinsic * inh_pop_size)
from_input_n_connections = int(p_intrinsic * input_size)
from_target_n_connections = int(p_intrinsic * target_pop_size)

from_neighbour_i_connections = int(p_extrinsic * inh_pop_size)
from_neighbour_target_connections = int(p_extrinsic * target_pop_size)

exc_ratio = 0.75
input_ratio = 1.
inh_ratio = exc_ratio*0.1#0.1#0.05
t_syn = 0.08#0.02#0.04#
av_lat_exc_w =w2s_target/(source_firing_rate * t_syn * from_target_n_connections*exc_ratio)
lat_exc_w = RandomDistribution('normal_clipped',[av_lat_exc_w,av_lat_exc_w/10.,0,av_lat_exc_w*2.])
av_lat_inh_w = w2s_target/(source_firing_rate * t_syn * from_i_n_connections*inh_ratio)
lat_inh_w = RandomDistribution('normal_clipped',[av_lat_inh_w,av_lat_inh_w/10.,0,av_lat_inh_w*2.])
av_weight =w2s_target/(source_firing_rate * t_syn * from_input_n_connections*input_ratio)
connection_weight =RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])

timestep = 1.#0.1

sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,n_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,n_per_core)

source_pop = [[] for _ in range(n_rbn)]
target_pop = [[] for _ in range(n_rbn)]
inh_pop = [[] for _ in range(n_rbn)]

source_target_proj= [[] for _ in range(n_rbn)]
source_inh_proj= [[] for _ in range(n_rbn)]
target_lat_proj= [[] for _ in range(n_rbn)]
target_inh_proj= [[] for _ in range(n_rbn)]
inh_lat_proj= [[] for _ in range(n_rbn)]
inh_target_proj= [[] for _ in range(n_rbn)]

target_data = [[] for _ in range(n_rbn)]
output_spikes = [[] for _ in range(n_rbn)]
inh_data = [[] for _ in range(n_rbn)]
inh_spikes = [[] for _ in range(n_rbn)]

for i in range(n_rbn):
    source_pop[i] = sim.Population(input_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
    target_pop[i] = sim.Population(target_pop_size,sim.IF_cond_exp,{},label='target_pop_fixed_weight_scale_cond')
    inh_pop[i] = sim.Population(inh_pop_size,sim.IF_cond_exp,{'tau_syn_E': 15.},label='inh_pop_fixed_weight_scale_cond')
    target_pop[i].record('spikes')
    inh_pop[i].record('spikes')

    distance_spread = 0.1

    # try:
    #     connecivity_data = np.load('./connectivity_files/{}_distance{}_connectivity.npz'.format(input_size,n_connections))
    #     i_t_list = connecivity_data['i_t_list']
    #     i_inh_list = connecivity_data['i_inh_list']
    #     t_t_list = connecivity_data['t_t_list']
    #     t_inh_list = connecivity_data['t_inh_list']
    #     inh_inh_list = connecivity_data['inh_inh_list']
    #     inh_t_list = connecivity_data['inh_t_list']

    # except:
    #intrinsic connections
    # if 1:
    #     i_t_list = normal_dist_connection_builder(input_size, target_pop_size, RandomDistribution,conn_num=from_input_n_connections, dist=1.,sigma=input_size * distance_spread)
    #     i_inh_list = normal_dist_connection_builder(input_size, inh_pop_size, RandomDistribution, conn_num=from_input_n_connections,dist=1., sigma=input_size * distance_spread)
    #     t_t_list = normal_dist_connection_builder(target_pop_size,target_pop_size,RandomDistribution,conn_num=from_target_n_connections,dist=1.,sigma=target_pop_size*distance_spread)
    #     t_inh_list = normal_dist_connection_builder(target_pop_size,inh_pop_size,RandomDistribution,conn_num=from_target_n_connections,dist=1.,sigma=target_pop_size*distance_spread)
    #     inh_inh_list = normal_dist_connection_builder(inh_pop_size,inh_pop_size,RandomDistribution,conn_num=from_i_n_connections,dist=1.,sigma=inh_pop_size*distance_spread)
    #     inh_t_list = normal_dist_connection_builder(inh_pop_size,target_pop_size,RandomDistribution,conn_num=from_i_n_connections,dist=1.,sigma=inh_pop_size*distance_spread)
    #     np.savez_compressed('./connectivity_files/{}_distance_connectivity_pex{}_pint{}.npz'.format(input_size,p_extrinsic,p_intrinsic),i_t_list=i_t_list,i_inh_list=i_inh_list,t_t_list=t_t_list,
    #                         t_inh_list=t_inh_list,inh_inh_list=inh_inh_list,inh_t_list=inh_t_list)
    if 1:
        # i_t_list = fixed_p_connection_builder(input_size, target_pop_size, p_intrinsic)
        i_t_list = fixed_p_connection_builder(input_size, target_pop_size, 0.01)
        i_inh_list = fixed_p_connection_builder(input_size, inh_pop_size, p_intrinsic)
        t_t_list = fixed_p_connection_builder(target_pop_size,target_pop_size,p_intrinsic)
        t_inh_list = fixed_p_connection_builder(target_pop_size,inh_pop_size,p_intrinsic)
        inh_inh_list = fixed_p_connection_builder(inh_pop_size,inh_pop_size,p_intrinsic)
        inh_t_list = fixed_p_connection_builder(inh_pop_size,target_pop_size,p_intrinsic)
        np.savez_compressed('./connectivity_files/{}_fixedp_connectivity_pex{}_pint{}.npz'.format(input_size,p_extrinsic,p_intrinsic),i_t_list=i_t_list,i_inh_list=i_inh_list,t_t_list=t_t_list,
                            t_inh_list=t_inh_list,inh_inh_list=inh_inh_list,inh_t_list=inh_t_list)
    source_target_proj[i] = sim.Projection(source_pop[i],target_pop[i],sim.FromListConnector(i_t_list),synapse_type=sim.StaticSynapse(weight=connection_weight))
    # source_inh_proj[i] = sim.Projection(source_pop[i],inh_pop[i],sim.FromListConnector(i_inh_list),synapse_type=sim.StaticSynapse(weight=connection_weight))
    target_lat_proj[i] = sim.Projection(target_pop[i],target_pop[i],sim.FromListConnector(t_t_list),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    target_inh_proj[i] = sim.Projection(target_pop[i],inh_pop[i],sim.FromListConnector(t_inh_list),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
    inh_lat_proj[i] = sim.Projection(inh_pop[i],inh_pop[i],sim.FromListConnector(inh_inh_list),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')
    inh_target_proj[i] = sim.Projection(inh_pop[i],target_pop[i],sim.FromListConnector(inh_t_list),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')

for i in range(n_rbn):
    #extrinsic connections
    for j in range(n_rbn):
        if j != i:
            t_t_list = fixed_p_connection_builder(target_pop_size, target_pop_size, p_extrinsic/n_rbn)
            t_inh_list = fixed_p_connection_builder(target_pop_size, inh_pop_size, p_extrinsic/n_rbn)
            inh_inh_list = fixed_p_connection_builder(inh_pop_size, inh_pop_size, p_extrinsic/n_rbn)
            inh_t_list = fixed_p_connection_builder(inh_pop_size, target_pop_size, p_extrinsic/n_rbn)

            ex_target_lat_proj = sim.Projection(target_pop[i],target_pop[j],sim.FromListConnector(t_t_list),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
            ex_target_inh_proj = sim.Projection(target_pop[i],inh_pop[j],sim.FromListConnector(t_inh_list),synapse_type=sim.StaticSynapse(weight=lat_exc_w))
            ex_inh_lat_proj = sim.Projection(inh_pop[i],inh_pop[j],sim.FromListConnector(inh_inh_list),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')
            ex_inh_target_proj = sim.Projection(inh_pop[i],target_pop[j],sim.FromListConnector(inh_t_list),synapse_type=sim.StaticSynapse(weight=lat_inh_w),receptor_type='inhibitory')

    max_period = 5000.
    num_recordings =int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)


for i in range(n_rbn):
    target_data[i] = target_pop[i].get_data(['spikes'])
    output_spikes[i] = target_data[i].segments[0].spiketrains
    inh_data[i] = inh_pop[i].get_data(['spikes'])
    inh_spikes[i] = inh_data[i].segments[0].spiketrains

    # mem_v = output_data.segments[0].filter(name='v')
    # cell_voltage_plot_8(mem_v, plt, duration / timestep, [], scale_factor=timestep / 1000., title='output pop')

    plot_spikes = output_spikes[i]
    plot_inh_spikes = inh_spikes[i]

    plt.figure("output pop activity")
    spike_raster_plot_8(plot_spikes, plt, duration / 1000., target_pop_size + 1, 0.001, title="output pop {}".format(i),subplots=(n_rbn,1,i+1))
    plt.figure("input pop activity")
    spike_raster_plot_8(plot_inh_spikes, plt, duration / 1000., inh_pop_size + 1, 0.001, title="inh pop {}".format(i),subplots=(n_rbn,1,i+1))
    plt.figure("output pop psth")
    psth_plot_8(plt,range(len(plot_spikes)),plot_spikes,0.01,duration/1000.,title="output pop {}".format(i),subplots=(n_rbn,1,i+1))
    plt.figure("inh pop psth")
    psth_plot_8(plt,range(len(plot_inh_spikes)),plot_inh_spikes,0.01,duration/1000.,title="inh pop {}".format(i),subplots=(n_rbn,1,i+1))

sim.end()

plt.show()
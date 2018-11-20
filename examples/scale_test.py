import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import spike_raster_plot_8,get_sub_pop_spikes,sub_pop_builder_auto,normal_dist_connection_builder,spatial_normal_dist_connection_builder
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint
#================================================================================================
# Simulation parameters
#================================================================================================
#input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
#cochlea_file = np.load(input_directory + '/spinnakear_asc_1s_30dB.npz')
#an_spikes = cochlea_file['scaled_times']

# test = RandomDistribution('normal',[15000,30000/15.])
# plt.figure()
# plt.hist(test.next(n=1000))
# plt.show()

cell_params_cond = {'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.6,#-70.0,
               'v_rest': -60.6,
               'v_thresh': -56.
               }

duration = 1000.
sub_pop = False

input_spikes =[]
# for neuron in an_spikes[0]:
#     input_spikes.append([spike_time for spike_time in neuron if spike_time <= duration])
spatial_dist = True
isi = 50
n_pres = int(duration/isi)
n_input = 1000.
n_per_core = 255#128
input_size = n_per_core*np.ceil(n_input/n_per_core)#len(input_spikes)#
# n_total = 2 * input_size
target_pop_size = n_per_core*np.ceil((2*n_input* 2./3 * 24./89)/n_per_core)#input_size#100.#int(np.round(n_total*10./89.)) #200
inh_pop_size = n_per_core*np.ceil((2*n_input* 1./3 * 24./89)/n_per_core)
max_post_per_core = 64.
n_sub_pops = int(np.ceil(target_pop_size/max_post_per_core))#32#40#13
sub_pre_size = 256.#input_size#input_size/4 #input_size/n_sub_pops#50.#2500.#256.#

source_firing_rate = 10. #Hz
isi_ms = 1000./source_firing_rate
n_repeats = int((duration/1000.)*source_firing_rate)

for _ in range(int(input_size)):
    input_spikes.append([isi_ms*i + (np.random.rand())*(isi_ms/2.) for i in range(n_repeats)])

spike_raster_plot_8(input_spikes,plt,duration/1000.,ylim=input_size+1,title="input_activity")

n_connections = 10#int(0.01*input_size)#76#10#
# n_connections = RandomDistribution('uniform',[30.,120.])

w2s_target = .1#7.#
av_weight =(w2s_target/(0.5*n_connections))#(w2s_target/50.)#w2s_target#(w2s_target/25.)#
connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])
# source_target_list = normal_dist_connection_builder(int(input_size),int(target_pop_size),RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/15.
#                                              ,conn_weight=connection_weight)

if spatial_dist:
    list_file_name = './conn_list_{}input_scaled_id.npz'.format(input_size)
else:
    list_file_name = './conn_list_{}input.npz'.format(input_size)
# if spatial_dist:
#     source_target_list = spatial_normal_dist_connection_builder(int(input_size),int(input_size),int(target_pop_size),RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/15.
#                                                  ,conn_weight=connection_weight)
#     source_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(input_size),int(inh_pop_size),RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/15.
#                                                  ,conn_weight=connection_weight)
#
#     target_target_list = spatial_normal_dist_connection_builder(int(input_size),int(target_pop_size),int(target_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/10.)
#     target_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(target_pop_size),int(inh_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/10.)
#     inh_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(inh_pop_size),int(inh_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/1.)
#     inh_target_list = spatial_normal_dist_connection_builder(int(input_size),int(inh_pop_size),int(target_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/1.)
#
# else:
#     source_target_list = normal_dist_connection_builder(int(input_size), int(target_pop_size),
#                                                                 RandomDistribution, conn_num=n_connections, dist=1.,
#                                                                 sigma=input_size / 15.
#                                                                 , conn_weight=connection_weight)
#     source_inh_list = normal_dist_connection_builder(int(input_size), int(inh_pop_size),
#                                                              RandomDistribution, conn_num=n_connections, dist=1.,
#                                                              sigma=input_size / 15.
#                                                              , conn_weight=connection_weight)
#     target_target_list = normal_dist_connection_builder(int(input_size), int(target_pop_size),
#                                                                 RandomDistribution, conn_num=5,
#                                                                 dist=1., sigma=2., conn_weight=w2s_target / 10.)
#     target_inh_list = normal_dist_connection_builder(int(target_pop_size), int(inh_pop_size),
#                                                              RandomDistribution, conn_num=5, dist=1., sigma=2.,
#                                                              conn_weight=w2s_target / 10.)
#     inh_inh_list = normal_dist_connection_builder(int(inh_pop_size), int(inh_pop_size),
#                                                           RandomDistribution, conn_num=5, dist=1., sigma=2.,
#                                                           conn_weight=w2s_target / 1.)
#     inh_target_list = normal_dist_connection_builder(int(inh_pop_size), int(target_pop_size),
#                                                              RandomDistribution, conn_num=5, dist=1., sigma=2.,
#                                                              conn_weight=w2s_target / 1.)
# np.savez_compressed(list_file_name,source_target_list=source_target_list,source_inh_list=source_inh_list,target_target_list=target_target_list,
#                     target_inh_list=target_inh_list,inh_inh_list=inh_inh_list,inh_target_list=inh_target_list)


connection_list_file = np.load(list_file_name)
source_target_list=connection_list_file['source_target_list']
source_inh_list=connection_list_file['source_inh_list']
target_target_list=connection_list_file['target_target_list']
target_inh_list=connection_list_file['target_inh_list']
inh_inh_list=connection_list_file['inh_inh_list']
inh_target_list=connection_list_file['inh_target_list']

# source_target_list = np.load('./conn_list.npy')
# source_target_list =[]
# for neuron in range(int(target_pop_size)):
#    source_ids = np.random.choice(int(input_size),n_connections,replace=False)
#    # source_ids = [neuron]
#    for source in source_ids:
#        source_target_list.append((source,neuron,av_weight,1.))

timestep = 1.#0.1
sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,n_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,n_per_core)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp,16)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,64.)
# input_pops,target_pops,an_on_projs = sub_pop_builder(sim,32,target_pop_size,sim.IF_cond_exp,cell_params_cond,"an_input","octopus",an_on_list)
# input_pops,target_pops,an_on_projs = sub_pop_builder_combine(sim,n_sub_pops,target_pop_size,sim.IF_cond_exp,cell_params_cond,"an_input","octopus",an_on_list)

# input_ssa = sim.Population(input_size, sim.SpikeSourceArray(spike_times=input_spikes),
#                                label="input_pop_ssa")
# input_pops,target_pops,an_on_projs = sub_pop_builder_single(sim,1,target_pop_size,sim.IF_cond_exp,cell_params_cond,sim.IF_cond_exp,one_to_one_cond_params,
#                                                             # "an_input","octopus_fixed_weight_scale_cond",an_on_list,input_spikes,sub_pre_pop_size=sub_pre_size)
#                                                             "an_input","octopus",an_on_list,input_spikes,sub_pre_pop_size=5000)
if sub_pop:
    import time
    t_total = time.time()
    input_pops,target_pops,an_on_projs,m_pre,posts_from_pop_index_dict = sub_pop_builder_auto(sim,int(target_pop_size),sim.IF_cond_exp,{},"SSA",input_spikes,
                                                                "input_pop","target_pop",source_target_list)
    elapsed_time = time.time() - t_total
    print "total sub pop building time: {}s".format(elapsed_time)

# input_pops,target_pops,an_on_projs,m_pre = sub_pop_builder_inter(sim,int(target_pop_size),sim.IF_cond_exp,{},"SSA",input_spikes,
#                                                             "input_pop","target_pop",source_target_list)
else:
    if spatial_dist:
        pop_size = max([input_size,target_pop_size,inh_pop_size])
        source_pop = sim.Population(pop_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
        target_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='target_pop')
        inh_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='inh_pop')
    else:
        source_pop = sim.Population(input_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
        target_pop = sim.Population(target_pop_size,sim.IF_cond_exp,{},label='target_pop')
        inh_pop = sim.Population(inh_pop_size,sim.IF_cond_exp,{},label='inh_pop')
    target_pop.record('spikes')#
    inh_pop.record('spikes')
    source_target_proj = sim.Projection(source_pop,target_pop,sim.FromListConnector(source_target_list),synapse_type=sim.StaticSynapse())
    source_inh_proj = sim.Projection(source_pop,inh_pop,sim.FromListConnector(source_inh_list),synapse_type=sim.StaticSynapse())
    target_lat_proj = sim.Projection(target_pop,target_pop,sim.FromListConnector(target_target_list),synapse_type=sim.StaticSynapse())
    target_inh_proj = sim.Projection(target_pop,inh_pop,sim.FromListConnector(target_inh_list),synapse_type=sim.StaticSynapse())
    inh_lat_proj = sim.Projection(inh_pop,inh_pop,sim.FromListConnector(inh_inh_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
    inh_target_proj = sim.Projection(inh_pop,target_pop,sim.FromListConnector(inh_target_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')

max_period = 5000.
num_recordings =int((duration/max_period)+1)
np.savez('./master_profiles.npz',spike_processing=np.zeros(0),dma_processing=np.zeros(0),n_calls=np.zeros(0))

for i in range(num_recordings):
    sim.run(duration/num_recordings)
if sub_pop:
    output_spikes=get_sub_pop_spikes(target_pops,posts_from_pop_index_dict)
else:
    target_data = target_pop.get_data(['spikes'])
    output_spikes = target_data.segments[0].spiketrains
    # inh_data = inh_pop.get_data(['spikes'])
    # inh_spikes = inh_data.segments[0].spiketrains

sim.end()
# mem_v = output_data.segments[0].filter(name='v')
# cell_voltage_plot_8(mem_v, plt, duration / timestep, [], scale_factor=timestep / 1000., title='output pop')
if spatial_dist:
    spike_raster_plot_8(output_spikes,plt,duration/1000.,pop_size+1,0.001,title="output pop activity")
else:
    spike_raster_plot_8(output_spikes, plt, duration / 1000., target_pop_size + 1, 0.001, title="output pop activity")

# spike_raster_plot_8(inh_spikes,plt,duration/1000.,pop_size+1,0.001,title="inh pop activity")
non_zero_spikes = [train for train in output_spikes if len(train)>0]
print "non zero output spikes length:{} target n_neurons:{}".format(len(non_zero_spikes),target_pop_size)
# spike_raster_plot_8(input_pop_spikes,plt,duration/1000.,input_size+1,0.001,title="input pop activity")
# spike_raster_plot_8(input_spikes,plt,duration/1000.,input_size+1,0.001,title="input an activity")

# print "average pre pop range in a projection:{}".format(np.mean(m_pre))
# plt.figure("number of sub pre pops each sub post pop forms connections with")
# plt.plot(np.asarray(m_pre)/float(len(input_pops)))
# plt.xlabel("sub post population")
# plt.ylabel("n connections to sub pre populations / n sub pre pops ")
plt.show()
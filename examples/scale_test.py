import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import spike_raster_plot_8,get_sub_pop_spikes,sub_pop_builder_auto,normal_dist_connection_builder,spatial_normal_dist_connection_builder
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint

def sub_pop_builder(sim,n_sub_pops,post_size,post_type,post_params,pre_name,post_name,projection_list):
    post_pops =[]
    pre_pops =[]
    pre_post_projs =[]
    post_offset = post_size / n_sub_pops
    remaining_post_neurons = post_size
    for i, post_neuron in enumerate(range(0, post_size, post_offset)):
        if post_neuron + post_offset < post_size:
            pop_size = post_offset
        else:
            pop_size = remaining_post_neurons
        post_pops.append(sim.Population(pop_size, post_type, post_params, label=post_name + "_sub_pop_{}".format(i)))
        post_pops[i].record(["spikes"])
        remaining_post_neurons -= pop_size
        sub_list = [(pre,post,weight,delay) for (pre,post,weight,delay) in projection_list if
                    post >= post_neuron and post < (post_neuron + post_offset)]
        pres = np.asarray([pre for (pre,post,weight,delay) in sub_list])
        pre_range = pres.max() - pres.min()
        pre_pops.append(
            sim.Population(pre_range, sim.SpikeSourceArray(spike_times=input_spikes[pres.min():pres.max()]),
                           label=pre_name + "_sub_pop_{}".format(i)))
        # create projection
        offset_list = [(pre - pres.min(),post - post_neuron,weight,delay) for (pre,post,weight,delay) in
                       sub_list]
        pre_post_projs.append(sim.Projection(pre_pops[i], post_pops[i], sim.FromListConnector(offset_list),
                                          synapse_type=sim.StaticSynapse()))

    return pre_pops,post_pops,pre_post_projs

def sub_pop_builder_combine(sim,n_sub_pops,post_size,post_type,post_params,pre_name,post_name,projection_list):
    post_pops =[]
    pre_pops =[]
    pres=[]
    pre_post_projs =[]
    sub_lists=[]
    post_offset = post_size / n_sub_pops
    remaining_post_neurons = post_size
    boundary=0
    pre_pop_index = 0
    pre_slice = [boundary]
    for i, post_neuron in enumerate(range(0, post_size, post_offset)):
        if post_neuron + post_offset < post_size:
            pop_size = post_offset
        else:
            pop_size = remaining_post_neurons
        post_pops.append(sim.Population(pop_size, post_type, post_params, label=post_name + "_sub_pop_{}".format(i)))
        post_pops[i].record(["spikes"])
        remaining_post_neurons -= pop_size
        sub_lists.append([(pre,post,weight,delay) for (pre,post,weight,delay) in projection_list if
                    post >= post_neuron and post < (post_neuron + post_offset)])
        pres.append(np.asarray([pre for (pre,post,weight,delay) in sub_lists[i]]))
        # pre_range = pres[i].max() - pres[i].min()
        #check what the shared pre ids are between each post in order, when we hit a none intersection we create a population boundary
        # the 'pre' population to be created will contain all previous pres from the pres list.
        # the post population can be created by merging
        #all the post ids that occured before the boundary was established
        intersection = np.intersect1d(pres[boundary],pres[i]).tolist()
        if len(intersection) > 100:
            pre_slice += intersection
        else:
            min_pre = min(pre_slice)
            max_pre = max(pre_slice)
            pre_range = max_pre - min_pre
            pre_pops.append(
                sim.Population(pre_range, sim.SpikeSourceArray(spike_times=input_spikes[min_pre:max_pre]),
                               label=pre_name + "_sub_pop_{}".format(i)))
            # create projections
            for post_index in range(boundary,i):
                offset_list = [(pre - min_pre,post - post_index*post_offset,weight,delay) for (pre,post,weight,delay) in
                               sub_lists[post_index]]
                pre_post_projs.append(sim.Projection(pre_pops[pre_pop_index], post_pops[post_index], sim.FromListConnector(offset_list),
                                                  synapse_type=sim.StaticSynapse()))
            pre_pop_index += 1
            boundary = i+1 #create boundary for next slice
            del pre_slice
            pre_slice = [boundary]

    return pre_pops,post_pops,pre_post_projs

def sub_pop_builder_single(sim,n_sub_pops,post_size,post_type,post_params,pre_type,pre_params,pre_name,post_name,projection_list,input_spikes,sub_pre_pop_size=1000.):
    post_pops =[]
    pre_pop_size = int(len(input_spikes))
    pre_pops =[]
    pre_post_projs =[]
    sub_lists=[]
    pres = []
    post_offset = post_size / n_sub_pops
    remaining_post_neurons = post_size

    #create pre pops
    pre_offset = int(sub_pre_pop_size)
    remaining_pre_neurons = pre_pop_size
    for i,pre_neuron in enumerate(range(0,pre_pop_size,pre_offset)):
        if pre_neuron + pre_offset < pre_pop_size:
            pop_size = pre_offset
        else:
            pop_size = remaining_pre_neurons
        pre_pops.append(
            sim.Population(pop_size, sim.SpikeSourceArray(spike_times=input_spikes[pre_neuron:pre_neuron+pop_size]),
            # sim.Population(pop_size, pre_type, pre_params,
                           label=pre_name + "_sub_pop_{}".format(i)))
        # pre_pops[i].record(["spikes"])
        pres.append(range(pre_neuron,int(pre_neuron+pop_size)))
        remaining_pre_neurons-=pre_offset
    #create post pops
    for i, post_neuron in enumerate(range(0, post_size, post_offset)):
        if post_neuron + post_offset < post_size:
            pop_size = post_offset
        else:
            pop_size = remaining_post_neurons
        post_pops.append(sim.Population(pop_size, post_type, post_params, label=post_name + "_sub_pop_{}".format(i)))
        post_pops[i].record(["spikes"])
        remaining_post_neurons -= pop_size
        sub_lists.append([(pre,post,weight,delay) for (pre,post,weight,delay) in projection_list if
                    post >= post_neuron and post < (post_neuron + post_offset)])
        #go through each of the connections and setup relevant projections
        pre_lists = [[] for _ in range(len(pres))]
        for (pre,post,weight,delay) in sub_lists[i]:
            #find sub pre pop index
            for idx,ids in enumerate(pres):
                if pre in ids:
                    pre_index = idx
                    break
            # print pre_index
            min_pre = min(pres[pre_index])
            pre_lists[pre_index].append((pre - min_pre,post - post_neuron,weight,delay))

        for j,pre_list in enumerate(pre_lists):
            if pre_list is not None and len(pre_list):
                pre_post_projs.append(sim.Projection(pre_pops[j], post_pops[i], sim.FromListConnector(pre_list),
                                              synapse_type=sim.StaticSynapse()))
    return pre_pops,post_pops,pre_post_projs

# def pre_group_generator(input_size,target_pop_size,source_target_list):
#     # go through connectivity list and separate post neurons into sub populations based on common pre neurons
#     # build 2D matrix of pre post connectivity
#     connectivity_matrix = np.zeros((int(input_size), int(target_pop_size)),dtype=bool)
#     weight_matrix = np.zeros((int(input_size), int(target_pop_size)))
#     delay_matrix = np.zeros((int(input_size), int(target_pop_size)))
#
#     for (pre, post, w, d) in source_target_list:
#         if w > 0.:
#             connectivity_matrix[pre][post] = 1
#             weight_matrix[pre][post] = w
#             delay_matrix[pre][post] = d
#
#     # group indices of rows of the connectivity matrix that share a percentage of post neurons
#     pre_groups = []
#     pre_groups_index = 0
#     for pre_neuron in range(int(input_size)):
#         pre_exists = False
#         for group in pre_groups:
#             if pre_neuron in group:
#                 pre_exists = True
#                 break
#         if pre_exists is False:
#             pre_groups.append([pre_neuron])
#             post_connections = connectivity_matrix[pre_neuron]
#             similarity_matrix = np.sum(connectivity_matrix * post_connections,axis=1)
#             #remove matching entry
#             similarity_matrix[pre_neuron] = 0
#             # group_ids = np.nonzero(similarity_matrix >= similarity_matrix.max() * 0.5)
#             group_ids = np.nonzero(similarity_matrix >= similarity_matrix.max() * 0.25)
#             for pre_index in group_ids[0]:
#                 # check index doesn't feature in a previous group
#                 pre_exists = False
#                 for group in pre_groups:
#                     if pre_index in group:
#                         pre_exists = True
#                         break
#                 if pre_exists is False:
#                     pre_groups[pre_groups_index].append(pre_index)
#             pre_groups_index += 1
#     # attempt to resort any pre_groups that are size 1
#     for i, group in enumerate(pre_groups):
#         if len(group) == 1:
#             pre_neuron = group[0]
#             post_connections = connectivity_matrix[pre_neuron]
#             similarity_matrix = np.sum(connectivity_matrix * post_connections,axis=1)
#             #remove matching entry
#             similarity_matrix[pre_neuron] = 0
#             max_index = np.argmax(similarity_matrix)
#             for j, other_group in enumerate(pre_groups):
#                 if max_index in other_group:
#                     pre_groups[j].append(pre_neuron)
#                     del pre_groups[i]
#                     break
#     for i,group in enumerate(pre_groups):
#         group.sort()
#         pre_groups[i]=np.asarray(group)
#     return pre_groups,connectivity_matrix,weight_matrix,delay_matrix
#
# def post_group_generator(pre_groups,connectivity_matrix):
#     post_groups = []
#     # create post pops
#     for i, pre_indices in enumerate(pre_groups):
#         # build post groups
#         # posts=[]
#         posts= np.nonzero(np.sum(connectivity_matrix[pre_indices],axis=0))[0]
#         # for pre_index in pre_indices:
#         #     ps = np.nonzero(connectivity_matrix[pre_index])[0]
#         #     for p in ps:
#         #         posts.append(p)
#         post_groups.append(posts)
#     #remove duplicate post neurons
#     for i, p in enumerate(post_groups):
#         removals = []
#         for j, idx in enumerate(p):
#             for k, other_group in enumerate(post_groups):
#                 if idx in other_group and i != k:
#                     removals.append(j)
#                     break
#         post_groups[i] = np.delete(post_groups[i], removals)
#     post_groups = np.asarray([group for group in post_groups if group.size > 0])
#     return post_groups
# def sub_pop_builder_auto(sim,post_size,post_type,post_params,pre_type,pre_params,pre_name,
#                           post_name,projection_list,max_sub_pre_pop_size=128.,max_post_per_core=64.,
#                           pre_pops=False,post_record_list=["spikes"]):
#     import numpy as np
#     if not isinstance(pre_type,str):
#         raise Exception("non spike source array pre pops currently unsupported")
#         #TODO: allow for non SSA pre pops to be passed in e.g. SpiNNakEar outputs
#     else:
#         input_spikes = pre_params
#     n_sub_pops = int(np.ceil(post_size / max_post_per_core))
#
#     post_pops =[]
#     post_pop_dict={}
#     posts_from_pop_index_dict={}
#     post_pop_index=0
#     created_post_pops=[]
#     pre_pop_size = int(len(input_spikes))
#     pre_post_projs =[]
#     # sub_lists=[]
#     pres = []
#     post_offset = post_size / n_sub_pops
#     remaining_post_neurons = post_size
#     max_pre_index_per_projection=[]
#     chip_index = 0
#     pre_neuron = 0
#
#     if pre_pops is False:
#         pre_pops =[]
#     else:#setup empty pre_iterations so we don't make new pre pops and calculate the pres list
#         pre_iterations = np.asarray([])
#         pres=[]
#         for pop in pre_pops:
#             pres.append(range(pre_neuron, int(pre_neuron + pop.size)))
#             pre_neuron += pop.size
#     import time
#     t = time.time()
#     pre_groups,connectivity_matrix,weight_matrix,delay_matrix = pre_group_generator(pre_pop_size, post_size, projection_list)
#     elapsed_pre_gen_time = time.time() - t
#     t = time.time()
#     post_groups = post_group_generator(pre_groups,connectivity_matrix)
#     elapsed_post_gen_time = time.time() - t
#     t = time.time()
#     for i, pre_indices in enumerate(pre_groups):
#         #create pre pop
#         pop_size=len(pre_indices)
#         sub_spikes = np.asarray(input_spikes)[pre_indices]
#         sub_spikes = sub_spikes.tolist()
#
#         pre_pops.append(
#             sim.Population(pop_size, sim.SpikeSourceArray(spike_times=sub_spikes),
#                            label=pre_name + "_sub_pop_{}".format(i),
#                             # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
#                             #                                    machine_chip_coordinates[chip_index][1])]
#                                                                ))
#         #find connected post groups
#         connected_post_group_indices=[]
#
#         pre_filter = weight_matrix[pre_indices]
#         for j, post_indices in enumerate(post_groups):
#             pre_post_filter = pre_filter[:, post_indices]
#             if np.count_nonzero(pre_post_filter)>0:
#                 connected_post_group_indices.append(j)
#
#         connected_post_groups=post_groups[connected_post_group_indices]
#         for posts in connected_post_groups:
#             pop_size=len(posts)
#             #if post pop not previously created
#             if str(posts) not in post_pop_dict:
#                 post_pops.append(sim.Population(pop_size, post_type, post_params,
#                                                 label=post_name + "_sub_pop_{}".format(i),
#                                                 # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
#                                                                                    # machine_chip_coordinates[chip_index][1])]
#                                                 ))
#                 post_pops[post_pop_index].record(post_record_list)
#                 post_pop_dict[str(posts)] = post_pop_index
#                 posts_from_pop_index_dict[str(post_pop_index)] = posts
#                 post_pop_index+=1
#
#             sub_lists=[]
#             pre_post_filter = pre_filter[:, posts]
#             delay_filter = delay_matrix[pre_indices]
#             pre_post_delay_filter = delay_filter[:,posts]
#             for pre_index,weights in enumerate(pre_post_filter):
#                 if np.count_nonzero(weights)>0:
#                     post_indices = np.nonzero(weights)[0]
#                     for post_index in post_indices:
#                         sub_lists.append((pre_index,post_index,weights[post_index],pre_post_delay_filter[pre_index][post_index]))
#
#             pre_post_projs.append(sim.Projection(pre_pops[i], post_pops[post_pop_dict[str(posts)]], sim.FromListConnector(sub_lists),
#                                                  synapse_type=sim.StaticSynapse()))
#
#         chip_index+=1
#     elapsed_pop_gen_projection_time = time.time() - t
#     print "sub pop building complete, n_sub_pre_pops={}, n_sub_post_pops={}".format(len(pre_pops),len(post_pops))
#     print "sub pop build times: sub_pre_calc={}, sub_post_calc={}, sub_pop_gen_project={} ".format(elapsed_pre_gen_time,elapsed_post_gen_time,elapsed_pop_gen_projection_time)
#
#     return pre_pops,post_pops,pre_post_projs,max_pre_index_per_projection,posts_from_pop_index_dict
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

isi = 50
n_pres = int(duration/isi)
n_input = 1000.
n_per_core = 255#128
input_size = n_per_core*np.ceil(n_input/n_per_core)#len(input_spikes)#
# n_total = 2 * input_size
target_pop_size = input_size#n_per_core*np.ceil((2*n_input* 2./3 * 24./89)/n_per_core)#100.#int(np.round(n_total*10./89.)) #200
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
source_target_list = spatial_normal_dist_connection_builder(int(input_size),int(input_size),int(target_pop_size),RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/15.
                                             ,conn_weight=connection_weight)
source_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(input_size),int(inh_pop_size),RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/15.
                                             ,conn_weight=connection_weight)

target_target_list = spatial_normal_dist_connection_builder(int(input_size),int(target_pop_size),int(target_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/10.)
target_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(target_pop_size),int(inh_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/10.)
inh_inh_list = spatial_normal_dist_connection_builder(int(input_size),int(inh_pop_size),int(inh_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/1.)
inh_target_list = spatial_normal_dist_connection_builder(int(input_size),int(inh_pop_size),int(target_pop_size),RandomDistribution,conn_num=5,dist=1.,sigma=2.,conn_weight=w2s_target/1.)
# np.save('./conn_list',source_target_list)
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
    pop_size = max([input_size,target_pop_size,inh_pop_size])
    source_pop = sim.Population(pop_size,sim.SpikeSourceArray(spike_times=input_spikes),label='input_pop')
    target_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='target_pop')
    target_pop.record('spikes')#
    # inh_pop = sim.Population(pop_size,sim.IF_cond_exp,{},label='inh_pop')
    # inh_pop.record('spikes')
    source_target_proj = sim.Projection(source_pop,target_pop,sim.FromListConnector(source_target_list),synapse_type=sim.StaticSynapse())
    # source_inh_proj = sim.Projection(source_pop,inh_pop,sim.FromListConnector(source_inh_list),synapse_type=sim.StaticSynapse())
    # target_lat_proj = sim.Projection(target_pop,target_pop,sim.FromListConnector(target_target_list),synapse_type=sim.StaticSynapse())
    # target_inh_proj = sim.Projection(target_pop,inh_pop,sim.FromListConnector(target_inh_list),synapse_type=sim.StaticSynapse())
    # inh_lat_proj = sim.Projection(inh_pop,inh_pop,sim.FromListConnector(inh_inh_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
    # inh_target_proj = sim.Projection(inh_pop,target_pop,sim.FromListConnector(inh_target_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')

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
spike_raster_plot_8(output_spikes,plt,duration/1000.,pop_size+1,0.001,title="output pop activity")
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
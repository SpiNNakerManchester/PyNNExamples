import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

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

#================================================================================================
# Simulation parameters
#================================================================================================
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_50dB_10000fibres.npz')
an_spikes = cochlea_file['scaled_times']

cell_params_cond = {'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.6,#-70.0,
               'v_rest': -60.6,
               'v_thresh': -56.
               }

n_sub_pops = 1#32
duration = 1000.

input_spikes =[]
for neuron in an_spikes:
    input_spikes.append([spike_time for spike_time in neuron if spike_time <= duration])

# isi = 50
# n_pres = int(duration/isi)
# input_spikes = []
# for _ in range(5000):
#     input_spikes.append([i*isi + ((np.random.rand() - 0.5) * 10.) for i in range(1,n_pres)])

input_size = len(input_spikes)
n_total = 2 * input_size
target_pop_size = int(np.round(n_total*10./89.)) #200
# n_connections = 76
n_connections = RandomDistribution('uniform',[30.,120.])

w2s_target = 7.#.1
av_weight =(w2s_target/45.)
connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])

timestep = 1.#0.1
sim.setup(timestep=timestep)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,16)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,32)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,1000.)

an_on_list = normal_dist_connection_builder(input_size,target_pop_size,RandomDistribution,conn_num=n_connections,dist=1.,sigma=input_size/20.
                                            ,conn_weight=connection_weight)

input_pops,target_pops,an_on_projs = sub_pop_builder(sim,n_sub_pops,target_pop_size,sim.IF_cond_exp,cell_params_cond,"an_input","octopus",an_on_list)

n_target_pops = len(target_pops)

max_period = 5000.
num_recordings =int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

output_spikes=[]
for i in range(n_target_pops):
    data = (target_pops[i].get_data(["spikes"]))
    spikes = data.segments[0].spiketrains
    for neuron in spikes:
        output_spikes.append(neuron)

sim.end()

# mem_v = output_data.segments[0].filter(name='v')
# cell_voltage_plot_8(mem_v, plt, duration / timestep, [], scale_factor=timestep / 1000., title='output pop')
spike_raster_plot_8(output_spikes,plt,duration/1000.,target_pop_size+1,0.001,title="output pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,input_size+1,0.001,title="input pop activity")

plt.show()
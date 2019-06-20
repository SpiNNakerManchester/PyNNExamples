import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pyNN.utility.plotting import Figure, Panel

#================================================================================================
# Simulation parameters
#================================================================================================
active_params = {'e_rev_E': -45,#-55,#increase to closer to thresh
                 'e_rev_E2': 100,# increase to something very large 100 (need to adjust ff weights)
                 'tau_syn_E': 20,#50,
                 'tau_syn_E2': 5, #try increasing this
                 'v_thresh': -40,
                 #'v_reset': -65.
                 #'tau_syn_I': 1. #TODO: maybe use the dual inh?
                 }
w2s =0.1#0.15#2.
winh = 0.15#0.5
wpred = 0.2#0.15#w2s/2.#0.05#1.
w2s_target = 5.

input_pop_size =1
column_size = 16#32
number_of_columns = 50#250
active_pop_size = column_size*number_of_columns
cd_pop_size = 4#int(1 * active_pop_size)
if 1:#cd_pop_size<2000:
    get_weights=True
else:
    get_weights=False
# assume 1% of 2048 columns are active per 1            scaled_a_plus = int(round(self._a_plus *self._w_max * w))
#if each column fired at 1Hz then there would be approx. 2 active columns per timestep
#we assume each column fires at around 10Hz, producing approx. 20 active columns per ms
column_firing_rate = 2.#10.
isi =300.# 1000./column_firing_rate
num_firings =10#80#7#1200
# predict_delay = 10#8
input_spikes = []
# for j in range(number_of_columns):
#     input_spikes.append([(j*column_offset)+i*isi for i in range(1,num_firings)])#[10.,30,50]
num_patterns_in_sequence = 2#4
num_sequences = 1 #ABCD XBCY
num_columns_active_per_pattern = 1#5#int(0.15*number_of_columns)
column_offset = 50.#20.#int(isi/num_patterns_in_sequence)#isi/number_of_columns

#================================================================================================
# Pattern generation + column selection
#================================================================================================
#randomly chose column indices to represent
if num_sequences > 1: #ABCDXY
    chosen_columns = np.random.choice(number_of_columns, num_columns_active_per_pattern * 6, replace=False)
    for j in range(num_sequences):
        for pattern_index in range(num_patterns_in_sequence):
            for _ in range(num_columns_active_per_pattern):
                input_spikes.append([(pattern_index*column_offset)+i*isi + (j*isi/num_sequences) for i in range(1,num_firings)])
else: #ABCD
    # chosen_columns = np.arange(number_of_columns)
    chosen_columns = np.random.choice(number_of_columns,num_columns_active_per_pattern*num_patterns_in_sequence,replace=False)
    for pattern_index in range(num_patterns_in_sequence):
        for _ in range(num_columns_active_per_pattern):
            input_spikes.append([(pattern_index*column_offset)+i*isi for i in range(1,num_firings)])
onset_times = []
ms_onset_times = []
for i in range(num_patterns_in_sequence*num_sequences):
    onset_times.append([time/1000. for time in input_spikes[num_columns_active_per_pattern*i]])
    ms_onset_times.append([time for time in input_spikes[num_columns_active_per_pattern*i]])

pattern_duration = 10. #actually just one spike but this helps visibility in plots and also accounts for any prop delay in activation
final_pattern_time = ms_onset_times[0][-1]

# predict_delay = 10.
# predict_spikes = []
# for i,column_firings in enumerate(input_spikes):
#     predict_spikes.append([time-predict_delay*i for time in column_firings[1:]])
#     # predict_spikes.append([time+predict_delay for time in column_firings[5:]])

# predict_spikes = [[] for _ in range(cd_pop_size)]
# offset = cd_pop_size/num_columns_active_per_pattern
predict_spikes = []
for idx,stim_times in enumerate(input_spikes[::num_columns_active_per_pattern]):
    # predict_spikes[idx*offset].append([time + 2 for time in stim_times])
    predict_spikes.append([time + 2 for time in stim_times])

#================================================================================================
# SpiNNaker setup
#================================================================================================
timestep = 1.
# timestep = 0.1
sim.setup(timestep=timestep)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp,32)
sim.set_number_of_neurons_per_core(sim.extra_models.IFCondExp2E2I,32)
# sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)

duration = int(num_firings * isi)
max_period = 60000.#10000.#
num_recordings =int((duration/max_period)+1)
#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(num_columns_active_per_pattern*num_patterns_in_sequence*num_sequences,
                           sim.SpikeSourceArray(spike_times=input_spikes))
#one large population containing multiple 'columns'
# active_pop =sim.Population(active_pop_size,sim.IF_curr_exp,cell_params,label="active_fixed_weight_scale")#label="active_pop")#
active_pop =sim.Population(active_pop_size,sim.extra_models.IFCondExp2E2I,active_params,label="cond_fixed_weight_scale_2e2i")#label="active_pop")#
# active_pop =sim.Population(active_pop_size,sim.extra_models.IFCondExp2E2I,active_params,label="htm_pop")#label="active_pop")#
# active_pop =sim.Population(active_pop_size,sim.extra_models.IFCondExp2E2I,active_params,label="2e2i")#label="active_pop")#
# cd_pop = sim.Population(cd_pop_size,sim.SpikeSourceArray(spike_times=predict_spikes))
# cd_pop = sim.Population(cd_pop_size,sim.IF_curr_exp,target_cell_params)#,label="cd_fixed_weight_scale")

column_indices =[]
for column_index in chosen_columns:
    for i in range(column_size):
        column_indices.append(column_index+i)

# active_pop.record("all",indexes=column_indices)
active_pop.record("all")
# active_pop.record("spikes")
# cd_pop.record("all")
#================================================================================================
# Input to column projection
#================================================================================================
#each stimulus neuron of the input_pop will project to all the neurons in a corresponding single column of active pop
input_to_columns_list=[]
if num_sequences==1:
    for i,chosen in enumerate(chosen_columns):
        # if i==0:
        #     for j in range(column_size):
        #         input_to_columns_list.append((i,chosen*column_size+j))
        for j in range(column_size):
            input_to_columns_list.append((i,chosen*column_size+j))
else:
    #ABCD XBCY case chosen columns where i=1,2 need to be reused for the connections from XBCY to same BC columns as ABCD
    for i in range(len(input_spikes)):
        if i < 5*num_columns_active_per_pattern:#ABCDX
            chosen = chosen_columns[i]
        else:
            if i>=5*num_columns_active_per_pattern and i<7*num_columns_active_per_pattern:#B or C
                chosen_index = i-4*num_columns_active_per_pattern
            else:#Y
                chosen_index = i-2*num_columns_active_per_pattern
            chosen = chosen_columns[chosen_index]
        for j in range(column_size):
            input_to_columns_list.append((i,chosen*column_size+j))

input_projection = sim.Projection(input_pop,active_pop,sim.FromListConnector(input_to_columns_list),
                                  synapse_type=sim.StaticSynapse(weight=w2s,delay=1.),receptor_type="excitatory2")

#================================================================================================
# WTA column setup
#================================================================================================
inh_connection_list = []
for column in range(number_of_columns):
    column_index = column*column_size
    for post in range(column_size):
        for pre in range(column_size):
            if pre!=post:
                inh_connection_list.append((column_index+pre,column_index+post))
active_inh_active_projection = sim.Projection(active_pop,active_pop,sim.FromListConnector(inh_connection_list),
                                              synapse_type=sim.StaticSynapse(weight=winh),receptor_type='inhibitory')

# #================================================================================================
# #  Active to CD projection
# #================================================================================================
# initial_sync_num = 4.#num_columns_active_per_pattern#column_size#30.#15.#13.#30.#50.#
# av_weight = w2s_target/initial_sync_num
# w_max_cd = av_weight*1.1#w2s_target/2.#
# w_min_cd = 0.#av_weight*0.5#
# a_plus_cd = 0.5#0.1#1.#
# a_minus_cd = 0.5#0.1#1.#
# tau_plus_cd = 16.
# tau_minus_cd =30.
#
# stdp_model_cd = sim.STDPMechanism(
#         # timing_dependence=sim.SpikePairRule(
#         #     tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
#         timing_dependence=sim.extra_models.Vogels2011Rule(alpha=0.3, tau=10.0,#alpha=0.2, tau=20.0,
#                                                           A_plus=a_plus_cd, A_minus=a_minus_cd),
#         # weight_dependence=sim.AdditiveWeightDependence(
#         weight_dependence=sim.MultiplicativeWeightDependence(
#             w_min=w_min_cd, w_max=w_max_cd), weight=av_weight,delay=1.)
#
# structure_model_with_stdp = sim.StructuralMechanismSTDP(
#     stdp_model=stdp_model_cd,
#     weight=av_weight,  # Use this weights when creating a new synapse
#     max_weight=av_weight*0.9,#av_weight*0.9*2.,#av_weight*2,
#     s_max=int(initial_sync_num*1.5),#int(num_columns_active_per_pattern*1.5),  # Maximum allowed fan-in per target-layer neuron
#     #TODO: weight scale for the post population should be calculated using this value?
#     grid=[1,active_pop_size], # 1d spatial org of neurons, uncomment this if wanted
#     random_partner=False,
#     #selecting one of the last neurons to have spiked
#     #sigma_form_forward=15.,
#     # sigma_form_lateral=0.,
#     # p_form_forward=0.5,#0.9,#
#     f_rew= 10 ** 4,  #Hz
#     p_elim_dep=0.9,#1.,#0.99,#
#     p_elim_pot=0.,#0.1,#
# )

# active_cd_projection = sim.Projection(
#     active_pop, cd_pop,
#     sim.FixedProbabilityConnector(0.0),  # No initial connections
#     synapse_type=structure_model_with_stdp,
#     label="active -> cd structurally_plastic_projection"
# )
#================================================================================================
#  CD to Active STDP + projection
#================================================================================================
tau_plus=16.
tau_minus=30.
a_plus =1.#0.5#0.1#0.001#
a_minus =1.#0.5#0.1#0.001#
w_min = 0
w_max = wpred
stdp_initial_weight = w_max/15.#0.0#RandomDistribution('uniform',(0.,w_max/10.))
#1.#RandomDistribution('uniform',(10.,14.))#1#

stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=stdp_initial_weight)

# cd_projection_list = [(column_index,column_index*column_size) for column_index in range(number_of_columns)]
# cd_projection_list = [(i,chosen*column_size) for i,chosen in enumerate(chosen_columns)]
cd_projection_list = []

for index in range(len(chosen_columns)-num_columns_active_per_pattern):
    cd_projection_list.append((chosen_columns[index]*column_size,chosen_columns[index+num_columns_active_per_pattern]*column_size))

structure_model_with_stdp_pred = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model,
    weight=stdp_initial_weight,#0.,  # Use this weights when creating a new synapse
    max_weight=stdp_initial_weight,#0.005,#wpred/2.,#0.001,#0.001,#*0.9,#w_max,#TODO:try decreasing this
    s_max=5,  # Maximum allowed fan-in per target-layer neuron
    #grid=[np.sqrt(active_pop_size), np.sqrt(active_pop_size)],  # 2d spatial org of neurons
    random_partner=False,  # Choose a partner neuron for formation at random,
    # as opposed to selecting one of the last neurons to have spiked
    f_rew=10 ** 2,  #10 ** 4,  # Hz
    p_elim_dep=0.1,#1.,#0.5,#
    p_elim_pot=0.,
)

# cd_active_projection = sim.Projection(
#     active_pop,active_pop,
#     sim.FromListConnector(cd_projection_list),  # No initial connections
#     synapse_type=stdp_model,
#     # synapse_type=sim.StaticSynapse(weight=wpred),
#     label="cd -> active null_projection"
# )

# cd_active_projection = sim.Projection(
#      cd_pop,active_pop,
#     sim.FixedProbabilityConnector(0.0),  # No initial connections
#     synapse_type=sim.StaticSynapse(),
#     label="cd -> active null_projection"
# )

cd_active_projection = sim.Projection(
     active_pop,active_pop,
    sim.FixedProbabilityConnector(0.0),  # No initial connections
    synapse_type=structure_model_with_stdp_pred,
    #receptor_type="excitatory",
    label="cd -> active structurally_plastic_projection"
)
#================================================================================================
#  Noise to Active and CD projections
#================================================================================================
# noise_active_projection = sim.Projection(noise_pop_active,active_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))
# noise_cd_projection = sim.Projection(noise_pop_cd,cd_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))
#================================================================================================
#  Run simuluation
#================================================================================================

if get_weights:
    weights_cd = cd_active_projection.get("weight", "list", with_address=True)
    # weights = active_cd_projection.get("weight", "list", with_address=True)
    varying_weights=[]
    varying_weights_cd=[]

run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if get_weights:
        if run_one:
            # weights_list = []
            # for (source, target, weight) in weights:
            #     weights_list.append((source, target, weight))
            # varying_weights.append(weights_list)
            # weights_cd_list = []
            # for (source, target, weight) in weights_cd:
            #     weights_cd_list.append((source, target, weight))
            # varying_weights_cd.append(weights_cd_list)
            run_one = False
        # weights_cd = cd_active_projection.get("weight", "list", with_address=True)
        # weights_cd_list=[]
        # for (source,target,weight) in weights_cd:
        #     weights_cd_list.append((source,target,weight))
        # varying_weights_cd.append(weights_cd_list)

        # weights = active_cd_projection.get("weight", "list", with_address=True)
        # weights_list=[]
        # for (source,target,weight) in weights:
        #     weights_list.append((source,target,weight))
        # varying_weights.append(weights_list)

active_data =active_pop.get_data()
# cd_data = cd_pop.get_data()

sim.end()
num_recordings+=1
#================================================================================================
# Analysis and result export
#================================================================================================
chosen_ids =[[ ]for _ in range(len(chosen_columns))]
chosen_column_list = []
for i,chosen in enumerate(chosen_columns):
    chosen_column_list.append('{}'.format(chosen*column_size))
    for j in range(column_size):
        chosen_ids[i].append(chosen*column_size+j)
        chosen_ids[i].sort()
#chosen_ids.sort()
sr = math.sqrt(len(chosen_columns))
num_cols = np.ceil(sr)
num_rows = np.ceil(len(chosen_columns)/num_cols)
mem_v = active_data.segments[0].filter(name='v')
g_syn = active_data.segments[0].filter(name='gsyn_exc')

# potential_predictions = [(source,target) for (source,target,weight) in weights if target in chosen_ids]
# print potential_predictions
# target_active_cells = [target for (source,target,weight,delay) in varying_weights[0].connections[0]]
# target_active_cells = target_active_cells[:10]
# for i in range(active_pop_size):
#     if i not in target_active_cells:
#         inh_id = i
#         break

# results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
#                     '/HTM/{}_patterns_{}sequences_{}columns_{}active_neurons_{}Hz_{}cds_{}Taup_{}taumin_{}alpha_spike_pair_structural_plasticty'\
#                     .format(num_patterns_in_sequence,num_sequences,number_of_columns,column_size,column_firing_rate,cd_pop_size,tau_plus,tau_minus,a_plus)
#
results_directory = None
# results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
#                     '/HTM/2e2i'
# results_directory+='_nocdact_10timescale'
if results_directory is not None:
    if not os.path.isdir(results_directory):
        bashCommand = ["mkdir",results_directory]
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
        output, error = process.communicate()

# vary_weight_plot(varying_weights,target_active_cells,None,duration/1000.,
#                  plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),
#                  title='Predictive Neuron to Active Neuron Weight',
#                  filepath=results_directory)

if 0:#get_weights:
    weight_dist_plot(varying_weights_cd,1,plt,0.0,w_max,title="cd->active weight distribution",filepath=results_directory)
    # weight_dist_plot(varying_weights,1,plt,w_min_cd,w_max_cd,title="active->cd weight distribution",filepath=results_directory)
    connection_surface_plot(varying_weights_cd, pre_size=active_pop_size, post_size=active_pop_size, plt=plt,
                         title="prediction_weights", filepath=results_directory,n_plots=int(num_recordings))
    # target_neurons = [16]  # range(int(2))
    # vary_weight_plot(varying_weights_cd, target_neurons, None, duration / 1000.,
    #                  plt, np=numpy, num_recs=num_recordings, ylim=w_max + (w_max / 10.),
    #                  title='Predictive Neuron to Active Neuron Weight')
    # connection_surface_plot(varying_weights, pre_size=active_pop_size, post_size=cd_pop_size, plt=plt, title="active->cd",
    #                      filepath=results_directory,n_plots=int(num_recordings/2))
    # connection_hist_plot(varying_weights_cd, pre_size=cd_pop_size, post_size=active_pop_size, plt=plt,
    #                      title="cd->active", filepath=results_directory,weight_min=0.04)
    # connection_hist_plot(varying_weights, pre_size=active_pop_size, post_size=cd_pop_size, plt=plt, title="active->cd",
    #                      filepath=results_directory)

# Figure(
#     # plot data for postsynaptic neuron
#     Panel(active_data.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",legend=False,
#           yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(active_data.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",legend=False,
#            yticks=True,xticks=True, xlim=(0, duration)),
#     # Panel(active_data.segments[0].filter(name='gsyn_inh')[0],
#     #       ylabel="gsyn inhibitory (mV)",legend=False,
#     #        yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(active_data.segments[0].spiketrains,marker='.',
#           yticks=True,markersize=3,
#                  markerfacecolor='black', markeredgecolor='none',
#                  markeredgewidth=0,xticks=True, xlim=(0, duration)),
# )
spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity",filepath=results_directory,
                    )#onset_times=onset_times,pattern_duration=pattern_duration)
spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity_start",filepath=results_directory,
                    onset_times=onset_times,pattern_duration=pattern_duration,xlim=(onset_times[0][0],onset_times[0][1]))
spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity_final",filepath=results_directory,
                    onset_times=onset_times,pattern_duration=pattern_duration,xlim=(onset_times[0][-1],0.001*duration))
# spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,cd_pop_size+1,0.001,title="cd pop activity",filepath=results_directory,
#                     onset_times=onset_times,pattern_duration=pattern_duration)
# spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,cd_pop_size+1,0.001,title="cd pop activity_final",filepath=results_directory,
#                     onset_times=onset_times,pattern_duration=pattern_duration,xlim=(onset_times[0][-1],0.001*duration))

# plt.figure("mem_v")
# # for i,ids in enumerate(chosen_ids):
# #     cell_voltage_plot_8(mem_v, plt, duration, [],id=ids,title="col {}".format(chosen_column_list[i]),
# #                         subplots=[num_rows,num_cols,i+1])
cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.001*timestep,id=chosen_ids[num_columns_active_per_pattern],title="memv col {}".format(chosen_column_list[num_columns_active_per_pattern]))

cell_voltage_plot_8(g_syn, plt, duration, [],scale_factor=0.001*timestep,id=chosen_ids[num_columns_active_per_pattern],title="gsyn col {}".format(chosen_column_list[num_columns_active_per_pattern]))
# plt.figure("g_syn")
# for i,ids in enumerate(chosen_ids):
#     cell_voltage_plot_8(g_syn, plt, duration, [],id=ids,title="col {}".format(chosen_column_list[i]),
#                         subplots=[num_rows,num_cols,i+1])

# Figure(
#     # plot data for postsynaptic neuron
#     Panel(cd_data.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",legend=False,
#           yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",legend=False,
#            yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].filter(name='gsyn_inh')[0],
#           ylabel="gsyn inhibitory (mV)",legend=False,
#            yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].spiketrains,marker='.',
#           yticks=True,markersize=3,
#                  markerfacecolor='black', markeredgecolor='none',
#                  markeredgewidth=0,xticks=True, xlim=(0, duration)),
#
# )
# mem_v = active_data.segments[0].filter(name='v')
# pred_id = potential_predictions[0]
# cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=potential_predictions,title='Predicted Active Neuron',filepath=results_directory)
# cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=inh_id,title='Inhibited Active Neuron',filepath=results_directory)

if results_directory is not None:

    # final_active_spike_train = []
    # for train in active_data.segments[0].spiketrains:
    #     final_active_spike_train.append([time for time in train if time >= final_pattern_time])
    # final_cd_spike_train = []
    # for train in cd_data.segments[0].spiketrains:
    #     final_cd_spike_train.append([time for time in train if time >= final_pattern_time])

    # np.savez_compressed(results_directory + '/final_spike_trains', final_active_spike_train=final_active_spike_train,
    #          final_cd_spike_train=final_cd_spike_train, ms_onset_times=ms_onset_times)
    np.savez_compressed(results_directory + '/spike_trains', active_spike_train=active_data.segments[0].spiketrains,
                        ms_onset_times=ms_onset_times,mem_v=mem_v,g_syn=g_syn)

        # selective_neuron_search(ms_onset_times,active_data.segments[0].spiketrains,time_window=pattern_duration,
    #                         final_pattern_start =ms_onset_times[0][-1],plt=plt,filepath=results_directory,np=np,
    #                         significant_spike_count=1)
    if get_weights:
        np.savez_compressed(results_directory + '/varying_weights', varying_weights=varying_weights,
                 varying_weights_cd=varying_weights_cd)
        # connection_hist_plot(varying_weights_cd, pre_size=cd_pop_size, post_size=active_pop_size,plt=plt,title="cd->active",filepath=results_directory)
        # connection_hist_plot(varying_weights, pre_size=active_pop_size, post_size=cd_pop_size,plt=plt,title="active->cd",filepath=results_directory)

if 1:#results_directory is None:
    plt.show()
print
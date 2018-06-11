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
winh = 0.5
wpred = 0.01#w2s/2.#0.05#1.
w2s_target = 5.

input_pop_size =1
column_size = 16#32
number_of_columns = 20#100
active_pop_size = column_size*number_of_columns
cd_pop_size = 1 * active_pop_size
# assume 1% of 2048 columns are active per 1ms timestep
#if each column fired at 1Hz then there would be approx. 2 active columns per timestep
#we assume each column fires at around 10Hz, producing approx. 20 active columns per ms
column_firing_rate = 2.#10.
isi = 1000./column_firing_rate
num_firings = 60
predict_delay = 10#8
input_spikes = []
# for j in range(number_of_columns):
#     input_spikes.append([(j*column_offset)+i*isi for i in range(1,num_firings)])#[10.,30,50]
num_patterns_in_sequence = 4
num_sequences = 1 #ABCD XBCY
num_columns_active_per_pattern = 4#int(0.15*number_of_columns)
column_offset = 20.#int(isi/num_patterns_in_sequence)#isi/number_of_columns

#================================================================================================
# Pattern generation + column selection
#================================================================================================
#randomly chose column indices to represent
chosen_columns = np.random.choice(number_of_columns,num_columns_active_per_pattern*num_patterns_in_sequence,replace=False)
# chosen_columns = np.random.choice(number_of_columns,num_columns_active_per_pattern*6,replace=False)#ABCDXY

for pattern_index in range(num_patterns_in_sequence):
    for _ in range(num_columns_active_per_pattern):
        input_spikes.append([(pattern_index*column_offset)+i*isi for i in range(1,num_firings)])

# for j in range(num_sequences):
#     for pattern_index in range(num_patterns_in_sequence):
#         for _ in range(num_columns_active_per_pattern):
#             input_spikes.append([(pattern_index*column_offset)+i*isi + (j*isi/num_sequences) for i in range(1,num_firings)])
onset_times = []
ms_onset_times = []
for i in range(num_patterns_in_sequence*num_sequences):
    onset_times.append([time/1000. for time in input_spikes[num_columns_active_per_pattern*i]])
    ms_onset_times.append([time for time in input_spikes[num_columns_active_per_pattern*i]])

pattern_duration = 5. #actually just one spike but this helps visibility in plots and also accounts for any prop delay in activation

# predict_spikes = []
# for i,column_firings in enumerate(input_spikes):
#     predict_spikes.append([time-1-predict_delay*i for time in column_firings[5:]])
#     # predict_spikes.append([time+predict_delay for time in column_firings[5:]])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,64)
#================================================================================================
# Populations
#================================================================================================
# input_pop = sim.Population(num_columns_active_per_pattern*num_patterns_in_sequence,sim.SpikeSourceArray(spike_times=input_spikes))
input_pop = sim.Population(num_columns_active_per_pattern*num_patterns_in_sequence*num_sequences,sim.SpikeSourceArray(spike_times=input_spikes))
#one large population containing multiple 32 neuron 'columns'
active_pop =sim.Population(active_pop_size,sim.IF_curr_exp,cell_params,label="active_pop")
# cd_pop = sim.Population(num_columns_active_per_pattern*num_patterns_in_sequence,sim.SpikeSourceArray(spike_times=predict_spikes))
cd_pop = sim.Population(cd_pop_size,sim.IF_curr_exp,target_cell_params,label="cd")

active_pop.record(["spikes","v"])
cd_pop.record(["spikes"])

#================================================================================================
# Input to column projection
#================================================================================================
#each stimulus neuron of the input_pop will project to all the neurons in a corresponding single column of active pop
input_to_columns_list=[]
if num_sequences==1:
    for i,chosen in enumerate(chosen_columns):
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
                                  synapse_type=sim.StaticSynapse(weight=w2s,delay=1.))

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

#================================================================================================
#  Active to CD projection
#================================================================================================
initial_sync_num = 3.#num_columns_active_per_pattern#column_size#30.#15.#13.#30.#50.#
av_weight = w2s_target/initial_sync_num
w_max_cd = w2s_target/(initial_sync_num*0.5)#0.4)
w_min_cd = av_weight#0
a_plus_cd = 0.01
a_minus_cd = 0.01
tau_plus_cd = 16.
tau_minus_cd =30.
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

stdp_model_cd = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min_cd, w_max=w_max_cd), weight=av_weight,delay=1.)

# structure_model_w_stdp = sim.StructuralMechanismStatic(weight=w2s)#sim.StructuralMechanismSTDP(stdp_model=stdp_model_cd)

# active_cd_projection = sim.Projection(active_pop,cd_pop,sim.FixedProbabilityConnector(p_connect=0.025),synapse_type=stdp_model_cd)

# active_cd_projection = sim.Projection(active_pop,cd_pop,sim.FixedProbabilityConnector(p_connect=0.05),#(p_connect=0.05),#(p_connect=0.01),
#                                       synapse_type=sim.StaticSynapse(weight=av_weight))

# active_cd_projection =  sim.Projection(active_pop,cd_pop,sim.FixedNumberPreConnector(100),synapse_type=structure_model_w_stdp)

#connections from random cells in every column (limited to one per column)
sparse_active_cd_projection_list = []
num_cd_connections =num_columns_active_per_pattern
for cd_neuron in range(cd_pop_size):
    connected_columns = np.random.choice(number_of_columns,num_cd_connections,replace=False)
    for column_index in connected_columns:
        column_cell = column_index*column_size + np.random.randint(column_size)
        sparse_active_cd_projection_list.append((column_cell,cd_neuron))

# active_cd_projection = sim.Projection(active_pop,cd_pop,sim.FromListConnector(sparse_active_cd_projection_list),#(p_connect=0.05),#(p_connect=0.01),
#                                       synapse_type=sim.StaticSynapse(weight=av_weight))

structure_model_with_stdp = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd,
    weight=av_weight,  # Use this weights when creating a new synapse
    s_max=32,  # Maximum allowed fan-in per target-layer neuron
    #grid=[np.sqrt(active_pop_size), np.sqrt(active_pop_size)],  # 2d spatial org of neurons
    grid=[active_pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    # random_partner=True,  # Choose a partner neuron for formation at random,
    # as opposed to selecting one of the last neurons to have spiked
    f_rew=10 ** 4,  # Hz
    sigma_form_forward=10.,  # spread of feed-forward connections
    #delay=20  # Use this delay when creating a new synapse
)

active_cd_projection = sim.Projection(
    active_pop, cd_pop,
    sim.FixedProbabilityConnector(0.0),  # No initial connections
    synapse_type=structure_model_with_stdp,
    label="active -> cd structurally_plastic_projection"
)
#================================================================================================
#  CD to Active STDP + projection
#================================================================================================
tau_plus=16.
tau_minus=30.
a_plus =0.1#1#0.001#
a_minus =0.1#1.#0.5#1#0.001#
w_min = 0
w_max = wpred

stdp_initial_weight = 0.#RandomDistribution('uniform',(0.,w_max/10.))
stdp_delays = 14#RandomDistribution('uniform',(10.,14.))#1#

stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=stdp_initial_weight,delay=stdp_delays)

# cd_projection_list = [(column_index,column_index*column_size) for column_index in range(number_of_columns)]
# cd_projection_list = [(i,chosen*column_size) for i,chosen in enumerate(chosen_columns)]
cd_projection_list = []
for chosen in chosen_columns:
    for i in range(cd_pop_size):
        cd_projection_list.append((i,chosen*column_size))

# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),
#                                        synapse_type=sim.StaticSynapse(weight=wpred,delay=5.))
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=stdp_model)
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=sim.StaticSynapse(weight=wpred,delay=25.))
# p_connect = float(number_of_columns)/active_pop_size
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FixedProbabilityConnector(p_connect=0.1),synapse_type=stdp_model)
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FixedProbabilityConnector(p_connect=0.1),synapse_type=sim.StaticSynapse(weight=wpred,delay=25.))

#connect back to a random active cell in each column
sparse_cd_active_projection_list = []
num_connections = 1
for cd_neuron in range(cd_pop_size):
    connected_columns = range(number_of_columns)#np.random.choice(number_of_columns,num_connections,replace=False)
    for column_index in connected_columns:
        column_cell = column_index*column_size + np.random.randint(column_size)
        sparse_cd_active_projection_list.append((cd_neuron,column_cell))

# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(sparse_cd_active_projection_list),synapse_type=stdp_model)

structure_model_with_stdp_pred = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model,
    weight=0.,  # Use this weights when creating a new synapse
    s_max=128,  # Maximum allowed fan-in per target-layer neuron
    #grid=[np.sqrt(active_pop_size), np.sqrt(active_pop_size)],  # 2d spatial org of neurons
    grid=[cd_pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    #random_partner=True,  # Choose a partner neuron for formation at random,
    # as opposed to selecting one of the last neurons to have spiked
    f_rew=10 ** 4,  # Hz
    sigma_form_forward=10.,  # spread of feed-forward connections
    delay=14  # Use this delay when creating a new synapse
)

cd_active_projection = sim.Projection(
     cd_pop,active_pop,
    sim.FixedProbabilityConnector(0.1),  # No initial connections
    synapse_type=structure_model_with_stdp_pred,
    label="cd -> active structurally_plastic_projection"
)
#================================================================================================
#  Run simuluation
#================================================================================================
weights_cd = cd_active_projection.get("weight", "list", with_address=True)
weights = active_cd_projection.get("weight", "list", with_address=True)

duration = num_firings * isi
num_recordings =10

varying_weights=[]
varying_weights_cd=[]

run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if run_one:
        varying_weights.append(weights)
        varying_weights_cd.append(weights_cd)
        run_one = False
    weights_cd = cd_active_projection.get("weight", "list", with_address=True)
    weights = active_cd_projection.get("weight", "list", with_address=True)
    varying_weights.append(weights)
    varying_weights_cd.append(weights_cd)

active_data =active_pop.get_data(["spikes","v"])
cd_data = cd_pop.get_data(["spikes"])

sim.end()
num_recordings+=1
#================================================================================================
# Analysis and result export
#================================================================================================
chosen_ids =[]
for chosen in chosen_columns:
    for i in range(column_size):
        chosen_ids.append(chosen+i)
chosen_ids.sort()
# potential_predictions = [(source,target) for (source,target,weight) in weights if target in chosen_ids]
# print potential_predictions
# target_active_cells = [target for (source,target,weight,delay) in varying_weights[0].connections[0]]
# target_active_cells = target_active_cells[:10]
# for i in range(active_pop_size):
#     if i not in target_active_cells:
#         inh_id = i
#         break

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                    '/HTM/{}_patterns_{}sequences_{}columns_{}active_neurons_{}Hz_{}cds_{}Taup_{}taumin_{}alpha_spike_pair'\
                    .format(num_patterns_in_sequence,num_sequences,number_of_columns,column_size,column_firing_rate,cd_pop_size,tau_plus,tau_minus,a_plus)

results_directory = None
if results_directory is not None:
    if not os.path.isdir(results_directory):
        bashCommand = ["mkdir",results_directory]
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
        output, error = process.communicate()

# vary_weight_plot(varying_weights,target_active_cells,None,duration/1000.,
#                  plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),
#                  title='Predictive Neuron to Active Neuron Weight',
#                  filepath=results_directory)

weight_dist_plot(varying_weights_cd,1,plt,0.0,w_max,title="cd->active weight distribution",filepath=results_directory)
weight_dist_plot(varying_weights,1,plt,0.0,w_max_cd,title="active->cd weight distribution",filepath=results_directory)

spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity",filepath=results_directory,
                    onset_times=onset_times,pattern_duration=pattern_duration)
spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity_final",filepath=results_directory,
                    onset_times=onset_times,pattern_duration=pattern_duration,xlim=(onset_times[0][-1],0.001*duration))
spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,cd_pop_size+1,0.001,title="cd pop activity",filepath=results_directory)
spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,cd_pop_size+1,0.001,title="cd pop activity_final",filepath=results_directory,
                    onset_times=onset_times,pattern_duration=pattern_duration,xlim=(onset_times[0][-1],0.001*duration))

# mem_v = active_data.segments[0].filter(name='v')
# pred_id = potential_predictions[0]
# cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=potential_predictions,title='Predicted Active Neuron',filepath=results_directory)
# cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=inh_id,title='Inhibited Active Neuron',filepath=results_directory)

if results_directory is not None:

    selective_neuron_search(ms_onset_times,active_data.segments[0].spiketrains,time_window=pattern_duration,
                            final_pattern_start =ms_onset_times[0][-1],plt=plt,filepath=results_directory,np=np,
                            significant_spike_count=1)

if results_directory is None:
    plt.show()
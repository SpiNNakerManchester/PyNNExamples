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
model = sim.IF_curr_exp#sim.IF_cond_exp

# ex_params = {'tau_m': 20.,#10.0,#2.,#3.,#
#             'tau_refrac': 5.0,#2.0,#
#             'tau_syn_E': 8.0,#2.5,#
#             'tau_syn_I': 10.0,#2.5,#
#             'e_rev_I': -65.0,
#             'v_thresh': -50,
#             'v_reset': -100.0 # use a large v_reset to produce 'boosting'
#                }

ex_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 5.,#10.0,#2.,#3.,#
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 3.0,#2.5,#
               'tau_syn_I': 2.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

#low mem + high syn taus
inh_params = {'e_rev_I': -80.,  # nF
               'tau_m': 0.1,#10.0,#2.,#3.,#
               'tau_refrac': 5.0,#2.0,#
               'tau_syn_E': 15.0,#2.5,#
               'tau_syn_I': 5.0,#2.5,#
               'v_reset': -70.0,
               'v_thresh': -60.
               }

w2s =2.
winh = 2.0#1.0#0.5
winh_forward = 0.0000061
p_connect_forward = 1.0
wstim = 2./20.#10#0.0021#0.1
stim_jitter = 0.001#0.00005
waccum = 0.023

get_weights = True
#================================================================================================
# Open input
#================================================================================================
input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
# input_spikes = np.load(input_directory+"/IC_spikes/ic_spikes_asc_test_60s.npy")
input_file = np.load(input_directory+"/IC_spikes/long_spike_trains_test.npz")
input_spikes = input_file['ascending_audio_train_spikes']

ear_file = numpy.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spike_trains_asc_test_60s.npz")
onset_times = ear_file['onset_times']
# input_spikes = np.load(input_directory+"/ic_spikes_asc_train_60s.npy")
max_time = 0
for neuron in input_spikes:
    if neuron.size>0 and neuron.max() > max_time:
        max_time = neuron.max().item()

input_pop_size = len(input_spikes)
duration = max_time

input_size = len(input_spikes)
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=14.0)
sim.set_number_of_neurons_per_core(model,32)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(input_size,sim.SpikeSourceArray(spike_times=input_spikes))
# input_pop = sim.Population(1.*input_size,sim.SpikeSourcePoisson(rate=50.))
column_pop = sim.Population(input_size,model,ex_params,label="column_pop_fixed_weight_scale")
kill_switch = sim.Population(1,sim.IF_cond_exp,inh_params,label="kill_switch_pop")

input_pop.record(["spikes"])
column_pop.record(["spikes",'v'])
kill_switch.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
w_dist = RandomDistribution('normal', mu=wstim, sigma=stim_jitter)

av_weight = wstim
w_max_cd = av_weight*1.1#w2s_target/2.#
w_min_cd = av_weight*0.5#0
a_plus_cd = 0.5#1.#
a_minus_cd = 0.5#1.#
tau_plus_cd = 16.
tau_minus_cd =30.
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

stdp_model_cd = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min_cd, w_max=w_max_cd), weight=start_weight,delay=1.)

structure_model_with_stdp = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd,
    weight=av_weight,  # Use this weights when creating a new synapse
    max_weight=av_weight*0.9*2.,
    s_max=int(10.), # Maximum allowed fan-in per target-layer neuron
    grid=[input_pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    random_partner=False,
    #selecting one of the last neurons to have spiked
    f_rew=10 ** 4,  # Hz
    p_elim_dep=1.,#0.99,
    p_elim_pot=0.,#0.1,#
)

input_column_projection = sim.Projection(
    input_pop, column_pop,
    sim.FixedProbabilityConnector(p_connect=10./input_size),  # No initial connections
    synapse_type=structure_model_with_stdp,
    label="active -> cd structurally_plastic_projection"
)

# diagonal_width = 20.
# diagonal_sparseness = 1.
# in2out_sparse = .67 * .67 / diagonal_sparseness
# dist = max(int(input_pop_size / input_pop_size), 1)
# sigma = dist * diagonal_width
# conn_num = int(sigma / in2out_sparse)
#
# input_column_list = normal_dist_connection_builder(input_pop_size,input_pop_size,RandomDistribution,NumpyRNG(),
#                                             conn_num,dist,sigma,w_dist)
#
# input_column_projection = sim.Projection(input_pop,column_pop,sim.FromListConnector(input_column_list))

# input_column_projection = sim.Projection(input_pop,column_pop,sim.FixedProbabilityConnector(p_connect=10./input_size),#sim.OneToOneConnector(),
#                                   synapse_type=sim.StaticSynapse(weight=w_dist,delay=1.))

#aiming for about 20-40 active columns per timestep
n_active_cells = 10.
inhibited_cells = input_pop_size - n_active_cells
n_inh_targets_single_cell = inhibited_cells / n_active_cells
p_connect_inh = n_inh_targets_single_cell/input_pop_size

column_column_inh_projection = sim.Projection(column_pop,column_pop,sim.FixedProbabilityConnector(p_connect=p_connect_inh),
                                  synapse_type=sim.StaticSynapse(weight=0.5,delay=1.),receptor_type='inhibitory')

# column_kill_switch_projection = sim.Projection(column_pop,kill_switch,sim.AllToAllConnector(),
#                                                synapse_type=sim.StaticSynapse(weight=waccum,delay=1.))

# kill_switch_column_projection = sim.Projection(kill_switch,column_pop,sim.AllToAllConnector(),
#                                                synapse_type=sim.StaticSynapse(weight=winh,delay=1.),receptor_type='inhibitory')

# input_column_inh_projection = sim.Projection(input_pop,column_pop,sim.FixedProbabilityConnector(p_connect=p_connect_forward),
#                                   synapse_type=sim.StaticSynapse(weight=winh_forward,delay=1.),receptor_type='inhibitory')

#TODO: add hard wta inh between output layer neurons

#================================================================================================
#  Run simuluation
#================================================================================================
duration = max_time
max_period = 10000.#60000.#
num_recordings =int((duration/max_period)+1)

if get_weights:
    weights = input_column_projection.get("weight", "list", with_address=True)
    varying_weights = []

run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if get_weights:
        if run_one:
            weights_list = []
            for (source, target, weight) in weights:
                weights_list.append((source, target, weight))
            varying_weights.append(weights_list)
            run_one = False

        weights = input_column_projection.get("weight", "list", with_address=True)
        weights_list=[]
        for (source,target,weight) in weights:
            weights_list.append((source,target,weight))
        varying_weights.append(weights_list)

input_data = input_pop.get_data(["spikes"])
kill_switch_data = kill_switch.get_data(["spikes"])
output_data = column_pop.get_data(["spikes","v"])

sim.end()

#================================================================================================
# Analysis and result export
#================================================================================================
np.save(input_directory+'/IC_spikes/sparse_spikes.npy',output_data.segments[0].spiketrains)

spike_raster_plot_8(input_data.segments[0].spiketrains,plt,duration/1000.,input_size+1,0.001,title="input activity")
spike_raster_plot_8(kill_switch_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="kill switch activity")
spike_raster_plot_8(output_data.segments[0].spiketrains,plt,duration/1000.,input_size+1,0.001,title="output pop activity")

mem_v = output_data.segments[0].filter(name='v')
#choose 10 random ids to plot
ids=np.random.choice(range(input_pop_size),10,replace=False)
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=ids,title='output pop')

if get_weights:
    weight_dist_plot(varying_weights, 1, plt, 0.0, w_max_cd, title="input->column weight distribution")
    connection_hist_plot(varying_weights, pre_size=input_pop_size, post_size=input_pop_size, plt=plt, title="input->column")

plt.show()

# sparsity_matrix = sparsity_measure(onset_times,output_data.segments[0].spiketrains)
# np.save(input_directory+'/IC_spikes/sparsity_matrix.npy',sparsity_matrix)



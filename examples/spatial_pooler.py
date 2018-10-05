import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
import time

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

ex_params = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               #'tau_m': 10.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
               'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,#-100.0, # use a large v_reset to produce 'boosting'
               'v_rest': -65.0,
               'v_thresh':-56.# -55.6
               }

ex_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               # 'tau_m': 5.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
               'e_rev_E': -57.0,
               'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh': -55.4
               }

#low mem + high syn taus
# inh_params = {'e_rev_I': -80.,  # nF
#                'tau_m': 0.1,#10.0,#2.,#3.,#
#                'tau_refrac': 5.0,#2.0,#
#                'tau_syn_E': 15.0,#2.5,#
#                'tau_syn_I': 5.0,#2.5,#
#                'v_reset': -70.0,
#                'v_thresh': -60.
#                }

inh_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 20.0,#2.0,#TODO remove this but drop alpha -
               'tau_syn_E': 1.0,#2.5,
               #'tau_syn_I': 10.0,#2.5,#TODO:increase this to aid in fast inhib?
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

inh_cond_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 1.0,#2.0,#
               #'tau_syn_E': 1.0,#2.5,
               #'tau_syn_I': 1.0,#2.5,
               'e_rev_E': -50.0,
               'e_rev_I': 0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh':-45.0#-55.4#
               }

w2s = 3.0#4.0#1.5#
w2s_inh = 5.0
av_weight_divide =8.#4.#16.#
winh = 1.0#2.0#0.5#0.5#
winh_forward = 0.001#0.0000061
p_connect_forward = 1.0
n_input_conn = 16.#8.#5.#20.#
wstim = w2s/n_input_conn#20.#10#0.0021#0.1
stim_jitter = 0.001#0.00005
w2s_target = 5.#0.5#2.5#
get_weights = True#False#
n_struc_inputs_column = 2#3#5 #
n_struc_inputs_input_inh = 3
random_partner = False#True#
#================================================================================================
# Open input
#================================================================================================
input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
# input_spikes = np.load(input_directory+"/IC_spikes/ic_spikes_asc_test_60s.npy")
# input_file = np.load(input_directory+"/IC_spikes/brainstem_asc_des_60s_20dB.npz")
input_file = np.load(input_directory+"/IC_spikes/brainstem_asc_des_a_i_u_60s_20dB.npz")
input_spikes = input_file['ic_times']
onset_times = input_file['onset_times']
#extra_input = [np.asarray([]) if i>=1000 else input_spikes[i] for i in range(2000)]
#input_spikes = np.asarray(extra_input)
# ear_file = numpy.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spike_trains_asc_test_60s.npz")
# ear_file = numpy.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spinnakear_asc_des_60s.npz")
# onset_times = ear_file['onset_times']
# input_spikes = np.load(input_directory+"/ic_spikes_asc_train_60s.npy")
max_time = 0
for neuron in np.asarray(input_spikes):
    if neuron.size > 0 and neuron.max() > max_time:
        max_time = neuron.max().item()

input_pop_size = len(input_spikes)
duration = max_time

input_size = len(input_spikes)
n_columns = int(input_size/1.)
input_inh_size = int(n_columns * 1)

# inh_spikes = [[] for _ in range(input_inh_size)]
# chosen = np.random.choice(input_inh_size,)
# for stimulus_times in onset_times:
#     for time in stimulus_times:
#         for i in range(input_inh_size):
#             inh_spikes[i].append(time+((2.*np.random.rand())-0.5))
#aiming for about 2% active columns per timestep
n_active_cells = 0.02*n_columns
waccum = w2s_target/n_active_cells#1.5#1.0#0.05##0.023
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=14.0)
sim.set_number_of_neurons_per_core(model,32)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,32)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(input_size,sim.SpikeSourceArray(spike_times=input_spikes),label="input_pop")
# input_pop = sim.Population(1.*input_size,sim.SpikeSourcePoisson(rate=50.))
column_pop = sim.Population(n_columns,model,ex_params,label="column_pop_fixed_weight_scale")
# column_pop = sim.Population(n_columns,sim.IF_cond_exp,ex_params_cond,label="column_pop_fixed_weight_scale_cond")
input_inh_pop = sim.Population(input_inh_size,model,inh_params,label="input_inh_fixed_weight_scale")
# input_inh_pop = sim.Population(input_inh_size,model,ex_params,label="input_inh_fixed_weight_scale")
# input_inh_pop = sim.Population(input_inh_size,sim.SpikeSourceArray(spike_times=inh_spikes),label="input_inh_pop")

# input_inh_pop = sim.Population(input_size,sim.IF_cond_exp,inh_cond_params,label="input_inh_fixed_weight_scale_cond")
# kill_switch = sim.Population(1,model,inh_params,label="kill_switch_pop_fixed_weight_scale")


input_inh_pop.record(["spikes"])
# input_pop.record(["spikes"])
# # column_pop.record(["spikes",'v'])
column_pop.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
w_dist = RandomDistribution('normal', mu=wstim, sigma=stim_jitter)

av_weight = w2s/av_weight_divide#wstim
w_max_cd = av_weight*2.#w2s# av_weight*1.1#w2s_target/2.#
w_min_cd = 0.0#av_weight*0.5#0
a_plus_cd = 1.*0.1#0.5#1.#
a_minus_cd = 1.*0.1#0.05#0.25#1.#
tau_plus_cd = 16.
tau_minus_cd =30.#1.#
perc = av_weight/20.
start_weight = 0.01#av_weight#RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

stdp_model_cd = sim.STDPMechanism(
        # timing_dependence=sim.SpikePairRule(
        # # timing_dependence=sim.extra_models.SpikeNearestPairRule( #TODO: try this rule with -10ms inh input
        #     tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        timing_dependence=sim.extra_models.Vogels2011Rule(alpha=0.1, tau=20.0,
                                                          A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.MultiplicativeWeightDependence(
        #weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min_cd, w_max=w_max_cd), weight=start_weight,delay=1.)

structure_model_with_stdp = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd,
    weight=av_weight,  # Use this weights when creating a new synapse
    inh_weight=0.01,#0.1,#av_weight,# TODO:try initial higher weight with -10ms inh input
    #max_weight=av_weight*0.9*2.,
    s_max=int(n_input_conn*n_struc_inputs_column),# int(8.), #Maximum allowed fan-in per target-layer neuron
    grid=[1,n_columns], # 1d spatial org of neurons, uncomment this if wanted
    random_partner=random_partner,
    p_form_forward=1.,#
    p_form_lateral=1.,#
    sigma_form_forward=5.,
    #sigma_form_lateral=1.,
    f_rew=10 ** 4,  # Hz
    p_elim_dep=0.1,#0.0001,#0.9,#0.9,#0.99,#1.0,#
    p_elim_pot=0.0#0.1,#0.1,#
)

# input_column_projection = sim.Projection(
#     input_pop, column_pop,
#     sim.FixedProbabilityConnector(p_connect=0.),#n_input_conn/input_size),  # No initial connections
#     synapse_type=structure_model_with_stdp,
#     label="input -> column structurally_plastic_projection"
# )
#
# column_column_projection = sim.Projection(
#     column_pop, column_pop,
#     sim.FixedProbabilityConnector(p_connect=0.),#n_input_conn/input_size),  # No initial connections
#     synapse_type=structure_model_with_stdp,
#     label="column -> column structurally_plastic_projection"
# )

random_weights = RandomDistribution('uniform',(av_weight-perc,av_weight+perc))
sigma=2.
input_column_projection = sim.Projection(
    input_pop, column_pop,
    sim.FromListConnector(normal_dist_connection_builder(input_pop_size,n_columns,RandomDistribution,NumpyRNG(),n_input_conn,1.,sigma)),
    synapse_type=sim.StaticSynapse(weight=random_weights),
    label="input -> column fixed_projection"
)

# column_column_projection = sim.Projection(
#     column_pop, column_pop,
#     sim.FixedProbabilityConnector(p_connect=0.01),#n_input_conn/input_size),  # No initial connections
#     synapse_type=sim.StaticSynapse(weight=random_weights),
#     label="column -> column fixed_projection"
# )

# n_inputs_inh = 8.#5.#20.#
# inh_weight = w2s_target/n_inputs_inh#3.#
inh_weight = w2s_inh/av_weight_divide
inh_w_max = inh_weight*2.

stdp_model_cd_inh = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0, w_max=inh_w_max), weight=inh_weight,delay=1.)

structure_model_with_stdp_inh = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd,#stdp_model_cd_inh,
    weight=inh_weight,  # Use this weights when creating a new synapse
    inh_weight=0.01,#inh_weight,#
    #max_weight=inh_weight*0.9*2.,
    s_max=int(32.), #int(n_inputs_inh*n_struc_inputs_input_inh),# Maximum allowed fan-in per target-layer neuron
    grid=[1,input_pop_size], # 1d spatial org of neurons, uncomment this if wanted
    random_partner=random_partner,
    # sigma_form_forward=10.,
    #sigma_form_lateral=5.,
    sigma_form_forward=5.,
    #sigma_form_lateral=10.,#
    p_form_forward=1.,#
    p_form_lateral=1.,#0.1,
    f_rew=10**4,#10 ** 4,  # Hz
    p_elim_dep=0.1,#0.9,#1.0,#
    p_elim_pot=0.0#0.1,#
)

# input_input_inh_projection = sim.Projection(
#     input_pop, input_inh_pop,
#     sim.FixedProbabilityConnector(p_connect=0.),#n_input_conn/input_size),  # No initial connections
#     # synapse_type=structure_model_with_stdp_inh,
#     synapse_type=structure_model_with_stdp,
#     label="input_inh_proj"
# )
# input_inh_input_inh_projection = sim.Projection(
#     input_inh_pop, input_inh_pop,
#     sim.FixedProbabilityConnector(p_connect=0.),#n_input_conn/input_size),  # No initial connections
#     synapse_type=structure_model_with_stdp_inh,receptor_type='inhibitory'
# )

perc = inh_weight/20.
sigma = 2.
random_weights = RandomDistribution('uniform',(inh_weight-perc,inh_weight+perc))
input_input_inh_projection = sim.Projection(
    input_pop, input_inh_pop,
    sim.FromListConnector(normal_dist_connection_builder(input_pop_size,n_columns,RandomDistribution,NumpyRNG(),n_input_conn,1.,sigma)),
    synapse_type=sim.StaticSynapse(weight=random_weights),
    label="input -> column fixed_projection"
)

# input_inh_column_projection = sim.Projection(
#     input_inh_pop, column_pop,
#     sim.FixedProbabilityConnector(p_connect=0.0),  # No initial connections
#     synapse_type=structure_model_with_stdp,receptor_type='inhibitory',
#     label="inh -> column structurally_plastic_projection"
# )

sigma = 2.
input_inh_column_projection = sim.Projection(
    input_inh_pop, column_pop,
    sim.FromListConnector(normal_dist_connection_builder(input_pop_size,n_columns,RandomDistribution,NumpyRNG(),n_input_conn,1.,sigma)),
    #sim.FixedProbabilityConnector(p_connect=0.01),  # No initial connections
    synapse_type=stdp_model_cd,receptor_type='inhibitory',
    label="inh -> column fixed_projection"
)

# column_column_exc_projection = sim.Projection(column_pop,column_pop,sim.FixedProbabilityConnector(p_connect=0.0),  # No initial connections
#                                             synapse_type=structure_model_with_stdp)
# column_column_inh_projection = sim.Projection(column_pop,column_pop,sim.FixedProbabilityConnector(p_connect=0.0),  # No initial connections
#                                             synapse_type=structure_model_with_stdp,receptor_type='inhibitory')

# column_input_inh_projection = sim.Projection(column_pop,input_inh_pop,sim.FixedProbabilityConnector(p_connect=0.0),  # No initial connections
#                                             synapse_type=structure_model_with_stdp_inh)

# column_column_inh_projection = sim.Projection(column_pop,column_pop,sim.FromListConnector(column_column_list),
#                                   synapse_type=sim.StaticSynapse(weight=1.0,delay=1.),receptor_type='inhibitory')

# column_kill_switch_projection = sim.Projection(column_pop,kill_switch,sim.AllToAllConnector(),
#                                                synapse_type=sim.StaticSynapse(weight=waccum,delay=1.))
# column_kill_switch_projection = sim.Projection(column_pop,kill_switch,sim.FixedProbabilityConnector(p_connect=0.1),
#                                                synapse_type=sim.StaticSynapse(weight=waccum*10,delay=1.))

# kill_switch_column_projection = sim.Projection(kill_switch,column_pop,sim.AllToAllConnector(),
#                                                synapse_type=sim.StaticSynapse(weight=winh,delay=1.),receptor_type='inhibitory')

#================================================================================================
#  Run simuluation
#================================================================================================
duration = max_time#30000.#60000#
max_period = 8000.#60000.#
num_recordings =int((duration/max_period)+1)

varying_weights = []
varying_weights_inh = []

if get_weights:
    weights = input_column_projection.get("weight", "list", with_address=True)
    weights_inh = input_inh_column_projection.get("weight", "list", with_address=True)


run_one=False
start_time = time.time()
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if get_weights:
        if run_one:
            weights_list = []
            for (source, target, weight) in weights:
                weights_list.append((source, target, weight))
            varying_weights.append(weights_list)
            weights_list = []
            for (source, target, weight) in weights_inh:
                weights_list.append((source, target, weight))
            varying_weights_inh.append(weights_list)
            run_one = False

        weights = input_column_projection.get("weight", "list", with_address=True)
        # weights = input_inh_input_inh_projection.get("weight", "list", with_address=True)
        weights_inh = input_inh_column_projection.get("weight", "list", with_address=True)
        # weights_inh = input_input_inh_projection.get("weight", "list", with_address=True)
        # weights_inh = input_inh_input_inh_projection.get("weight", "list", with_address=True)

        weights_list=[]
        for (source,target,weight) in weights:
            weights_list.append((source,target,weight))
        varying_weights.append(weights_list)
        weights_list = []
        for (source, target, weight) in weights_inh:
            weights_list.append((source, target, weight))
        varying_weights_inh.append(weights_list)


# input_data = input_pop.get_data(["spikes"])
input_inh_data = input_inh_pop.get_data(["spikes"])
# kill_switch_data = kill_switch.get_data(["spikes"])
# output_data = column_pop.get_data(["spikes","v"])
output_data = column_pop.get_data(["spikes"])

run_time = time.time() - start_time

sim.end()

#================================================================================================
# Analysis and result export
#================================================================================================
onset_times_s = []
for times in onset_times:
    onset_times_s.append([time/1000. for time in times])

spike_raster_plot_8(input_inh_data.segments[0].spiketrains,plt,duration/1000.,input_size+1,0.001,title="inh activity",
                    onset_times=onset_times_s,pattern_duration=100.)
# spike_raster_plot_8(input_data.segments[0].spiketrains,plt,duration/1000.,input_size+1,0.001,title="input activity",
#                     onset_times=onset_times_s,pattern_duration=100.)
spike_raster_plot_8(output_data.segments[0].spiketrains,plt,duration/1000.,n_columns+1,0.001,title="output pop activity",
                    onset_times=onset_times_s,pattern_duration=100.)

# mem_v = output_data.segments[0].filter(name='v')
mem_v=[]
inh_pop_spikes=input_inh_data.segments[0].spiketrains
# inh_pop_spikes=[]
column_spikes = output_data.segments[0].spiketrains
np.savez_compressed(input_directory+'/spatial_pooler_mult',sparse_mem_v=mem_v,
         column_spikes = column_spikes,inh_pop_spikes=inh_pop_spikes,varying_weights=varying_weights,
         varying_weights_inh=varying_weights_inh,run_time=run_time,sim_duration=duration)

sparsity_matrix = sparsity_measure(onset_times,output_data.segments[0].spiketrains,onset_window=100.,from_time=0.)
plt.figure('output pop sparsity')
for stimulus in sparsity_matrix:
    plt.plot(stimulus)
    plt.savefig(input_directory+'/sparsity.eps')

# sparsity_matrix = sparsity_measure(onset_times,input_inh_data.segments[0].spiketrains,onset_window=100.,from_time=0)
# plt.figure('inh pop sparsity')
# for stimulus in sparsity_matrix:
#     plt.plot(stimulus)
#
# sparsity_matrix = sparsity_measure(onset_times,input_spikes,onset_window=100.,from_time=0)
# plt.figure('input sparsity')
# for stimulus in sparsity_matrix:
#     plt.plot(stimulus)


#choose 10 random ids to plot
# ids=np.random.choice(range(input_pop_size),10,replace=False)
# cell_voltage_plot_8(mem_v, plt, duration,[],scale_factor=0.001,title='output pop')

if 0:#get_weights:
   # weight_dist_plot(varying_weights, 1, plt, 0.0, inh_weight*0.9*2., title="input->input_inh weight distribution")
   weight_dist_plot(varying_weights, 1, plt, 0.0, w_max_cd, title="input->column weight distribution")
   connection_hist_plot(varying_weights, pre_size=input_pop_size, post_size=input_pop_size, plt=plt, title="input->column")#,weight_min=av_weight)
   # weight_dist_plot(varying_weights, 1, plt, 0.0, inh_w_max, title="input_inh->input_inh weight distribution")
   # connection_hist_plot(varying_weights, pre_size=input_pop_size, post_size=input_pop_size, plt=plt, title="input_inh->input_inh",weight_min=0.01)
   # weight_dist_plot(varying_weights_inh, 1, plt, 0.0, w_max_cd, title="inh->column weight distribution")
   # connection_hist_plot(varying_weights_inh, pre_size=input_inh_size, post_size=n_columns, plt=plt, title="inh->column")#,weight_min=0.01)
   weight_dist_plot(varying_weights_inh, 1, plt, 0.0, inh_w_max, title="input->inh weight distribution")
   connection_hist_plot(varying_weights_inh, pre_size=input_pop_size, post_size=input_inh_size, plt=plt, title="input->inh")
print "plotting"
plt.show()

# sparsity_matrix = sparsity_measure(onset_times,output_data.segments[0].spiketrains)
# np.save(input_directory+'/IC_spikes/sparsity_matrix.npy',sparsity_matrix)

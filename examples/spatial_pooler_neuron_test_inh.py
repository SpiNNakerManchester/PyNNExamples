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
              # 'tau_syn_E': 1.0,#2.5,
              # 'tau_syn_I': 1.0,#2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

inh_cond_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 10.0,#2.5,
               #'tau_syn_I': 1.0,#2.5,
               'e_rev_E': -50.0,
               'e_rev_I': 0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh':-45.0#-55.4#
               }

neuron_params = {
    "v_thresh": 100,
    "v_reset": 0,
    "v_rest": 0,
    "i_offset": 0,
    "e_rev_E": 80,
#     "tau_syn_E":50,
    "e_rev_I": 0 # DC input
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

ex_params = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               #'tau_m': 10.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
               # 'tau_syn_I': 10.0,#2.5,#TODO:see if this can be set to default
                'v_reset': -70.0,#
                'v_rest': -65.0,
                'v_thresh': -55.6
               }

inh_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 20.0,#2.0,#TODO remove this but drop alpha -
               'tau_syn_E': 1.0,#2.5,
               # 'tau_syn_I': 10.0,#2.5,#TODO:increase this to aid in fast inhib?
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

cd_pop_size = 1000
w2s_target = 3.0#0.12#0.05#2.5#5.
w2s_inh = 5.0
n_connections = 16
initial_weight = w2s_target/n_connections
connection_weight = initial_weight#*2.#/2.
number_of_inputs = 1000
n_inh = number_of_inputs#n_connections-number_of_inputs
inh_weight = initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

av_weight =w2s_target/4.#initial_weight #0.05
w_max_cd = av_weight*2#1.1#initial_weight*2#w2s_target/2.#
w_min_cd = 0.0#av_weight*0.5#0
a_plus_cd = 0.1#1.#
a_minus_cd = 0.025#0.1#1.#
tau_plus_cd = 16.
tau_minus_cd =30.#1.#
ten_perc = av_weight/10.
start_weight = av_weight#RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

#TODO investigate a different inh max weight mechanism.
#TODO achieve a balance between competing exc and inh inputs, if they are balanced it doesnt matter how many
#TODO investigate fast INH neurons with large number of inputs, these should reliably fire (in time) to general input activity levels
#TODO need to parameter tune to balance the inh growth with exc
stdp_model_cd = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
        # timing_dependence=sim.extra_models.SpikeNearestPairRule(
            tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min_cd, w_max=w_max_cd), weight=start_weight,delay=1.)
structure_model_with_stdp = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd,
    weight=av_weight,  # Use this weight when creating a new synapse
    inh_weight=0.01,#av_weight,#
    #max_weight=av_weight*0.9*2.,
    s_max=int(n_connections*2),#int(10.), # Maximum allowed fan-in per target-layer neuron
    # grid=[input_pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    # grid=[number_of_inputs, 1], # 1d spatial org of neurons, uncomment this if wanted
    grid=[1,number_of_inputs], # 1d spatial org of neurons, uncomment this if wanted
    random_partner=True,#False,#
    p_form_forward=1.0,
    #sigma_form_forward=10.,
    #sigma_form_lateral=5.,
    #p_form_forward=0.1,
    f_rew=10**4,#10 ** 4,  # Hz
    #p_elim_dep=0.99,#0.9,#0.99,#1.0,#
    #p_elim_pot=0.0,#0.1,#
)

inh_weight_init = w2s_inh/4.
inh_w_max = w2s_inh/2.

stdp_model_cd_inh = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=16., tau_minus=30., A_plus=0.1, A_minus=0.025),#A_minus=0.1),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0, w_max=inh_w_max), weight=inh_weight_init,delay=1.)

structure_model_with_stdp_inh = sim.StructuralMechanismSTDP(
    stdp_model=stdp_model_cd_inh,
    weight=inh_weight_init,  # Use this weights when creating a new synapse
    inh_weight=0.01,
    #max_weight=inh_weight*0.9*2.,
    s_max=int(64), #int(128*1),# Maximum allowed fan-in per target-layer neuron
    # grid=[number_of_inputs, 1], # 1d spatial org of neurons, uncomment this if wanted
    grid=[1,number_of_inputs], # 1d spatial org of neurons, uncomment this if wanted
    random_partner=True,#False,#
    p_form_forward=1.0,
    p_form_lateral=1.0,
    sigma_form_forward=10.,
    #sigma_form_lateral=10.,
    f_rew=10**4,#10 ** 4, # Hz
    #p_elim_dep=0.9,#1.0,#0.1,#
    #p_elim_pot=0.1,#
)

#================================================================================================
# Generate Inputs
#================================================================================================

input_spikes =[]
n_repeats = 40#0
isi = 200.#100.
jitter_input = 25.#10.
jitter_inh = 10.

# out of a pop of 1000 neurons we measure stim 1 to have 2.7% activity across a duration of 100ms
# therefore there's a 1000*rate number of active neurons in the stimulus presentation window of 100ms
# how do we divide this up?
# we must assume that a less sparse rep uses fewer neurons. so each stimulus will have similar rates
stim_1_rate = 1.7
stim_2_rate = 2.4
stim_1_prop = stim_1_rate/(stim_1_rate+stim_2_rate)
n_stim_1_neurons = number_of_inputs*stim_1_prop
stim_2_prop = stim_2_rate/(stim_1_rate+stim_2_rate)
n_stim_2_neurons = number_of_inputs*stim_2_prop

stim_isi_repeats = int(np.round(((stim_1_rate/100.)*number_of_inputs*jitter_input)/n_stim_1_neurons))
# stim_2_isi_repeats = ((stim_2_rate/100.)*number_of_inputs*jitter_input)/n_stim_2_neurons

input_spikes = [[]for _ in range(number_of_inputs)]

# stim_1_ids = np.random.choice(range(number_of_inputs),number_of_inputs*0.4,replace=False)
# stim_2_ids = np.random.choice(range(number_of_inputs),number_of_inputs*0.2,replace=False)
stim_1_ids = np.random.choice(range(number_of_inputs),n_stim_1_neurons,replace=False)
stim_2_ids = np.random.choice(range(number_of_inputs),n_stim_2_neurons,replace=False)

onset_times = []
# onset_times.append([i*isi - jitter_input/2. for i in range(n_repeats)])
# onset_times.append([i*isi + isi/2. - jitter_input/2. for i in range(n_repeats)])
onset_times.append([i*isi for i in range(n_repeats)])
onset_times.append([i*isi + isi/2. for i in range(n_repeats)])

for i in range(n_repeats):

    for neuron in stim_1_ids:
        for _ in range(stim_isi_repeats):
            input_spikes[neuron].append(i * isi + int(jitter_input * np.random.rand()))#int(jitter_input * (np.random.rand() - 0.5))
    for neuron in stim_2_ids:
        for _ in range(stim_isi_repeats):
            input_spikes[neuron].append(i * isi + isi/2. + int(jitter_input * np.random.rand()))#+ int(jitter_input * (np.random.rand() - 0.5))+ int(np.random.rand()*isi/2.))

# spike_raster_plot_8(input_spikes, plt, (n_repeats*isi) / 1000., number_of_inputs + 1, 0.001, title="input activity")
# plt.show()
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,32)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="input_pop")
# inh_pop = sim.Population(n_inh,sim.SpikeSourceArray(spike_times=inh_spikes),label="inh_pop")
inh_pop = sim.Population(n_inh,sim.IF_curr_exp,inh_params,label="fast_inh_fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_cond_exp,inh_cond_params,label="fixed_weight_scale_cond")
cd_pop = sim.Population(cd_pop_size,sim.IF_curr_exp,ex_params,label="target_fixed_weight_scale")
# noise_pop = sim.Population(1,sim.SpikeSourcePoisson(rate=10.),label="noise_pop")

cd_pop.record(["spikes","v"])
inh_pop.record(["spikes","v"])
# noise_pop.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
# input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
input_projection = sim.Projection(input_pop,cd_pop,sim.FixedProbabilityConnector(0.0),synapse_type=structure_model_with_stdp)
# inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
inh_column_projection = sim.Projection(inh_pop,cd_pop,sim.FixedProbabilityConnector(0.0),synapse_type=structure_model_with_stdp,receptor_type='inhibitory')
# noise_projection = sim.Projection(noise_pop,cd_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))
# noise_projection = sim.Projection(noise_pop,cd_pop,sim.FixedProbabilityConnector(0.0),synapse_type=structure_model_with_stdp)

input_inh_projection = sim.Projection(
    input_pop, inh_pop,
    sim.FixedProbabilityConnector(p_connect=0.),#n_input_conn/input_size),  # No initial connections
    synapse_type=structure_model_with_stdp_inh,
    label="input_inh_proj"
)
inh_inh_projection = sim.Projection(
    inh_pop, inh_pop,
    sim.FixedProbabilityConnector(p_connect=0.0),#n_input_conn/input_size),  # No initial connections
    synapse_type=structure_model_with_stdp_inh,receptor_type='inhibitory'
)

# for inh pop we want large number of ff connections but then strong lateral inh connectivity
# to quickly supress inh neuron firing after stimulus onset
# input_inh_projection = sim.Projection(
#     input_pop, inh_pop,
#     sim.FixedProbabilityConnector(p_connect=0.1),
#     synapse_type=sim.StaticSynapse(weight=w2s_inh/10.))

# inh_inh_projection = sim.Projection(inh_pop, inh_pop,sim.FixedProbabilityConnector(p_connect=0.1),
#                                     synapse_type=sim.StaticSynapse(weight=w2s_inh/5.),receptor_type='inhibitory')

#================================================================================================
# Run Simulation
#================================================================================================
duration = n_repeats * isi
if duration<=10000:
    get_weights = True#False#
else:
    get_weights=False

max_period = 1000.#60000.#
num_recordings =int((duration/max_period)+1)
varying_weights = []
varying_weights_inh = []
if get_weights:
    weights = input_projection.get("weight", "list", with_address=True)

run_one=False#True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if get_weights:
        if run_one:
            weights_list = []
            for (source, target, weight) in weights:
                weights_list.append((source, target, weight))
            varying_weights.append(weights_list)
            run_one = False

        weights = input_inh_projection.get("weight", "list", with_address=True)
        weights_list=[]
        for (source,target,weight) in weights:
            weights_list.append((source,target,weight))
        varying_weights.append(weights_list)

        weights_inh = inh_inh_projection.get("weight", "list", with_address=True)
        weights_list = []
        for (source, target, weight) in weights_inh:
            weights_list.append((source, target, weight))
        varying_weights_inh.append(weights_list)

else:
    sim.run(duration)

num_recordings+=1

cd_data = cd_pop.get_data(["spikes","v"])
inh_data = inh_pop.get_data(['spikes','v'])
inh_spikes = inh_data.segments[0].spiketrains
# noise_data = noise_pop.get_data(["spikes"])

sim.end()

mem_v = cd_data.segments[0].filter(name='v')
mem_v = inh_data.segments[0].filter(name='v')

if duration<=10000:
    xlim=None
    cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.001,title='cd pop')
    cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.001,title='inh pop')
else:
    xlim=((duration-10000.)/1000.,duration/1000.)#final 10s

np.savez('./spatial_pooler',sparse_mem_v=mem_v,column_spikes = cd_data.segments[0].spiketrains,
         inh_pop_spikes=inh_spikes,varying_weights=varying_weights,varying_weights_inh=varying_weights_inh,
         onset_times=onset_times,onset_window=jitter_input)

sparsity_matrix = sparsity_measure(onset_times,cd_data.segments[0].spiketrains,onset_window=jitter_input,from_time=0)

plt.figure()
for stimulus in sparsity_matrix:
    plt.plot(stimulus)

# spike_raster_plot_8(noise_data.segments[0].spiketrains,plt,duration/1000.,8001,0.001,title="noise input activity")
spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,cd_pop_size+1,0.001,title="cd pop activity",xlim=xlim)
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity",xlim=xlim)
spike_raster_plot_8(inh_spikes,plt,duration/1000.,n_inh+1,0.001,title="inh activity",xlim=xlim)

if get_weights:
    # vary_weight_plot(varying_weights,range(1),[],duration/1000.,
    #                              plt,np=numpy,num_recs=num_recordings,ylim=w_max_cd+(w_max_cd/10.))
    # weight_dist_plot(varying_weights, 1, plt, 0.0, w_max_cd, title="input->column weight distribution")
    # weight_dist_plot(varying_weights, 1, plt, 0.0, w_max_cd, title="inh->column weight distribution")
    weight_dist_plot(varying_weights_inh, 1, plt, 0.0, inh_w_max, title="inh->inh weight distribution")
    weight_dist_plot(varying_weights, 1, plt, 0.0, inh_w_max, title="input->inh weight distribution")
    # connection_hist_plot(varying_weights, pre_size=number_of_inputs, post_size=n_inh, plt=plt, title="inh->column")
    connection_hist_plot(varying_weights_inh, pre_size=n_inh, post_size=n_inh, plt=plt, title="inh->inh")
    connection_hist_plot(varying_weights, pre_size=number_of_inputs, post_size=n_inh, plt=plt, title="input->inh")

plt.show()
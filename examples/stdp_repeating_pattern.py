import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import math
import logging
import os
import subprocess

#function that calculates minimum alphas based on w_max and tau_minus/tau_plus
def stdp_param_check(alpha_plus,alpha_minus,w_max,tau_minus,tau_plus):

    min_w_delta = w_max/2.**16
    max_tau_plus_delta = -math.log((min_w_delta/alpha_plus))*tau_plus
    max_tau_minus_delta = -math.log((min_w_delta/alpha_minus))*tau_minus

    return max_tau_plus_delta,max_tau_minus_delta,min_w_delta

def probability_of_one_post_respond_to_stimulus(p_connect,n_pattern_neurons,n_post_neurons):
    return (p_connect**n_pattern_neurons) * n_post_neurons

# Population parameters
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
               'tau_m': 10.0,#2.,#3.,#
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

#simulation parameters
stim_size = 1000
target_pop_size = 32#128
num_pattern_neurons = 10#20#100#stim_size*0.1#100#
max_sync_pattern_neurons = num_pattern_neurons/2.
p_connect = 0.35#0.6#1.0#
p_response=probability_of_one_post_respond_to_stimulus(p_connect=p_connect,n_pattern_neurons=max_sync_pattern_neurons,
                                                       n_post_neurons=target_pop_size)
# if p_response<1.:
#     raise IOError("the probability that one post neuron will respond to a pattern is {} reconsider simulation parameters".format(p_response))
number_of_firings_per_pattern =50#10#1# TODO: get to root of this max limit
num_patterns =2#1#5#
noise_rate = 20.#20.#1.#TODO:derive equation for the ratio of this and num_pattern_neurons (SNR) from fan in
#num_recordings = 10#2#
duration = 360. * 1000.
num_recordings = int(numpy.ceil(duration/4000))
w2s =2.#3.#
w2s_target = 5.
w_min = 0.#w2s_target/num_pattern_neurons#
if num_pattern_neurons>0:
    av_weight = w2s_target/(num_pattern_neurons)#w2s_target/num_pattern_neurons#w_max/2.# w_max/3.#
    w_max = w2s_target / max_sync_pattern_neurons  # w2s_target/2#w2s/2.#2*w2s/num_pattern_neurons#w2s/10#w2s/stim_size#1.#1.#
else:
    av_weight = w2s_target/10.
    w_max = w2s_target
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))#av_weight#
tau_plus=16
tau_minus=30.
a_plus =0.01#0.001#
a_minus =0.01#0.001#
nearest_pair = True

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,16)

pattern_duration = 1000#200#100#
#calculate maximum repeat rate with at least 0.5s between patterns
pattern_repeat_rate = 1000./(num_patterns*(pattern_duration+200))#0.5#0.05#5.#6.#
# pattern_repeat_rate = 1
num_pattern_presentations = int((duration/1000.)*pattern_repeat_rate)
pattern_repeat_period = 1000./pattern_repeat_rate
stim_times = []
for j in range(num_patterns):
    stim_times.append([])
    min_distance = 0
    while min_distance < 4:#a pattern with a higher rate can't be encoded by a single neuron (rate limit ~200Hz)
        pattern = np.random.choice(pattern_duration, number_of_firings_per_pattern, replace=False)
        pattern = sorted(pattern)
        diff = float("inf")
        for t in range(len(pattern)-1):
            t_diff = abs(pattern[t]-pattern[t+1])
            if t_diff<diff:
                diff = t_diff
        min_distance = diff
    # pattern = [1,20,23,30,65,80,132,145,164,170]
    for i in range(num_pattern_presentations):
    # for i in range(num_pattern_presentations/num_patterns):
        #variation in pattern start time, should not overlap
        random_variation = 0#int(((pattern_repeat_period/num_patterns)-pattern_duration)*(np.random.rand()))#
        for beep in pattern:
            stim_times[j].append(int(beep+i*pattern_repeat_period+random_variation+j*(pattern_repeat_period/num_patterns)))
            # stim_times[j].append(int(beep+i*pattern_repeat_period+random_variation+j*pattern_repeat_period*(num_pattern_presentations/num_patterns)))

    stim_times[j].sort()

# spike_raster_plot_8(stim_times,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity",pattern_times=stim_times,pattern_duration=pattern_duration)
# plt.show()

final_pattern_start_time = stim_times[0][-number_of_firings_per_pattern]

# plt.show()
pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
#pre_pop = sim.Population(stim_size,sim.Izhikevich,{},label="pre_pop")
pre_pop.record("spikes")

target_pop = sim.Population(target_pop_size,sim.IF_curr_exp,target_cell_params,label="target")
#target_pop = sim.Population(1,sim.Izhikevich,{},label="target")
target_pop.record(["spikes"])

target_inh = sim.Population(target_pop_size,sim.IF_curr_exp,cell_params,label="inh")
target_inh.record(["spikes"])

# ac2acinh_proj = sim.Projection(target_pop,target_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
# acinh2ac_proj = sim.Projection(target_inh,target_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1,delay=1.),
#                                receptor_type='inhibitory')
pattern_pops=[]
for i in range(num_patterns):
    pattern_pops.append(sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times[i])))

if num_pattern_neurons>0:
    ext_stim = sim.Population(
        stim_size - num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        label="stim_poisson_{}Hz".format(noise_rate))

    noise2pattern_pop = sim.Population(
       num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=final_pattern_start_time-1),
         label = "pattern_stim_poisson_{}Hz".format(noise_rate))
    # noise2pattern_pop = sim.Population(
    #     num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate*0.9, duration=duration*0.9),#90% duration to examine target pattern
    #       label = "pattern_stim_poisson_{}Hz".format(noise_rate))
    #increased pattern pop noise
    # noise2pattern_pop = sim.Population(
    #     num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate*10, duration=duration * 0.9),
    #     label="pattern_stim_poisson_{}Hz".format(noise_rate))

    chosen_int = numpy.random.choice(stim_size,num_patterns*num_pattern_neurons,replace=False).tolist()
    pattern_proj_list = []
    noise_proj_list = []
    noise2pattern_proj_list =[]
    noise_index = 0
    pre_index = 0
    for post in range(stim_size):
        if post not in chosen_int:
            noise_proj_list.append((pre_index,post))
            pre_index+=1
    for i in range(num_patterns):
        pattern_indices = chosen_int[i*num_pattern_neurons:(i+1)*num_pattern_neurons]
        pattern_proj_list.append([])
        for count,post in enumerate(pattern_indices):
            pattern_proj_list[i].append((0, post))
            noise2pattern_proj_list.append((i*num_pattern_neurons+count,post))

    pattern_projs = []
    for i in range(num_patterns):
        pattern_projs.append(sim.Projection(pattern_pops[i],pre_pop,
                                            sim.FromListConnector(pattern_proj_list[i]),
                                            synapse_type=sim.StaticSynapse(weight=w2s)))

    noise2pattern_proj = sim.Projection(noise2pattern_pop,pre_pop,
                                        sim.FromListConnector(noise2pattern_proj_list),
                                        synapse_type=sim.StaticSynapse(weight=w2s))
    noise_proj = sim.Projection(ext_stim,pre_pop,sim.FromListConnector(noise_proj_list),
                                synapse_type=sim.StaticSynapse(weight=w2s))

else:
    ext_stim = sim.Population(
        stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        label="stim_poisson_{}Hz".format(noise_rate))

    noise_proj = sim.Projection(ext_stim,pre_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))
    chosen_int=[]

# Plastic Connection between pre_pop and post_pop
if nearest_pair:
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.extra_models.SpikeNearestPairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=start_weight)
else:
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=start_weight)

stdp_proj=sim.Projection(
    #pre_pop, target_pop, sim.AllToAllConnector(), receptor_type='excitatory',
    pre_pop, target_pop,sim.FixedProbabilityConnector(p_connect=p_connect),receptor_type='excitatory',
    #synapse_type=sim.StaticSynapse(weight=av_weight))
    synapse_type=stdp_model)
weights = stdp_proj.get("weight", "list", with_address=True)

#target noise for STDP stable tests TODO: this won't allow for synapse strengthening when enabled
target_noise = sim.Population(target_pop_size, sim.SpikeSourcePoisson(rate=1., duration=final_pattern_start_time-1))
target_noise_proj = sim.Projection(target_noise,target_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))

varying_weights=[]
run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if run_one:
        varying_weights.append(weights)
        run_one = False
    weights = stdp_proj.get("weight", "list", with_address=True)
    varying_weights.append(weights)

target_data =target_pop.get_data(["spikes"])
inh_data = target_inh.get_data(['spikes'])
stim_data = pre_pop.get_data("spikes")

sim.end()
num_recordings+=1
weight = [weight for (pre, post, weight) in weights]

if nearest_pair:
    results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                        '/nearest_pair_stdp/{}_patterns_{}pre{}post_{}patternneurons_{}Hznoise_{}s_{}alpha_P_conn{}'.format(num_patterns,stim_size,
                                                                                     target_pop_size,num_pattern_neurons,
                                                                                      int(noise_rate),int(duration/1000.),a_plus,p_connect)

else:
    results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                        '/pair_stdp/{}_patterns_{}pre{}post_{}patternneurons_{}Hznoise_{}s_{}alpha_P_conn{}'.format(
                                                                                    num_patterns, stim_size,
                                                                                    target_pop_size, num_pattern_neurons,
                                                                                    int(noise_rate), int(duration / 1000.), a_plus,p_connect)

#results_directory = None
if results_directory is not None:
    if not os.path.isdir(results_directory):
        bashCommand = ["mkdir",results_directory]
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
        output, error = process.communicate()

    np.save(results_directory+'/pattern_spikes.npy',stim_times)
    np.save(results_directory+'/target_spikes.npy',target_data.segments[0].spiketrains)
    np.save(results_directory + '/final_weights.npy',weights)

else:
    np.save('./pattern_spikes.npy',stim_times)
    np.save('./target_spikes.npy',target_data.segments[0].spiketrains)

print "max weight = {}, min weight = {}".format(max(weight),min(weight))
for i,pattern in enumerate(stim_times):
    print "pattern {} times: {}".format(i+1,pattern)
print "p response: {}".format(p_response)
# obtain onset times for pattern
onset_times = []
for i, stimulus in enumerate(stim_times):
    onset_times.append([])
    onset = 0
    for time in stimulus:
        if time >= onset:
            onset_times[i].append(time * 0.001)
            onset = time + pattern_duration
if target_pop_size <= 100:
    vary_weight_plot(varying_weights,range(int(target_pop_size)),chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),filepath=results_directory)

    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop activity",filepath=results_directory,onset_times=onset_times,pattern_duration=pattern_duration)
    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop final activity",filepath=results_directory,xlim=(0.001*(final_pattern_start_time-1),0.001*duration),
                        onset_times=onset_times,pattern_duration=pattern_duration)
else:
    #vary_weight_plot(varying_weights,range(100),chosen_int,duration/1000.,
    # vary_weight_plot(varying_weights,chosen_int,chosen_int,duration/1000.,
    #                          plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))
    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop activity",filepath=results_directory,pattern_times=onset_times,pattern_duration=pattern_duration)
    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop final activity",filepath=results_directory,xlim=(0.001*(final_pattern_start_time-1),0.001*duration),
                        pattern_times=onset_times,pattern_duration=pattern_duration)

weight_dist_plot(varying_weights,1,plt,0.0,w_max,title="pre-pop weight distribution",filepath=results_directory)
selective_neuron_search(stim_times,target_data.segments[0].spiketrains,time_window=3.,
                        final_pattern_start =final_pattern_start_time,plt=plt,filepath=results_directory,np=np,
                        significant_spike_count=number_of_firings_per_pattern)
# for i in range(num_patterns):
#     psth_plot_8(plt,[i],stim_times,0.010,duration/1000.,title="stimulus {} PSTH".format(i),filepath=results_directory)

#mem_v = target_data.segments[0].filter(name='v')
#cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
#spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity")
#spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,title="target pop activity")
# spike_raster_plot_8(inh_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,title="target pop inh activity")
#psth_plot_8(plt,[0],target_data.segments[0].spiketrains,target_pop_size+1,duration/1000.,title="target neuron PSTH")
if results_directory is None:
    plt.show()
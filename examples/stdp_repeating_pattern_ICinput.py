import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import math
import logging
import os
import subprocess

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
target_pop_size = 64
num_pattern_neurons = 10#200#stim_size*0.1#100#
max_sync_pattern_neurons = num_pattern_neurons/2.
p_connect =0.35# 0.6#0.3#0.6#1.0#
p_response=probability_of_one_post_respond_to_stimulus(p_connect=p_connect,n_pattern_neurons=max_sync_pattern_neurons,
                                                       n_post_neurons=target_pop_size)

num_patterns =2#1#
noise_rate = 20.#10.#TODO:derive equation for the ratio of this and num_pattern_neurons (SNR) from fan in
w2s =2.
w2s_target = 5.
w_min = 0.
initial_sync_num = 20.#15.#30.#50.#
if num_pattern_neurons>0:
    av_weight = w2s_target/initial_sync_num#(num_pattern_neurons)#w2s_target/num_pattern_neurons#w_max/2.# w_max/3.#
    w_max = w2s_target/(initial_sync_num*0.4) #max_sync_pattern_neurons  # w2s_target/2#w2s/2.#2*w2s/num_pattern_neurons#w2s/10#w2s/stim_size#1.#1.#
else:
    av_weight = w2s_target/10.
    w_max = w2s_target
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))#av_weight#
tau_plus=16
tau_minus=30.
a_plus =0.01#0.005#0.001#
a_minus =0.01#0.005#0.001#
nearest_pair = False
min_delay = 1.
max_delay = 51.
stdp_delays = RandomDistribution('uniform',(min_delay,max_delay))#1.#

Fs = 22050.
# audio_data = generate_signal(signal_type='file',dBSPL=40.,fs=Fs,ramp_duration=0.01,silence=True,silence_duration=0.1,
#                              file_name='../../OME_SpiNN/1speakers_{}numbers_300repeats.wav'.format(num_patterns),plt=None)#10speakers_2numbers_5repeats.wav
pattern_duration = 1000.
audio_asc = generate_signal(signal_type="sweep_tone", freq=[30, 8000], dBSPL=50., duration=1.,
                             modulation_freq=0., fs=Fs, ramp_duration=0.0025, plt=None, silence=True,silence_duration=0.1,ascending=True)

audio_des = generate_signal(signal_type="sweep_tone", freq=[30, 8000], dBSPL=50., duration=1.,
                             modulation_freq=0., fs=Fs, ramp_duration=0.0025, plt=None, silence=True,silence_duration=0.1,ascending=False)

audio_data = []
for _ in range (100):
    audio_data += audio_asc.tolist()
    audio_data += audio_des.tolist()

onset_times = audio_stimulus_onset_detector(audio_data,Fs,num_patterns)
# stim_times = np.load('../../Brainstem/ic_spikes.npy')
# stim_times = np.load('../../OME_SpiNN/spike_times_1sp_{}num_300rep.npy'.format(num_patterns))
# stim_times = np.load('./ic_spikes_{}patterns.npy'.format(num_patterns))
stim_times = np.load('./ic_spikes_interleaved_sweep.npy')

duration=0
for train in stim_times:
    if train.size>0 and train.max()>duration:
        duration = train.max()
duration=duration.item()
duration  = 180*1000.
final_pattern_start_time = duration*0.9

stim_size = stim_times.size
num_recordings = int(numpy.ceil(duration/4000))

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=max_delay)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,32)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,16)

pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
pre_pop.record("spikes")
target_pop = sim.Population(target_pop_size,sim.IF_curr_exp,target_cell_params,label="target")
target_pop.record(["spikes"])
target_inh = sim.Population(target_pop_size,sim.IF_curr_exp,cell_params,label="inh")
target_inh.record(["spikes"])

# ac2acinh_proj = sim.Projection(target_pop,target_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
# acinh2ac_proj = sim.Projection(target_inh,target_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1,delay=1.),
#                                receptor_type='inhibitory')

ic_pop=(sim.Population(stim_size,sim.SpikeSourceArray(spike_times=stim_times)))

if num_pattern_neurons>0:
    ext_stim = sim.Population(
        stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=final_pattern_start_time-1),
        label="stim_poisson_{}Hz".format(noise_rate))

    pattern_proj=(sim.Projection(ic_pop,pre_pop,
                                        sim.OneToOneConnector(),
                                        synapse_type=sim.StaticSynapse(weight=w2s)))

    # noise_proj = sim.Projection(ext_stim,pre_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))

# Plastic Connection between pre_pop and post_pop
if nearest_pair:
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.extra_models.SpikeNearestPairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=start_weight,delay=stdp_delays)
else:
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=start_weight,delay=stdp_delays)

stdp_proj=sim.Projection(
    #pre_pop, target_pop, sim.AllToAllConnector(), receptor_type='excitatory',
    pre_pop, target_pop,sim.FixedProbabilityConnector(p_connect=p_connect),receptor_type='excitatory',
    #synapse_type=sim.StaticSynapse(weight=av_weight))
    synapse_type=stdp_model)
weights = stdp_proj.get("weight", "list", with_address=True)

#target noise for STDP stable tests
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

if type(stdp_delays) == float:
    print_delays = '{}ms_delays'.format(stdp_delays)
else:
    print_delays = 'random_dist_delays({}-{})'.format(min_delay,max_delay)
if nearest_pair:
    results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/nearest_pair_stdp' \
                        '/AN_input_{}patterns_{}fibres_{}targets_{}pattern_neurons_{}p_con_{}sp{}num{}s'.format(num_patterns,stim_size,
                                                                                                            target_pop_size,num_pattern_neurons,p_connect,
                                                                                                            1, 1, duration/1000.) + print_delays

else:
    results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                        '/pair_stdp/AN_inputinterleaved_{}patterns_{}fibres_{}targets_{}pattern_neurons_{}p_con_{}sp{}num{}s'.format(num_patterns,stim_size,
                                                                                                                      target_pop_size,num_pattern_neurons,p_connect,
                                                                                                                      1, 1, duration/1000.) + print_delays

# results_directory = None
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
print "p response: {}".format(p_response)

if target_pop_size <= 100:
    vary_weight_plot(varying_weights,range(int(target_pop_size)),[],duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),filepath=results_directory)

    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop activity",filepath=results_directory,onset_times=onset_times,pattern_duration=pattern_duration)
    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,
                        title="target pop final activity",filepath=results_directory,xlim=(0.001*(final_pattern_start_time-1),0.001*duration),
                        onset_times=onset_times,pattern_duration=pattern_duration)

    spike_raster_plot_8(stim_data.segments[0].spiketrains, plt, duration / 1000., stim_size + 1, 0.001,
                        title="pre pop activity",
                        xlim=(0.001 * (final_pattern_start_time - 1), 0.001 * duration), filepath=results_directory)

weight_dist_plot(varying_weights,1,plt,0.0,w_max,title="pre-pop weight distribution",filepath=results_directory)

# selective_neuron_search(onset_times,target_data.segments[0].spiketrains,time_window=pattern_duration+max_delay,
#                         final_pattern_start =final_pattern_start_time,plt=plt,filepath=results_directory,np=np)
if results_directory is None:
    plt.show()
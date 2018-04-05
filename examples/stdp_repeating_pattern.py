import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import math
import logging

#function that calculates minimum alphas based on w_max and tau_minus/tau_plus
def stdp_param_check(alpha_plus,alpha_minus,w_max,tau_minus,tau_plus):

    min_w_delta = w_max/2.**16
    max_tau_plus_delta = -math.log((min_w_delta/alpha_plus))*tau_plus
    max_tau_minus_delta = -math.log((min_w_delta/alpha_minus))*tau_minus

    return max_tau_plus_delta,max_tau_minus_delta,min_w_delta

# Population parameters
model = sim.IF_curr_exp
cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

#simulation parameters
#STDP -ve decay due to noise (0 num pattern neurons) currently only possible at max 250 incoming connections (stim_size)
stim_size = 1000
num_pattern_neurons = 0#stim_size*0.01#
noise_rate = 10.#1.#
num_recordings = 2#10
duration = 20.0 * 1000.
w2s =2.
w2s_stdp = 127.
w_max = w2s/stim_size#w2s/2.#w2s_stdp/10#w2s/num_pattern_neurons#1.#1.#
#start_weight = w_max/2.
av_weight = w_max/2.
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))
tau_plus=30.#16.7
tau_minus=30.#33.7
a_plus = 0.1#0.005#0.005#0.0075#0.0015#0.0025#
a_minus = 0.1#0.005#0.005#0.001#0.002#
max_tau_plus_delta,max_tau_minus_delta,min_w_delta = stdp_param_check(a_plus,a_minus,w_max,tau_minus,tau_plus)

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,128)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,128)

sos_pattern = [1.,201.,401.,601.,1101.,1601.,1801.,2001.]
#sos_pattern = [0.5*i for i in sos_pattern]
sos_duration = 3000.
stim_times = [beep+(i*sos_duration) for i in range(int(duration/sos_duration)) for beep in sos_pattern]
#pattern_freq = 10
#stim_times = [i*(1000./pattern_freq) for i in range(int(duration))]

pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
#pre_pop = sim.Population(stim_size,sim.Izhikevich,{},label="pre_pop")
pre_pop.record("spikes")

target_pop = sim.Population(1,sim.IF_curr_exp,cell_params,label="target")
#target_pop = sim.Population(1,sim.Izhikevich,{},label="target")
target_pop.record(["spikes",'v'])

pattern_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times))

if num_pattern_neurons>0:
    ext_stim = sim.Population(
        stim_size - num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        # stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),
        label="stim_poisson_{}Hz".format(noise_rate))
    noise2pattern_pop = sim.Population(
        num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),#90 duration to examine target pattern
        label="pattern_stim_poisson_{}Hz".format(noise_rate))

    chosen_int = numpy.random.choice(stim_size,num_pattern_neurons,replace=False)
    pattern_proj_list = []
    noise_proj_list = []
    noise2pattern_proj_list =[]
    noise_index = 0
    pre_index = 0
    for post in range(stim_size):
        if post not in chosen_int:
            noise_proj_list.append((pre_index,post))
            pre_index+=1

    for post in chosen_int:
        pattern_proj_list.append((0,post))
        noise2pattern_proj_list.append((noise_index,post))
        noise_index+=1
    pattern_proj = sim.Projection(pattern_pop,pre_pop,sim.FromListConnector(pattern_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))
    noise2pattern_proj = sim.Projection(noise2pattern_pop,pre_pop,sim.FromListConnector(noise2pattern_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))
    noise_proj = sim.Projection(ext_stim,pre_pop,sim.FromListConnector(noise_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))

else:
    ext_stim = sim.Population(
        stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        # stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),
        label="stim_poisson_{}Hz".format(noise_rate))

    noise_proj = sim.Projection(ext_stim,pre_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))

    chosen_int=[]


# Plastic Connection between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
    #timing_dependence=sim.extra_models.SpikeNearestPairRule(
        tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
    weight_dependence=sim.AdditiveWeightDependence(
        w_min=0.0, w_max=w_max), weight=start_weight)

stdp_proj=sim.Projection(
    pre_pop, target_pop, sim.AllToAllConnector(), receptor_type='excitatory',
 #  synapse_type=sim.StaticSynapse(weight=w2s))
    synapse_type=stdp_model)

#target noise for STDP stable tests
target_noise = sim.Population(1, sim.SpikeSourcePoisson(rate=1., duration=duration))
target_noise_proj = sim.Projection(target_noise,target_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))

varying_weights=[]
#add initial weights
#varying_weights.append([start_weight for _ in range(stim_size)])
varying_weights.append(start_weight.next(n=stim_size).tolist())

for i in range(num_recordings):
    sim.run(duration/num_recordings)
    weights = stdp_proj.get("weight","list",with_address=False)
    varying_weights.append(weights)
    stim_data_spin = pre_pop.spinnaker_get_data("spikes")

target_data =target_pop.get_data(["spikes","v"])
stim_data = pre_pop.get_data("spikes")

sim.end()
num_recordings+=1
print "max weight = {}, min weight = {}".format(max(weights),min(weights))

# print "----------Weight scaling + STDP parameters----------"
# print "max_tau_plus:{}".format(max_tau_plus_delta), "max_tau_minus:{}".format(max_tau_minus_delta),\
#     "min_w_delta:{}".format(min_w_delta)

if stim_size <= 100:
    vary_weight_plot(varying_weights,range(int(stim_size)),chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))
else:
    vary_weight_plot(varying_weights,range(100),chosen_int,duration/1000.,
    #vary_weight_plot(varying_weights,chosen_int,chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))
weight_dist_plot(varying_weights,1,plt,title="pre-pop weight distribution")

mem_v = target_data.segments[0].filter(name='v')
#cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity")
spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="target pop activity")
psth_plot_8(plt,[0],target_data.segments[0].spiketrains,1.,duration/1000.,title="target neuron PSTH")
plt.show()
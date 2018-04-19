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
target_pop_size = 10
num_pattern_neurons = 10#stim_size*0.1#100#
noise_rate = 10.#1.#
#num_recordings = 10#2#
duration = 360.0 * 1000.
num_recordings = int(numpy.ceil(duration/4000))
w2s =2.
w_max = w2s/2.#2*w2s/num_pattern_neurons#w2s/10#w2s/stim_size#1.#1.#
#start_weight = w_max/2.
av_weight = w_max/2.
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))
tau_plus=16#30.#16.7#
tau_minus=30.#33.7
a_plus = 0.001#0.0001#1./32#0.005#0.001#0.0075#0.0015#0.0025#
a_minus = 0.001#0.0001#1./32#0.005#0.001#0.001#0.002#
max_tau_plus_delta,max_tau_minus_delta,min_w_delta = stdp_param_check(a_plus,a_minus,w_max,tau_minus,tau_plus)

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,64)

#sos_pattern = [1.,201.,401.,601.,1101.,1601.,1801.,2001.]
pattern_duration = 100
pattern = [75, 36, 16,  0, 94, 24, 38, 92, 12,  2]#np.random.choice(pattern_duration,10,replace = False)#
#fifty_ms_pattern = [27,34,4,35,46,42,29,0,47,49]#np.random.choice(50,10,replace = False)

#pattern should occur at a rate of approx 6Hz
pattern_repeat_rate = 1.#0.1#6.#
num_pattern_presentations = int((duration/1000.)*pattern_repeat_rate)
pattern_repeat_period = 1000./pattern_repeat_rate
stim_times = []
for i in range(num_pattern_presentations):
    #variation in pattern start time, should not overlap
    random_variation = int((pattern_repeat_period-pattern_duration)*(np.random.rand()-0.5))
    for beep in pattern:
        stim_times.append(beep+i*pattern_repeat_period+random_variation)

max_pattern_spike = max(stim_times)

#sos_pattern = [0.5*i for i in sos_pattern]
#sos_duration = 3000.
#stim_times = [beep+(i*sos_duration)+int(20*(np.random.rand()-0.5)) for i in range(int(duration/sos_duration)) for beep in sos_pattern]
#stim_times = [beep+(i*sos_duration) for i in range(int(duration/sos_duration)) for beep in sos_pattern]


#pattern_freq = 10
#stim_times = [i*(1000./pattern_freq) for i in range(int(duration))]

pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
#pre_pop = sim.Population(stim_size,sim.Izhikevich,{},label="pre_pop")
pre_pop.record("spikes")

target_pop = sim.Population(target_pop_size,sim.IF_curr_exp,cell_params,label="target")
#target_pop = sim.Population(1,sim.Izhikevich,{},label="target")
target_pop.record(["spikes",'v'])

target_inh = sim.Population(target_pop_size,sim.IF_curr_exp,cell_params,label="inh")
target_inh.record(["spikes"])

ac2acinh_proj = sim.Projection(target_pop,target_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
acinh2ac_proj = sim.Projection(target_inh,target_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1,delay=1.),
                               receptor_type='inhibitory')

pattern_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times))


if num_pattern_neurons>0:
    ext_stim = sim.Population(
        stim_size - num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        # stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),
        label="stim_poisson_{}Hz".format(noise_rate))

    # noise2pattern_pop = sim.Population(
    #     num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),#90% duration to examine target pattern
    #       label = "pattern_stim_poisson_{}Hz".format(noise_rate))

    #increased pattern pop noise
    noise2pattern_pop = sim.Population(
        num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate*10, duration=duration * 0.9),
        label="pattern_stim_poisson_{}Hz".format(noise_rate))

    chosen_int = numpy.random.choice(stim_size,num_pattern_neurons,replace=False).tolist()
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
    #noise2pattern_proj = sim.Projection(noise2pattern_pop,pre_pop,sim.FromListConnector(noise2pattern_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))
    noise_proj = sim.Projection(ext_stim,pre_pop,sim.FromListConnector(noise_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))

else:
    ext_stim = sim.Population(
        stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
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
weights = stdp_proj.get("weight", "list", with_address=True)

#target noise for STDP stable tests
target_noise = sim.Population(target_pop_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9))
#target_noise_proj = sim.Projection(target_noise,target_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))

varying_weights=[]
#add initial weights
#varying_weights.append([start_weight for _ in range(stim_size)])
#varying_weights.append(start_weight.next(n=stim_size).tolist())
run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    # weights = stdp_proj.get("weight","list",with_address=False)
    # varying_weights.append(weights)
    if run_one:
        # make empty list for each AC neuron
        weights_list = [[] for _ in range(target_pop_size)]
        # add weights for all incoming connections to each AC neuron
        for (pre, post, weight) in weights:
            weights_list[post].append(weight)
        #varying_weights.append(weights_list)
        varying_weights.append(weights)
        run_one = False
    weights = stdp_proj.get("weight", "list", with_address=True)
    # make empty list for each AC neuron
    weights_list = [[] for _ in range(target_pop_size)]
    # add weights for all incoming connections to each AC neuron
    for (pre, post, weight) in weights:
        weights_list[post].append(weight)
    #varying_weights.append(weights_list)
    varying_weights.append(weights)

target_data =target_pop.get_data(["spikes","v"])
inh_data = target_inh.get_data(['spikes'])
stim_data = pre_pop.get_data("spikes")

sim.end()
num_recordings+=1
weight = [weight for (pre, post, weight) in weights]

print "max weight = {}, min weight = {}".format(max(weight),min(weight))
print "pattern times: {}".format(pattern)

# print "----------Weight scaling + STDP parameters----------"
# print "max_tau_plus:{}".format(max_tau_plus_delta), "max_tau_minus:{}".format(max_tau_minus_delta),\
#     "min_w_delta:{}".format(min_w_delta)

if target_pop_size <= 100:
    vary_weight_plot(varying_weights,range(int(target_pop_size)),chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))
else:
    #vary_weight_plot(varying_weights,range(100),chosen_int,duration/1000.,
    vary_weight_plot(varying_weights,chosen_int,chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))
#weight_dist_plot(varying_weights,1,plt,0.0,w_max,title="pre-pop weight distribution")

mem_v = target_data.segments[0].filter(name='v')
#cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
#spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity")
spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,title="target pop activity")
#spike_raster_plot_8(inh_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="target pop inh activity")
psth_plot_8(plt,[0],target_data.segments[0].spiketrains,target_pop_size+1,duration/1000.,title="target neuron PSTH")
plt.show()
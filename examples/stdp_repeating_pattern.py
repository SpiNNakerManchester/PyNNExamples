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
target_pop_size = 100
num_pattern_neurons = 20#100#stim_size*0.1#100#
num_patterns = 4
noise_rate = 10.#20.#1.#
#num_recordings = 10#2#
duration = 60.0 * 1000.
num_recordings = int(numpy.ceil(duration/4000))
w2s =2.#3.#
w2s_target = 5.
w_max = w2s_target/2#w2s/2.#2*w2s/num_pattern_neurons#w2s/10#w2s/stim_size#1.#1.#
w_min = 0.#w2s_target/num_pattern_neurons#
#start_weight = w_max/2.#
av_weight = w2s_target/num_pattern_neurons#w_max/2.# w_max/3.#
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))
#start_weight = w2s_target/num_pattern_neurons
tau_plus=16#30.#16.7#
tau_minus=30.#33.7
a_plus = 0.005#0.001#
a_minus = 0.005#0.001#

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson,32)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,32)

#sos_pattern = [1.,201.,401.,601.,1101.,1601.,1801.,2001.]
pattern_duration = 200#100
#pattern_a = np.random.choice(pattern_duration,10,replace = False)#[1.,50,100,150,200,250,300,350,400,450]#[75, 36, 16,  0, 94, 24, 38, 92, 12,  2]#[1.]#
#pattern_b = np.random.choice(pattern_duration,10,replace = False)
#fifty_ms_pattern = [27,34,4,35,46,42,29,0,47,49]#np.random.choice(50,10,replace = False)

pattern_repeat_rate = 0.5#1.#0.05#5.#6.#
num_pattern_presentations = int((duration/1000.)*pattern_repeat_rate)
pattern_repeat_period = 1000./pattern_repeat_rate
stim_times = []
#stim_times_b = []
for j in range(num_patterns):
    stim_times.append([])
    pattern = np.random.choice(pattern_duration, 10, replace=False)
    for i in range(num_pattern_presentations):
        #variation in pattern start time, should not overlap
        random_variation = 0#int((pattern_repeat_period-pattern_duration)*(np.random.rand()-0.5))
        for beep in pattern:
            stim_times[j].append(beep+i*pattern_repeat_period+random_variation+j*(pattern_repeat_period/num_patterns))

#stim_times = [1000,2000,3000,4000,5000]
#max_pattern_spike = max(stim_times_b)

#sos_pattern = [0.5*i for i in sos_pattern]
#sos_duration = 3000.
#stim_times = [beep+(i*sos_duration)+int(20*(np.random.rand()-0.5)) for i in range(int(duration/sos_duration)) for beep in sos_pattern]
#stim_times = [beep+(i*sos_duration) for i in range(int(duration/sos_duration)) for beep in sos_pattern]


#pattern_freq = 10
#stim_times = [i*(1000./pattern_freq) for i in range(int(duration))]

pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
#pre_pop = sim.Population(stim_size,sim.Izhikevich,{},label="pre_pop")
pre_pop.record("spikes")

target_pop = sim.Population(target_pop_size,sim.IF_curr_exp,target_cell_params,label="target")
#target_pop = sim.Population(1,sim.Izhikevich,{},label="target")
target_pop.record(["spikes"])

target_inh = sim.Population(target_pop_size,sim.IF_curr_exp,cell_params,label="inh")
target_inh.record(["spikes"])

#ac2acinh_proj = sim.Projection(target_pop,target_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
#acinh2ac_proj = sim.Projection(target_inh,target_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1,delay=1.),
#                               receptor_type='inhibitory')
pattern_pops=[]
for i in range(num_patterns):
    pattern_pops.append(sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times[i])))
#pattern_pop_a = sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times_a))
#pattern_pop_b = sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times_b))

if num_pattern_neurons>0:
    ext_stim = sim.Population(
        stim_size - num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
        # stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),
        label="stim_poisson_{}Hz".format(noise_rate))

    noise2pattern_pop = sim.Population(
        num_patterns*num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),#90% duration to examine target pattern
          label = "pattern_stim_poisson_{}Hz".format(noise_rate))
    # noise2pattern_pop = sim.Population(
    #     num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate*0.9, duration=duration*0.9),#90% duration to examine target pattern
    #       label = "pattern_stim_poisson_{}Hz".format(noise_rate))
    #increased pattern pop noise
    # noise2pattern_pop = sim.Population(
    #     num_pattern_neurons, sim.SpikeSourcePoisson(rate=noise_rate*10, duration=duration * 0.9),
    #     label="pattern_stim_poisson_{}Hz".format(noise_rate))

    #chosen_int = numpy.random.choice(stim_size,num_pattern_neurons,replace=False).tolist()
    chosen_int = numpy.random.choice(stim_size,num_patterns*num_pattern_neurons,replace=False).tolist()
    pattern_proj_list_a = []
    pattern_proj_list_b = []
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
            # if count<num_pattern_neurons:
            #     # pattern a
            #     pattern_proj_list_a.append((0,post))
            # else:
            #     #pattern b
            #     pattern_proj_list_b.append((0,post))
            pattern_proj_list[i].append((0, post))
            noise2pattern_proj_list.append((i*num_pattern_neurons+count,post))

    # pattern_a_proj = sim.Projection(pattern_pop_a,pre_pop,sim.FromListConnector(pattern_proj_list_a),synapse_type=sim.StaticSynapse(weight=w2s))
    # pattern_b_proj = sim.Projection(pattern_pop_b,pre_pop,sim.FromListConnector(pattern_proj_list_b),synapse_type=sim.StaticSynapse(weight=w2s))
    pattern_projs = []
    for i in range(num_patterns):
        pattern_projs.append(sim.Projection(pattern_pops[i],pre_pop,sim.FromListConnector(pattern_proj_list[i]),synapse_type=sim.StaticSynapse(weight=w2s)))

    noise2pattern_proj = sim.Projection(noise2pattern_pop,pre_pop,sim.FromListConnector(noise2pattern_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))
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
        w_min=w_min, w_max=w_max), weight=start_weight)

p_connect = 0.5
stdp_proj=sim.Projection(
    #pre_pop, target_pop, sim.AllToAllConnector(), receptor_type='excitatory',
    pre_pop, target_pop,sim.FixedProbabilityConnector(p_connect=p_connect),receptor_type='excitatory',
    #synapse_type=sim.StaticSynapse(weight=w2s_target))
    synapse_type=stdp_model)
weights = stdp_proj.get("weight", "list", with_address=True)

#target noise for STDP stable tests
target_noise = sim.Population(target_pop_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9))
#target_noise_proj = sim.Projection(target_noise,target_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))

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

target_data =target_pop.get_data(["spikes"])
inh_data = target_inh.get_data(['spikes'])
stim_data = pre_pop.get_data("spikes")

sim.end()
num_recordings+=1
weight = [weight for (pre, post, weight) in weights]

#np.save('./pattern_a_spikes.npy',stim_times_a)
#np.save('./pattern_b_spikes.npy',stim_times_b)
np.save('./pattern_spikes.npy',stim_times)
np.save('./target_spikes.npy',target_data.segments[0].spiketrains)

print "max weight = {}, min weight = {}".format(max(weight),min(weight))
for i,pattern in enumerate(stim_times):
    print "pattern {} times: {}".format(i,pattern)
#print "pattern b times: {}".format(pattern_b)

# print "----------Weight scaling + STDP parameters----------"
# print "max_tau_plus:{}".format(max_tau_plus_delta), "max_tau_minus:{}".format(max_tau_minus_delta),\
#     "min_w_delta:{}".format(min_w_delta)

if target_pop_size <= 100:
    vary_weight_plot(varying_weights,range(int(target_pop_size)),chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))

    spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,title="target pop activity")

else:
    #vary_weight_plot(varying_weights,range(100),chosen_int,duration/1000.,
    vary_weight_plot(varying_weights,chosen_int,chosen_int,duration/1000.,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))

#weight_dist_plot(varying_weights,1,plt,0.0,w_max,title="pre-pop weight distribution")

#mem_v = target_data.segments[0].filter(name='v')
#cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
#spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity")
#spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,target_pop_size+1,0.001,title="target pop activity")
#spike_raster_plot_8(inh_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="target pop inh activity")
#psth_plot_8(plt,[0],target_data.segments[0].spiketrains,target_pop_size+1,duration/1000.,title="target neuron PSTH")
plt.show()
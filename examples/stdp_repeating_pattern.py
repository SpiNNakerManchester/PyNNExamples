import spynnaker8 as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution


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

stim_size = 300
num_pattern_neurons = 0
pattern_freq = 10
noise_rate = 10.
num_recordings = 10
duration = 20.0 * 1000.
w2s =2.
w_max = w2s/2.
start_weight = w_max/2.
a_plus = 0.005#0.03125
a_minus = 0.005#0.85 * a_plus
#a_plus = 0.

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

stim_times = [i*(1000./pattern_freq) for i in range(int(duration))]

ext_stim = sim.Population(
    stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration*0.9),#90 duration to examine target pattern
    label="stim_poisson_{}Hz".format(noise_rate))

pre_pop = sim.Population(stim_size,sim.IF_curr_exp,cell_params,label="pre_pop")
pre_pop.record("spikes")

target_pop = sim.Population(1,sim.IF_curr_exp,cell_params,label="target")
target_pop.record(["spikes",'v'])

pattern_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=stim_times))

#fixed projection from stim to pre pop
stim_proj = sim.Projection(ext_stim,pre_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s))

#pattern neuron projection to pre pop
if num_pattern_neurons>0:
    chosen_int = numpy.random.choice(stim_size,num_pattern_neurons,replace=False)
    #for i in chosen_int:
    pattern_proj_list = []
    for post in chosen_int:
        pattern_proj_list.append((0,post))
    pattern_proj = sim.Projection(pattern_pop,pre_pop,sim.FromListConnector(pattern_proj_list),synapse_type=sim.StaticSynapse(weight=w2s))
else:
    chosen_int=[]
# Plastic Connection between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.extra_models.SpikeNearestPairRule(
        tau_plus=16.7, tau_minus=33.7, A_plus=a_plus, A_minus=a_minus),
    weight_dependence=sim.AdditiveWeightDependence(
        w_min=0.0, w_max=w_max), weight=start_weight)

stdp_proj=sim.Projection(
    pre_pop, target_pop, sim.AllToAllConnector(), receptor_type='excitatory',
 #  synapse_type=sim.StaticSynapse(weight=w2s))
    synapse_type=stdp_model)

varying_weights=[]
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    weights = stdp_proj.get("weight","list",with_address=False)
    varying_weights.append(weights)
    stim_data_spin = pre_pop.spinnaker_get_data("spikes")

target_data =target_pop.get_data(["spikes","v"])
stim_data = pre_pop.get_data("spikes")
print "max weight = {}, min weight = {}".format(max(weights),min(weights))

sim.end()

if stim_size < 1000:
    vary_weight_plot(varying_weights,range(int(stim_size)),chosen_int,duration,
                             plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.))

weight_dist_plot(varying_weights,1,plt,title="pre-pop weight distribution")

#mem_v = target_data.segments[0].filter(name='v')
#cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001,title="pre pop activity")
spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="target pop activity")
psth_plot_8(plt,[0],target_data.segments[0].spiketrains,1.,duration/1000.,title="target neuron PSTH")
plt.show()
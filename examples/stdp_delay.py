import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

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
               'tau_m': 5.,#10.0,#2.,#3.,#
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 3.0,#2.5,#
               'tau_syn_I': 2.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }
w2s =2.
wpred = 0.05#1.
w2s_target = 5.

input_pop_size =3
active_pop_size = input_pop_size

isi = 200.
num_firings = 50
predict_delay = 14#.1.#14/2.#8#1#

input_spikes =[]
for j in range(input_pop_size):
    input_spikes.append([i*isi + j*20. for i in range(1,num_firings) if i<20 or i>40])
# predict_spikes = [[i-20. for i in input_spikes if i/isi<20 or i/isi>40]]
# predict_spikes = [[i-20. for i in input_spikes]]
# input_spikes = [i*isi for i in range(1,num_firings) if i<20 or i>40]
# delayed_predict_spikes = [[i+predict_delay for i in predict_spikes[0]]]

tau_plus=10.#16.#
tau_minus=10.#30.#
a_plus =0.1
a_minus =0.1
w_min = 0
w_max = wpred

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,32)

#create populations
input_pop = sim.Population(input_pop_size,sim.SpikeSourceArray(spike_times=input_spikes))
active_pop =sim.Population(active_pop_size,sim.IF_curr_exp,cell_params,label="active_fixed_weight_scale")
# cd_pop =sim.Population(active_pop_size,sim.IF_curr_exp,target_cell_params,label="segment_fixed_weight_scale")
cd_pop =sim.Population(active_pop_size,sim.IF_curr_exp,target_cell_params,label="segment_fixed_weight_scale")
# cd_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=predict_spikes))
noise_pop = sim.Population(active_pop_size,sim.SpikeSourcePoisson(rate=10.0))
active_pop.record(["spikes","v"])
cd_pop.record(["spikes"])

#projections
input_projection = sim.Projection(input_pop,active_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s,delay=1.))
active_cd_projection = sim.Projection(active_pop,cd_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))
stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        # timing_dependence=sim.extra_models.Vogels2011Rule(alpha=0.1, tau=10.0,A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
        # weight_dependence=sim.MultiplicativeWeightDependence(
            w_min=w_min, w_max=w_max), weight=0.02,delay=predict_delay)

# cd_projection_list = [(0,1),(1,0),(0,0),(1,1)]
# cd_projection_list = [(0,1)]#[(0,1),(1,0)]
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=stdp_model)
cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.AllToAllConnector(),synapse_type=stdp_model)
# noise_projection = sim.Projection(noise_pop,cd_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target))

weights = cd_active_projection.get("weight", "list", with_address=True)

duration = num_firings * isi
num_recordings = 10

varying_weights=[]
run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if run_one:
        varying_weights.append(weights)
        run_one = False
    weights = cd_active_projection.get("weight", "list", with_address=True)
    varying_weights.append(weights)
active_data =active_pop.get_data(["spikes","v"])
cd_data = cd_pop.get_data(["spikes"])

sim.end()
num_recordings+=1
target_neurons = range(int(active_pop_size))#[1]#
vary_weight_plot(varying_weights,target_neurons,None,duration/1000.,
                 plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),
                 title='Negative Window Predictive Neuron to Active Neuron Weight (connection delay = {})'.format(predict_delay),
                 legend=True,figsize=(16,12),filepath=None)#"/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/HTM/stdp_bug/stdp_delay")

active_spikes = active_data.segments[0].spiketrains
predict_spikes = cd_data.segments[0].spiketrains
delayed_predict_spikes=[]
for spikes in predict_spikes:
    delayed_predict_spikes.append([i.item()+predict_delay for i in spikes])

test_spikes =[]
legend_string=[]

for i in range(input_pop_size):
    test_spikes.append(active_spikes[i])
    test_spikes.append(predict_spikes[i])
    test_spikes.append(delayed_predict_spikes[i])
    legend_string.append('active_{}'.format(i))
    legend_string.append('segment_{}'.format(i))
    legend_string.append('segment_delayed{}'.format(i))

# test_spikes = [active_spikes[0],predict_spikes[0],delayed_predict_spikes[0],active_spikes[1],predict_spikes[1],delayed_predict_spikes[1]]
# test_spikes = [active_spikes[0],predict_spikes[0],delayed_predict_spikes[0],active_spikes[1]]
colours = ['b', 'g', 'r','c', 'm', 'y']
neuron_legend = []
plt.figure('spike times (connection delay = {})'.format(predict_delay),figsize=(16,12))
for i,spikes in enumerate(test_spikes):
    for xc in spikes:
        plt.axvline(x=xc,color=colours[i%len(colours)])
    neuron_legend.append(plt.Line2D([0], [0], color=colours[i%len(colours)], lw=4))

plt.legend(neuron_legend,legend_string,
           bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                        ncol=3, mode="expand", borderaxespad=0.)
plt.xlabel("time(ms)")
# plt.xlim(200,300)
# plt.savefig("/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/HTM/stdp_bug/stdp_delay/total_timings_{}delay".format(predict_delay))

# spike_raster_plot_8(active_spikes,plt,duration/1000.,3.,0.001,title="active pop activity")
# spike_raster_plot_8(predict_spikes,plt,duration/1000.,3.,0.001,title="cd pop activity")

print "final weights:"
for final_weights in varying_weights[-1]:
    print final_weights

plt.show()
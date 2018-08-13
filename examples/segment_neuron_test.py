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
               'tau_m': 10.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
                #'tau_syn_I': 10.0,#2.5,#
                'v_reset': -70.0,#-100.0, # use a large v_reset to produce 'boosting'
                'v_rest': -65.0,
                'v_thresh': -55.6
               }

w2s_target = 4.0#0.12#0.05#2.5#5.
n_connections = 16
initial_weight = w2s_target/n_connections
connection_weight = initial_weight*2.#/2.
number_of_inputs = 10
inh_weight = initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

av_weight = initial_weight
w_max_cd = av_weight*2#1.1#w2s_target/2.#
w_min_cd = 0.0#av_weight*0.5#0
a_plus_cd = 0.1#1.#
a_minus_cd = 0.1#1.#
tau_plus_cd = 16.
tau_minus_cd =30.#1.#
ten_perc = av_weight/10.
start_weight = av_weight#RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

stdp_model_cd = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus_cd, tau_minus=tau_minus_cd, A_plus=a_plus_cd, A_minus=a_minus_cd),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min_cd, w_max=w_max_cd), weight=start_weight,delay=1.)


input_spikes =[]
inh_spikes = []
isi = 100.
n_repeats = 20

for neuron in range(number_of_inputs):
    input_spikes.append([i*isi for i in range(n_repeats) if i<5 or i>10])
    inh_spikes.append([(i*isi)-1 for i in range(n_repeats) if i<5 or i>10])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes))
inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_cond_exp,inh_cond_params,label="fixed_weight_scale_cond")
cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")
# inh_pop =

cd_pop.record(["spikes","v"])

#================================================================================================
# Projections
#================================================================================================
input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
#inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=stdp_model_cd,receptor_type='inhibitory')

duration = max(input_spikes[0])

sim.run(duration)

cd_data = cd_pop.get_data(["spikes","v"])

sim.end()

mem_v = cd_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.001,title='cd pop')


spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="cd pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")

plt.show()
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
winh = 0.5
wpred = 0.05#1.
w2s_target = 5.

input_pop_size =1
active_pop_size = 32
# assume 1% of 2048 columns are active per 1ms timestep
#if each column fired at 1Hz then there would be approx. 2 active columns per timestep
#we assume each column fires at around 10Hz, producing approx. 20 active columns per ms
column_firing_rate = 10.
isi = 1000./column_firing_rate
num_firings = 50
predict_delay = 8
input_spikes = [i*isi for i in range(1,num_firings)]#[10.,30,50]
predict_spikes = [i*isi-predict_delay for i in range(1,num_firings)]#[5.]

tau_plus=16.
tau_minus=30.
a_plus =0.5#0.001#
a_minus =0.5#0.001#
w_min = 0
w_max = wpred

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

#create populations
input_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=input_spikes))
active_pop =sim.Population(active_pop_size,sim.IF_curr_exp,cell_params,label="fixed_weight_scale")
cd_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=predict_spikes))

active_pop.record(["spikes","v"])

#projections
input_projection = sim.Projection(input_pop,active_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=w2s,delay=1.))
inh_connection_list = []
for post in range(active_pop_size):
    for pre in range(active_pop_size):
        if pre!=post:
            inh_connection_list.append((pre,post))
active_inh_active_projection = sim.Projection(active_pop,active_pop,sim.FromListConnector(inh_connection_list),synapse_type=sim.StaticSynapse(weight=winh),receptor_type='inhibitory')

stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max), weight=0.,delay=predict_delay)

cd_projection_list = [(0,0)]
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=sim.StaticSynapse(weight=wpred,delay=1.))
cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=stdp_model)
weights = cd_active_projection.get("weight", "list", with_address=True)

# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=wpred,delay=1.))
duration = num_firings * isi
num_recordings =10

varying_weights=[]
run_one=True
for i in range(num_recordings):
    sim.run(duration/num_recordings)
    if run_one:
        varying_weights.append(weights)
        run_one = False
    weights = cd_active_projection.get("weight", "list", with_address=True)
    varying_weights.append(weights)
#sim.run(duration)
active_data =active_pop.get_data(["spikes","v"])

sim.end()
num_recordings+=1

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                    '/HTM/{}active_neurons_{}Hz_{}prediction_delay_{}Taup_{}taumin_{}alpha_spike_pair'\
                    .format(active_pop_size,column_firing_rate,predict_delay,tau_plus,tau_minus,a_plus)

results_directory = None
if results_directory is not None:
    if not os.path.isdir(results_directory):
        bashCommand = ["mkdir",results_directory]
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
        output, error = process.communicate()

vary_weight_plot(varying_weights,range(int(1)),None,duration/1000.,
                 plt,np=numpy,num_recs=num_recordings,ylim=w_max+(w_max/10.),
                 title='Predictive Neuron to Active Neuron Weight',
                 filepath=results_directory)

spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity",filepath=results_directory)
mem_v = active_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=0,title='Predicted Active Neuron',filepath=results_directory)
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=1,title='Inhibited Active Neuron',filepath=results_directory)
# cell_voltage_plot_8(mem_v, plt, 100., 1.,scale_factor=0.001)
if results_directory is None:
    plt.show()
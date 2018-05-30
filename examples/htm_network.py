import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution

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
winh = 1.0
wpred = 1.
w2s_target = 5.

input_pop_size =1
active_pop_size = 10
isi = 20.
num_firings = 20
predict_delay = 5
input_spikes = [i*isi for i in range(1,num_firings)]#[10.,30,50]
predict_spikes = [i*isi-predict_delay for i in range(10,num_firings)]#[5.]
#input_pop_spike_source_array = [input_spikes for _ in range(input_pop_size)]

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

#create populations
input_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=input_spikes))
active_pop =sim.Population(active_pop_size,sim.IF_curr_exp,cell_params,label="active_pop")
active_inh_pop = sim.Population(active_pop_size,sim.IF_curr_exp,target_cell_params,label="active_inh_pop")
cd_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=predict_spikes))

active_pop.record(["spikes","v"])
active_inh_pop.record(["spikes"])

#projections
input_projection = sim.Projection(input_pop,active_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=w2s,delay=1.))
active_active_inh_projection = sim.Projection(active_pop,active_inh_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=w2s_target,delay=1.))
inh_connection_list = []
for post in range(active_pop_size):
    for pre in range(active_pop_size):
        if pre!=post:
            inh_connection_list.append((pre,post))
# active_inh_active_projection = sim.Projection(active_inh_pop,active_pop,sim.FromListConnector(inh_connection_list),synapse_type=sim.StaticSynapse(weight=winh),receptor_type='inhibitory')
active_inh_active_projection = sim.Projection(active_pop,active_pop,sim.FromListConnector(inh_connection_list),synapse_type=sim.StaticSynapse(weight=winh),receptor_type='inhibitory')

cd_projection_list = [(0,0),(0,1)]
cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.FromListConnector(cd_projection_list),synapse_type=sim.StaticSynapse(weight=wpred,delay=1.))
# cd_active_projection =  sim.Projection(cd_pop,active_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=wpred,delay=1.))
duration = num_firings * isi
sim.run(duration)

active_data =active_pop.get_data(["spikes","v"])
active_inh_data = active_inh_pop.get_data(['spikes'])

sim.end()

spike_raster_plot_8(active_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop activity")
# spike_raster_plot_8(active_inh_data.segments[0].spiketrains,plt,duration/1000.,active_pop_size+1,0.001,title="active pop inh activity")
mem_v = active_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=0)
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001,id=1)
# cell_voltage_plot_8(mem_v, plt, 100., 1.,scale_factor=0.001)
plt.show()
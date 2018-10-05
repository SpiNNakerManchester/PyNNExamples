import matplotlib.pylab as plt

from pacman.model.constraints.placer_constraints.chip_and_core_constraint import ChipAndCoreConstraint

import spynnaker8 as sim

from signal_prep import *

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
target_cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,#2.,#3.,#
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 1.0,#2.5,
               'tau_syn_I': 1.0,#2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }
w2s = 2.0
w2s_post = 5.
# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
duration = 20*1000
pop_size =10

pre_pop = sim.Population(pop_size,sim.IF_curr_exp,cell_params,label="pre")
post_pop = sim.Population(pop_size,sim.IF_curr_exp,target_cell_params,label="fixed_weight_scale")

source_spikes = sim.Population(2,sim.SpikeSourcePoisson(rate=10.,duration=duration))

source_list = [(0,4),(0,5)]
source_pre_proj = sim.Projection(source_spikes,pre_pop,sim.FromListConnector(source_list),
                                 synapse_type=sim.StaticSynapse(weight=w2s,delay=1.))



pre_pop.record(["spikes"])
post_pop.record(["spikes"])

weights = plastic_projection.get("weight", "list", with_address=True)

varying_weights=[]
sim.run(duration)
varying_weights.append(weights)
weights = plastic_projection.get("weight", "list", with_address=True)
varying_weights.append(weights)

pre_data  = pre_pop.get_data(["spikes"])
post_data = post_pop.get_data(["spikes"])

sim.end()

spike_raster_plot_8(pre_data.segments[0].spiketrains,plt,duration/1000.,pop_size+1,0.001,title="pre-synaptic")
spike_raster_plot_8(post_data.segments[0].spiketrains,plt,duration/1000.,pop_size+1,0.001,title="post-synaptic")

plt.show()



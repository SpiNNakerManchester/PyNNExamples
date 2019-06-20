import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from signal_prep import *

# sim_data = np.load('sim_data.npz',)

# plt.show()

active_params = {'e_rev_E': -55,
                 'tau_m': 10.,
                 'tau_syn_E': 20,
                 # 'tau_syn_E2': 5.,
                 'v_thresh': -50,
                 #'v_reset': -65.
                 'tau_syn_I': 1. #TODO: maybe use the dual inh?
}
p.setup(1.)
p.set_number_of_neurons_per_core(p.IF_cond_exp,32)
runtime = 500#125
num_repeats = 1.
column_size = 16

active_spikes = []
for i in range(2):
    active_spikes.append([i for i in range(20*i,runtime,100)])

# prediction_spikes = [i-20. for i in active_spikes if i > 0.]
prediction_spikes = active_spikes[0]
# Spike source to send spike via plastic synapse
exc_src = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': prediction_spikes}, label="src1")
exc2_src = p.Population(2, p.SpikeSourceArray,
                        {'spike_times': active_spikes}, label="src2")
# inh_src = p.Population(1, p.SpikeSourceArray,
#                           {'spike_times': [30]}, label="src3")
# inh2_src = p.Population(1, p.SpikeSourceArray,
#                           {'spike_times': [160]}, label="src4")

# Post-synapse population
pop_exc = p.Population(2*column_size, p.IF_cond_exp,active_params, label="cond_fixed_weight_scale_2e2i")
# pop_exc = p.Population(column_size, p.IF_cond_exp,{},  label="test")
# pop_exc = p.Population(column_size, p.IF_curr_exp,{},  label="test")

# pop_exc.set(e_rev_E=-55)
# pop_exc.set(tau_syn_E=20)
# pop_exc.set(tau_syn_E2=5)

# pop_exc.set(e_rev_E2=-10)
# pop_exc.set(e_rev_E2=-80)
# pop_exc.set(e_rev_E2=-50)

stdp_model = p.STDPMechanism(
        timing_dependence=p.SpikePairRule(
            tau_plus=16., tau_minus=30., A_plus=0.5, A_minus=0.5),
        weight_dependence=p.AdditiveWeightDependence(
            w_min=0., w_max=0.1), weight=0.0,delay=1.)
            # w_min=0., w_max=2.1), weight=2.,delay=1.)

structure_model_with_stdp_pred = p.StructuralMechanismSTDP(
    stdp_model=stdp_model,
    weight=0.,  # Use this weights when creating a new synapse
    max_weight=0.01,#0.001,#0.001,#*0.9,#w_max,#TODO:try decreasing this
    s_max=5,  # Maximum allowed fan-in per target-layer neuron
    #grid=[np.sqrt(active_pop_size), np.sqrt(active_pop_size)],  # 2d spatial org of neurons
    random_partner=True,#False,  # Choose a partner neuron for formation at random,
    # as opposed to selecting one of the last neurons to have spiked
    f_rew=10 ** 2,  #10 ** 4,  # Hz
    p_elim_dep=0.,#1.,#0.5,#
    p_elim_pot=0.,
)
pred_list = [(0,16)]
# synapse_exc = p.Projection(
#     # exc_src, pop_exc, p.OneToOneConnector(),
#     # pop_exc, pop_exc, p.FromListConnector(pred_list),
#     exc_src, pop_exc, p.FromListConnector(pred_list),
#     # p.StaticSynapse(weight=0.1, delay=1), receptor_type="excitatory")
#     synapse_type=stdp_model, receptor_type="excitatory")

synapse_exc = p.Projection(
    pop_exc, pop_exc, p.FixedProbabilityConnector(p_connect=0.0),
    synapse_type=structure_model_with_stdp_pred, receptor_type="excitatory")

stim_list = []
for i in range(column_size):
    stim_list.append((0,i))
    stim_list.append((1,i+column_size))

# synapse_exc2 = p.Projection(
#     exc2_src, pop_exc, p.FromListConnector(stim_list),#p.AllToAllConnector(),
#     p.StaticSynapse(weight=0.1, delay=1), receptor_type="excitatory2")

inh_connection_list = []
winh = 1.#0.2
for post in range(column_size):
    for pre in range(column_size):
        if pre!=post:
            inh_connection_list.append((pre,post))

for post in range(column_size,column_size*2):
    for pre in range(column_size,column_size*2):
        if pre!=post:
            inh_connection_list.append((pre,post))

active_inh_active_projection = p.Projection(pop_exc,pop_exc,p.FromListConnector(inh_connection_list),
                                              synapse_type=p.StaticSynapse(weight=winh),receptor_type='inhibitory')

# synapse_inh = p.Projection(
#     inh_src, pop_exc, p.OneToOneConnector(),
#     p.StaticSynapse(weight=0.33, delay=1), receptor_type="inhibitory")
# synapse_inh2 = p.Projection(
#     inh2_src, pop_exc, p.AllToAllConnector(),
#     p.StaticSynapse(weight=1.32, delay=1), receptor_type="inhibitory2")

pop_exc.record("all")
weights = []
for _ in range(int(num_repeats)):
    p.run(runtime/num_repeats)
    # runtime = runtime/0.1 # temporary scaling to account for new recording
    weights.append(synapse_exc.get('weight', 'list',
                                       with_address=True))#[0])

exc_data = pop_exc.get_data()

print "Post-synaptic neuron firing frequency: {} Hz".format(
    len(exc_data.segments[0].spiketrains[0]))

print "weights",weights
p.end()
# Plot
Figure(
    # plot data for postsynaptic neuron
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",legend=False,
          yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",legend=False,
           yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",legend=False,
           yticks=True,xticks=True, xlim=(0, runtime)),
    Panel(exc_data.segments[0].spiketrains,marker='.',
          yticks=True,markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0,xticks=True, xlim=(0, runtime)),
    annotations="Post-synaptic neuron firing frequency: {} Hz".format(
        len(exc_data.segments[0].spiketrains[0]))
)

recorded_active_spikes = [spike_time.item() for spike_time in exc_data.segments[0].spiketrains[0]]

# spike_raster_plot_8([recorded_active_spikes,prediction_spikes],plt,runtime/1000.,ylim=3.)
# plt.figure()
# plt.subplot(2,1,1)
# plt.eventplot([recorded_active_spikes,prediction_spikes],colors=['b','r'])

# plt.subplot(2,1,2)
# x = np.linspace(0,runtime,num_repeats)
# plt.plot(x,weights)

np.savez_compressed('./sim_data.npz',active_spikes=exc_data.segments[0].spiketrains,mem_v=exc_data.segments[0].filter(name='v'),
                    gsyn_exc=exc_data.segments[0].filter(name='gsyn_exc'),g_syn_inh=exc_data.segments[0].filter(name='gsyn_inh'))

plt.show()
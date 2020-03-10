import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from frozen_poisson import build_input_spike_train
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


cycle_time = 1024
num_repeats = 100
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0.8,
    "v_rest": 0,
    "w_fb": 0.75
    }

target_data = []
for i in range(1024):
            target_data.append(
                0 + 2 * np.sin(2 * i * 2* np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2* np.pi / 1024))
                )

readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": target_data,
    "eta": 0.0001
    }

pynn.setup(timestep=1)

input_size = 100
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourceArray,
                            {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            label='input_pop')
# input_pop = pynn.Population(input_size,
#                             pynn.SpikeSourceArray,
#                             {'spike_times': [np.linspace(0, 1000, 10) for i in range(input_size)]},
#                             label='input_pop')

# neuron = pynn.Population(1,
#                          pynn.extra_models.EPropAdaptive(**neuron_params),
#                          label='eprop_pop')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.SinusoidReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

base_weight = np.random.randn() / np.sqrt(input_size)#0
# start_w = [(-0.5) / float(input_size*3) for i in range(input_size*3)]
start_w = [0 for i in range(input_size*3)]
# start_w = [base_weight for i in range(input_size)]
eprop_learning = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2, w_max=2, reg_rate=0.0))#,
    # weight=start_w, delay=[i for i in range(input_size)])#[0, 0])

start_w_ln = -0.5#[-0.5 for i in range(input_size)]
eprop_learning_1n = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=0.0),
    weight=start_w_ln, delay=0)#[i for i in range(input_size)])#[0, 0])

in_proj = pynn.Projection(input_pop,
                          # neuron,
                          readout_pop,
                          pynn.FromListConnector([[i, 0, base_weight, i] for i in range(input_size)]),
                          synapse_type=eprop_learning,
                          label='input_connections',
                          receptor_type='input_connections')


# out_proj = pynn.Projection(neuron,
#                            readout_pop,
#                            pynn.OneToOneConnector(),
#                            synapse_type=eprop_learning,
#                            label='input_connections',
#                            receptor_type='input_connections')
#
# learning_proj = pynn.Projection(readout_pop,
#                                 neuron,
#                                 pynn.OneToOneConnector(),
#                                 pynn.StaticSynapse(weight=[0.5], delay=[0]),
#                                 receptor_type='learning_signal')

# self_learning_proj = pynn.Projection(readout_pop,
#                                      readout_pop,
#                                      pynn.OneToOneConnector(),
#                                      pynn.StaticSynapse(weight=[0.5], delay=[0]),
#                                      receptor_type='learning_signal')

input_pop.record('spikes')
# neuron.record('all')
readout_pop.record('all')

runtime = cycle_time * num_repeats
pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
# neuron_res = neuron.get_data('all')
readout_res = readout_pop.get_data('all')



# Plot rec neuron output
# plt.figure()
# # plt.tight_layout()
#
# plt.subplot(4, 1, 1)
# plt.plot(neuron_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')
#
# plt.subplot(4, 1, 2)
# plt.plot(neuron_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')
#
# plt.subplot(4, 1, 3)
# plt.plot(neuron_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')
#
# plt.subplot(4,1,4)
# plt.plot(in_spikes.segments[0].spiketrains, label='in_spikes')

plt.figure()
Figure(
    # Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),
    #
    # Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),
    #
    # Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    # Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

#     title="eprop neuron"
# )
# plt.show()
#
# plt.figure()
# Figure(
    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    title="readout neuron"
)
# plt.show()

# Plot Readout output
# plt.figure()
# # plt.tight_layout()
#
# plt.subplot(3, 1, 1)
# plt.plot(readout_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')
#
# plt.subplot(3, 1, 2)
# plt.plot(readout_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')
#
# plt.subplot(3, 1, 3)
# plt.plot(readout_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')


plt.show()

pynn.end()
print("job done")
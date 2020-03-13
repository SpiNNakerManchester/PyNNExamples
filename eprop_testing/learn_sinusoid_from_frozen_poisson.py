import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from frozen_poisson import build_input_spike_train
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size)
    # base_weight = 0
    return base_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0):
    connections = []
    for j in range(post_pop_size):
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
                conn = [i, j, weight_distribution(pre_pop_size), delay_count]
                delay_count += 1
                connections.append(conn)
    return connections

np.random.seed(272727)

cycle_time = 1024
num_repeats = 100
pynn.setup(1.0)

target_data = []
for i in range(1024):
            target_data.append(#1)
                0 + 2 * np.sin(2 * i * 2* np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2* np.pi / 1024))
                )

readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": target_data,
    "eta": 0.5
    }

pynn.setup(timestep=1, max_delay=250)
pynn.set_number_of_neurons_per_core(8)

input_size = 10
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourceArray,
                            {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            label='input_pop')
# input_pop = pynn.Population(input_size,
#                             pynn.SpikeSourceArray,
#                             {'spike_times': [np.linspace(0, 1000, 10) for i in range(input_size)]},
#                             label='input_pop')

neuron_pop_size = 10
neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0,
    "w_fb": [np.random.random() for i in range(neuron_pop_size)],
    # "B": 0.0,
    "beta": 0.0,
    "eta": 0.5
    }
neuron = pynn.Population(neuron_pop_size,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.SinusoidReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

reg_rate = 0.001
start_w = [weight_distribution(neuron_pop_size*input_size) for i in range(input_size)]
eprop_learning_neuron = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=reg_rate))#,
    # weight=start_w, delay=[i for i in range(neuron_pop_size*input_size)])

# from_list_in = [[i, j, weight_distribution(input_size), i+(j*input_size)] for i in range(input_size) for j in range(neuron_pop_size)]
from_list_in = probability_connector(input_size, neuron_pop_size, 1.)
in_proj = pynn.Projection(input_pop,
                          neuron,
                          # readout_pop,
                          pynn.FromListConnector(from_list_in),
                          # pynn.AllToAllConnector(),
                          synapse_type=eprop_learning_neuron,
                          label='input_connections',
                          receptor_type='input_connections')


# start_w = [(-0.5) / float(input_size*3) for i in range(input_size*3)]
# start_w = [0 for i in range(input_size*3)]
# start_w = [base_weight for i in range(input_size)]
eprop_learning_output = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2, w_max=2, reg_rate=0.0))#,
    # weight=start_w, delay=[i for i in range(input_size)])#[0, 0])

from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
out_proj = pynn.Projection(neuron,
                           readout_pop,
                           # pynn.OneToOneConnector(),
                           pynn.FromListConnector(from_list_out),
                           synapse_type=eprop_learning_output,
                           label='input_connections',
                           receptor_type='input_connections')

learning_proj = pynn.Projection(readout_pop,
                                neuron,
                                pynn.OneToOneConnector(),
                                pynn.StaticSynapse(weight=[0.5], delay=[0]),
                                receptor_type='learning_signal')


# start_w = [(-0.5) / float(input_size*3) for i in range(input_size*3)]
# start_w = [0 for i in range(input_size*3)]
# start_w = [base_weight for i in range(input_size)]
eprop_learning_recurrent = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2, w_max=2, reg_rate=0.0))#,
    # weight=start_w, delay=[i for i in range(input_size)])#[0, 0])

# from_list_rec = probability_connector(neuron_pop_size, neuron_pop_size, 0.4, offset=100)
# recurrent_proj = pynn.Projection(neuron,
#                                  neuron,
#                                  pynn.FromListConnector(from_list_rec),
#                                  synapse_type=eprop_learning_recurrent,
#                                  label='recurrent_connections',
#                                  receptor_type='recurrent_connections')

input_pop.record('spikes')
neuron.record('all')
readout_pop.record('all')

experiment_label = "\neta:{}/{} - in size:{} - reg_rate: {}\n".format(readout_neuron_params["eta"], neuron_params["eta"], input_size, reg_rate)
print experiment_label

runtime = cycle_time * num_repeats
pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
neuron_res = neuron.get_data('all')
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

total_error = 0.0
cycle_error = [0.0 for i in range(num_repeats)]
for cycle in range(num_repeats):
    for time_index in range(1024):
        instantaneous_error = np.abs(float(
            readout_res.segments[0].filter(name='v')[0][time_index+(cycle*1024)][0]) - target_data[time_index])
        cycle_error[cycle] += instantaneous_error
        total_error += instantaneous_error
new_connections_in = in_proj.get('weight', 'delay').connections[0]#[]
# for conn in in_proj.get('weight', 'delay').connections[0]:
#     if conn[2] > 3:
#         new_connections_in.append([conn[0], conn[1], conn[2]-16, conn[3]])
#     else:
#         new_connections_in.append([conn[0], conn[1], conn[2], conn[3]])
connection_diff_in = []
for i in range(len(from_list_in)):
    connection_diff_in.append(new_connections_in[i][2] - from_list_in[i][2])
print "Input connections\noriginal\n", np.array(from_list_in)
print "new\n", np.array(new_connections_in)
print "diff\n", np.array(connection_diff_in)
new_connections_out = out_proj.get('weight', 'delay').connections[0]#[]
# for conn in out_proj.get('weight', 'delay').connections[0]:
#     if conn[2] > 3:
#         new_connections_out.append([conn[0], conn[1], conn[2]-16, conn[3]])
#     else:
#         new_connections_out.append([conn[0], conn[1], conn[2], conn[3]])
connection_diff_out = []
for i in range(len(from_list_out)):
    connection_diff_out.append(new_connections_out[i][2] - from_list_out[i][2])
print "Output connections\noriginal\n", np.array(from_list_out)
print "new\n", np.array(new_connections_out)
print "diff\n", np.array(connection_diff_out)
print experiment_label
print "cycle_error =", cycle_error
print "total error =", total_error

plt.figure()
Figure(
    Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

#     title="eprop neuron"
# )
# plt.show()
#
# plt.figure()
# Figure(
    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    # Panel(cycle_error, ylabel='cycle error', yticks=True, xticks=True, xlim=(0, num_repeats)),

    title="neuron data for {}".format(experiment_label)
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

'''
plt.figure()
plt.scatter([i for i in range(num_repeats)], cycle_error)
plt.show()
'''
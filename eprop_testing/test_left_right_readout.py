# Copyright (c) 2019 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pyNN.spiNNaker as pynn
import numpy as np
import matplotlib.pyplot as plt
# from frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size) #+ 0.5
    if abs(base_weight) < np.exp(-10):  # checking because if too many are small neurons can't learn
        print("\nweight too small: {}\n".format(base_weight))
    # base_weight = 0
    return base_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0, added_weight=0.):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
                neuron_syn_count += 1
                conn = [i, j, weight_distribution(pre_pop_size)+added_weight, delay_count]
                delay_count += 1
                connections.append(conn)
        if neuron_syn_count > max_syn_per_neuron:
            max_syn_per_neuron = neuron_syn_count
    return connections, max_syn_per_neuron

np.random.seed(272727)

number_of_cues = 1
cycle_time = (number_of_cues*150)+1000+150
num_repeats = 800
pynn.setup(1.0)

target_data = []
for i in range(1024):
            target_data.append(#1)
                0 + 2 * np.sin(2 * i * 2* np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2* np.pi / 1024))
                )


reg_rate = 0.000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = True
synapse_eta = 0.0005
tau_a = 2500#[cycle_time - 150 + (np.random.randn() * 200) for i in range(100)]
input_split = 20
window_cycles = 10
window_size = cycle_time*window_cycles
eprop_beta = 3

in_weight = 0.75
prompt_weight = 0.75
rec_weight = 0#-0.5
out_weight = 0#0.01
weight_string = "i{}-p{}-r{}-o{}".format(in_weight, prompt_weight, rec_weight, out_weight)


pynn.setup(timestep=1)

pynn.set_number_of_neurons_per_core(pynn.extra_models.EPropAdaptive, 6)

input_size = 40
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "poisson_pop_size": input_size / 4,
    "rate_on": 100,
    # "tau_m": tau_a,
    "w_fb": [1, -1, 0],
    "eta": synapse_eta * 5.,
    "window_size": window_size,
    }
rates = []
for i in range(input_size):
    if i >= (3*input_size) / 4:
        rates.append(10)
    else:
        rates.append(0)
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourcePoisson(rate=rates),
                            # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            # {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, input_split, input_split, input_size)},
                            label='input_pop')

neuron_pop_size = 100
ratio_of_LIF = 0.5
beta = []
for i in range(neuron_pop_size):
    if i < neuron_pop_size/2:
    # if i % 2 == 0:
        beta.append(0) #this should be 0, just testing all ALIF
    else:
        beta.append(eprop_beta)
neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0,
    # "w_fb": [np.random.random() for i in range(neuron_pop_size)], # best it seems
    # "w_fb": [(np.random.random() * 2) - 1. for i in range(neuron_pop_size)],
    "w_fb": [4*np.random.random() - 4*np.random.random() for i in range(neuron_pop_size)],  ## for both feedback weights
    # "w_fb": [3]*(neuron_pop_size/4) + [-3]*(neuron_pop_size/4) + [3]*(neuron_pop_size/4) + [-3]*(neuron_pop_size/4),
    # "B": 0.0,
    "beta": beta,
    "target_rate": 10,
    "tau_a": tau_a,
    "eta": synapse_eta * 5.,
    "window_size": window_size,
    }
neuron = pynn.Population(neuron_pop_size,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.LeftRightReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

poisson_control_edge = SpynnakerExternalDevicePluginManager.add_edge(
    readout_pop._vertex, input_pop._vertex, "CONTROL")
# pynn.external_devices.activate_live_output_to(
#     readout_pop, input_pop, "CONTROL")
input_pop._vertex.set_live_poisson_control_edge(poisson_control_edge)
# pynn.external_devices.add_poisson_live_rate_control(input_pop)

# start_w = [weight_distribution(neuron_pop_size*input_size) for i in range(input_size)]
eprop_learning_neuron = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=reg_rate))

from_list_in, max_syn_per_input = probability_connector(input_size, neuron_pop_size, p_connect_in, added_weight=in_weight)
if max_syn_per_input > 100:
    Exception
else:
    print("max number of synapses per neuron:", max_syn_per_input)
in_proj = pynn.Projection(input_pop,
                          neuron,
                          pynn.FromListConnector(from_list_in),
                          # pynn.AllToAllConnector(),
                          synapse_type=eprop_learning_neuron,
                          label='input_connections',
                          receptor_type='input_connections')

eprop_learning_output = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=0.0))

# from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
from_list_out, max_syn_per_output = probability_connector(neuron_pop_size, 2, p_connect_out, added_weight=out_weight)
if max_syn_per_output > 100:
    Exception
else:
    print("max number of synapses per readout:", max_syn_per_output)
out_proj = pynn.Projection(neuron,
                           readout_pop,
                           # pynn.OneToOneConnector(),
                           pynn.FromListConnector(from_list_out),
                           synapse_type=eprop_learning_output,
                           label='input_connections',
                           receptor_type='input_connections')

learning_proj = pynn.Projection(readout_pop,
                                neuron,
                                pynn.AllToAllConnector(),
                                pynn.StaticSynapse(weight=0.5, delay=0),
                                receptor_type='learning_signal')

# learning_proj = pynn.Projection(readout_pop,
#                                 readout_pop,
#                                 pynn.AllToAllConnector(),
#                                 pynn.StaticSynapse(weight=0.5, delay=0),
#                                 receptor_type='learning_signal')

if recurrent_connections:
    eprop_learning_recurrent = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-2.0, w_max=2.0, reg_rate=reg_rate))

    from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec, offset=0, added_weight=rec_weight)
    if max_syn_per_rec > 150:
        Exception
    else:
        print("max number of synapses per neuron:", max_syn_per_rec)
    recurrent_proj = pynn.Projection(neuron,
                                     neuron,
                                     pynn.FromListConnector(from_list_rec),
                                     synapse_type=eprop_learning_recurrent,
                                     label='recurrent_connections',
                                     receptor_type='recurrent_connections')

input_pop.record('spikes')
neuron.record('spikes')
# neuron.record(['gsyn_exc', 'v', 'gsyn_inh'], indexes=[i for i in range(45, 55)])
neuron[[i for i in range(45, 55)]].record(['gsyn_exc', 'v', 'gsyn_inh'])
readout_pop.record('all')

runtime = cycle_time * num_repeats

experiment_label = "eta:{}/{} - size:{}/{} - reg_rate:{} - p_conn:{}/{}/{} - rec:{} - cycle:{}/{}/{} noresetv 0d b:{}".format(
    readout_neuron_params["eta"], neuron_params["eta"], input_size, neuron_pop_size,
    reg_rate, p_connect_in, p_connect_rec, p_connect_out, recurrent_connections,
    cycle_time, window_size, runtime, eprop_beta)
print("\n", experiment_label, "\n")

pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
neuron_res = neuron.get_data('all')
readout_res = readout_pop.get_data('all')

total_error = 0.0
cycle_error = [0.0 for i in range(num_repeats)]
correct_or_not = [0 for i in range(num_repeats)]
soft_max = [[], []]
cross_entropy = [] # how do I extract the answer easily? final gsyn exc value?
all_cross = [[], []]
from_soft = [[], []]
for cycle in range(num_repeats):
    ticks_for_mean = 0
    mean_0 = 0.
    mean_1 = 0.
    # mean_0_all = 0.
    # mean_1_all = 0.
    for time_index in range(cycle_time):
        instantaneous_error = np.abs(float(
            readout_res.segments[0].filter(name='gsyn_inh')[0][time_index+(cycle*cycle_time)][0]))
        cycle_error[cycle] += instantaneous_error
        total_error += instantaneous_error
    if cycle_error[cycle] < 75:
        correct_or_not[cycle] = 1

        # if time_index > cycle_time - 150:
        #     ticks_for_mean = 1
        #     instantaneous_v0 = float(readout_res.segments[0].filter(name='v')[0][time_index+(cycle*cycle_time)][0])
        #     instantaneous_v1 = float(readout_res.segments[0].filter(name='v')[0][time_index+(cycle*cycle_time)][1])
        #     mean_0 = instantaneous_v0 * 0.1
        #     mean_1 = instantaneous_v1 * 0.1
        #     exp_0 = np.exp(mean_0 / ticks_for_mean)
        #     exp_1 = np.exp(mean_1 / ticks_for_mean)
        #     if exp_0 == 0 and exp_1 == 0:
        #         if instantaneous_v0 > instantaneous_v1:
        #             soft_max[0].append(1)
        #             soft_max[1].append(0)
        #         else:
        #             soft_max[0].append(0)
        #             soft_max[1].append(1)
        #     else:
        #         soft_max[0].append(-np.log(exp_0 / (exp_0 + exp_1)))
        #         soft_max[1].append(-np.log(exp_1 / (exp_0 + exp_1)))
        #     if float(readout_res.segments[0].filter(name='gsyn_exc')[0][time_index+(cycle*cycle_time)][2]) < 3.5:
        #         cross_entropy.append(soft_max[0][-1])
        #     else:
        #         cross_entropy.append(soft_max[1][-1])
        #
        #     from_soft[0].append(-np.log(float(readout_res.segments[0].filter(name='gsyn_exc')[0][time_index+(cycle*cycle_time)][0])))
        #     from_soft[1].append(-np.log(float(readout_res.segments[0].filter(name='gsyn_exc')[0][time_index+(cycle*cycle_time)][1])))
        # else:
        #     soft_max[0].append(0)
        #     soft_max[1].append(0)
        #     from_soft[0].append(0)
        #     from_soft[1].append(0)
        #     cross_entropy.append(0)
        #
        # instantaneous_v0 = float(readout_res.segments[0].filter(name='v')[0][time_index + (cycle * cycle_time)][0])
        # instantaneous_v1 = float(readout_res.segments[0].filter(name='v')[0][time_index + (cycle * cycle_time)][1])
        # # mean_0_all += instantaneous_v0
        # # mean_1_all += instantaneous_v1
        # exp_0 = np.exp(instantaneous_v0 * 0.1)
        # exp_1 = np.exp(instantaneous_v1 * 0.1)
        # all_cross[0].append(-np.log(exp_0 / (exp_0 + exp_1)))
        # all_cross[1].append(-np.log(exp_1 / (exp_0 + exp_1)))



new_connections_in = []#in_proj.get('weight', 'delay').connections[0]#[]
for partition in in_proj.get('weight', 'delay').connections:
    for conn in partition:
        new_connections_in.append(conn)
new_connections_in.sort(key=lambda x:x[1])
new_connections_in.sort(key=lambda x:x[0])
from_list_in.sort(key=lambda x:x[1])
from_list_in.sort(key=lambda x:x[0])
connection_diff_in = []
for i in range(len(from_list_in)):
    connection_diff_in.append(new_connections_in[i][2] - from_list_in[i][2])
print("Input connections\noriginal\n", np.array(from_list_in))
print("new\n", np.array(new_connections_in))
print("diff\n", np.array(connection_diff_in))

new_connections_out = []#out_proj.get('weight', 'delay').connections[0]#[]
for partition in out_proj.get('weight', 'delay').connections:
    for conn in partition:
        new_connections_out.append(conn)
new_connections_out.sort(key=lambda x:x[1])
new_connections_out.sort(key=lambda x:x[0])
from_list_out.sort(key=lambda x:x[1])
from_list_out.sort(key=lambda x:x[0])
connection_diff_out = []
for i in range(len(from_list_out)):
    connection_diff_out.append(new_connections_out[i][2] - from_list_out[i][2])
print("Output connections\noriginal\n", np.array(from_list_out))
print("new\n", np.array(new_connections_out))
print("diff\n", np.array(connection_diff_out))

if recurrent_connections:
    new_connections_rec = []#out_proj.get('weight', 'delay').connections[0]#[]
    for partition in recurrent_proj.get('weight', 'delay').connections:
        for conn in partition:
            new_connections_rec.append(conn)
    new_connections_rec.sort(key=lambda x:x[1])
    new_connections_rec.sort(key=lambda x:x[0])
    from_list_rec.sort(key=lambda x:x[1])
    from_list_rec.sort(key=lambda x:x[0])
    connection_diff_rec = []
    for i in range(len(from_list_out)):
        connection_diff_rec.append(new_connections_rec[i][2] - from_list_rec[i][2])
    print("Recurrent connections\noriginal\n", np.array(from_list_out))
    print("new\n", np.array(new_connections_out))
    print("diff\n", np.array(connection_diff_out))

print(experiment_label)
print("cycle_error =", cycle_error)
print("total error =", total_error)
print("correct:")# =", correct_or_not
for i in range(int(np.ceil(len(correct_or_not) / float(window_cycles)))):
    print(correct_or_not[i*window_cycles:(i+1)*window_cycles], np.average(correct_or_not[i*window_cycles:(i+1)*window_cycles]))
print("average error = ", np.average(cycle_error))
print("weighted average", np.average(cycle_error, weights=[i for i in range(num_repeats)]))
print("minimum error = ", np.min(cycle_error))
print("minimum iteration =", cycle_error.index(np.min(cycle_error)), "- with time stamp =", cycle_error.index(np.min(cycle_error)) * 1024)

plt.figure()
Figure(
    Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    title="neuron data for {}".format(experiment_label)
)
plt.show()

# plt.figure()
# Figure(
#     Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(runtime-(window_size*1.5), runtime)),
#
#     title="neuron data for {}".format(experiment_label)
# )
# plt.show()

fig, axs = plt.subplots(1, 1)
axs.set_title(experiment_label)
# axs[0].plot([i for i in range(len(cross_entropy))], cross_entropy)
# axs[1].plot([i for i in range(len(all_cross[0]))], all_cross[0])
# axs[2].plot([i for i in range(len(all_cross[1]))], all_cross[1])
axs.scatter([i for i in range(len(cycle_error))], cycle_error)
# axs[1].plot([i for i in range(len(from_soft[0]))], from_soft[0])
# axs[2].plot([i for i in range(len(from_soft[1]))], from_soft[1])
plt.show()

pynn.end()
print("job done")

'''
plt.figure()
plt.scatter([i for i in range(num_repeats)], cycle_error)
plt.title(experiment_label)
plt.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

v_mem = []
sine_wave = []
for timestep in readout_res.segments[0].filter(name='v')[0]:
    v_mem.append(timestep[0])
    sine_wave.append(timestep[1])

ave_mem = moving_average(v_mem, 20)
ave_sine = moving_average(sine_wave, 20)

plt.figure()
plt.plot([i for i in range(len(ave_mem))], cum_mem)
plt.plot([i for i in range(len(ave_sine))], cum_sine)
plt.title(experiment_label)
plt.show()

plt.figure()
plt.plot([i for i in range(len(soft_max[0]))], soft_max[0])
plt.plot([i for i in range(len(soft_max[1]))], soft_max[1])
plt.title(experiment_label)
plt.show()
'''
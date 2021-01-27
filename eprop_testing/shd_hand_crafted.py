import spynnaker8 as pynn
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from eprop_testing.frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import tables
import pickle

def weight_distribution(pop_size):
    dist_weight = np.random.randn() / np.sqrt(pop_size)
    return dist_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0, base_weight=0.0):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
                neuron_syn_count += 1
                conn = [i, j, weight_distribution(pre_pop_size)+base_weight, delay_count]
                delay_count += 1
                connections.append(conn)
        if neuron_syn_count > max_syn_per_neuron:
            max_syn_per_neuron = neuron_syn_count
    return connections, max_syn_per_neuron

def shd_to_spike_array(file_name):
    spike_array = [[] for i in range(700)]
    data_labels = []
    h5data = tables.open_file(file_name, mode='r')
    units = h5data.root.spikes.units
    times = h5data.root.spikes.times
    labels = h5data.root.labels
    data_count = 0
    for idx, label in enumerate(labels):
        if label < 10:
            data_labels.append(label)
            spike_times = times[idx]
            neuron_idx = units[idx]
            for spike, unit in zip(spike_times, neuron_idx):
                spike_array[unit].append((spike*1000)+(cycle_time*data_count))
            data_count += 1
    return spike_array, data_labels

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

np.random.seed(272727)

cycle_time = 1000
num_repeats = 950
# pynn.setup(1.0)

# file_name = '/data/mbaxrap7/Heidelberg speech/shd_test.h5'
# print("formatting data")
# spike_times, labels = shd_to_spike_array(file_name)
# print("pickling")
# filename = "shd_testing_english.pickle"
# outfile = open(filename, 'wb')
# pickle.dump([spike_times, labels], outfile)
# outfile.close()
infile = open("shd_training_english.pickle", 'rb')
spike_times, labels = pickle.load(infile)
infile.close()

reg_rate = 0.0000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = False
synapse_eta = 0.5
hidden_eta_modifier = 0.2
base_weight_in = 0.35
base_weight_out = 0.
base_weight_rec = 0.0
max_weight = 8
output_size = 10
threshold_beta = 0.3
ratio_of_LIF = 0.5
forced_w_fb = False


pynn.setup(timestep=1, max_delay=1000)


input_size = 700
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": labels,#[0:num_repeats],
    "eta": synapse_eta + 0.0,
    "update_ready": cycle_time
    }
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourceArray,
                            # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            # {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, input_split,
                            #                                            input_speed_up, input_size, use_old=False)},
                            {'spike_times': spike_times},

                            label='input_pop')

neuron_pop_size = 256
beta = []
w_fb = []
for i in range(neuron_pop_size):
    # if i < neuron_pop_size/2:
    if np.random.random() < ratio_of_LIF:
        beta.append(0)
        # beta.append(threshold_beta)
    else:
        beta.append(threshold_beta)
    if forced_w_fb:
        feedback_weights = [0. for j in range(output_size)]
        feedback_weights[np.random.randint(output_size)] = 1.
    else:
        feedback_weights = [np.random.random() for j in range(output_size)]
    w_fb.append(feedback_weights)
w_fb = np.array(w_fb).T.tolist()
neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0,
#     "w_fb": [[np.random.random() for j in range(output_size)] for i in range(neuron_pop_size)], # best it seems
#     "w_fb": [RandomDistribution("uniform", low=0.0, high=1.0) for i in range(output_size)], # best it seems
    "w_fb": w_fb,
    # "w_fb": [(np.random.random() * 2) - 1. for i in range(neuron_pop_size)],
    # "small_b": 1.0,
    "beta": beta,
    "target_rate": 70,#[10 + np.random.randn() for i in range(neuron_pop_size)],
    "eta": synapse_eta + hidden_eta_modifier,
    "tau_err": 1000*1.,
    "tau_a": 2*cycle_time,
    "window_size": cycle_time,
    "input_synapses": input_size,
    "rec_synapses": recurrent_connections * neuron_pop_size,
    "number_of_cues": 1,
    # "scalar": 1
    }
if neuron_pop_size:
    neuron = pynn.Population(neuron_pop_size,
                             pynn.extra_models.EPropAdaptive(**neuron_params),
                             label='eprop_pop')

# Output population
readout_pop = pynn.Population(output_size, # HARDCODED 1
                       pynn.extra_models.SHDReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

eprop_learning_output = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-max_weight, w_max=max_weight, reg_rate=0.0))

if neuron_pop_size:
    eprop_learning_neuron = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    from_list_in, max_syn_per_input = probability_connector(input_size, neuron_pop_size, p_connect_in,
                                                            base_weight=base_weight_in)
    if max_syn_per_input > 100:
        Exception
    else:
        print("max number of synapses per neuron:", max_syn_per_input)
    in_proj = pynn.Projection(input_pop,
                              neuron,
                              pynn.FromListConnector(from_list_in),
                              synapse_type=eprop_learning_neuron,
                              label='input_connections',
                              receptor_type='input_connections')

    # from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
    from_list_out, max_syn_per_output = probability_connector(neuron_pop_size, output_size, p_connect_out,
                                                            base_weight=base_weight_out)
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
                                    # pynn.OneToOneConnector(),
                                    # pynn.StaticSynapse(weight=[0.5], delay=[0]),
                                    pynn.AllToAllConnector(),
                                    pynn.StaticSynapse(weight=0.5, delay=0),
                                    receptor_type='learning_signal')
else:
    from_list_out, max_syn_per_output = probability_connector(input_size, output_size, p_connect_out,
                                                              base_weight=base_weight_out)
    if max_syn_per_output > 100:
        Exception
    else:
        print("max number of synapses per readout:", max_syn_per_output)
    out_proj = pynn.Projection(input_pop,
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')

if recurrent_connections:
    eprop_learning_recurrent = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec, offset=0,
                                                           base_weight=base_weight_rec)
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
if neuron_pop_size:
    neuron.record('all')
#     neuron.record('spikes')
#     neuron.record(['gsyn_exc', 'v', 'gsyn_inh'], indexes=[0, 1, 9, 17, 25, 33])
readout_pop.record('all')

# experiment_label = "eta:{}/{} - size:{}/{} - reg_rate:{} - p_conn:{}/{}/{} - rec:{} - 10*{}hz all2all".format(
#     readout_neuron_params["eta"], neuron_params["eta"], input_size, neuron_pop_size, reg_rate, p_connect_in, p_connect_rec, p_connect_out, recurrent_connections, input_split)
experiment_label = "english training - base_w in{} out{} rec{}{} - eta h{}r{} - b{}-{} - w_fb{}".format(
    base_weight_in, base_weight_out, base_weight_rec, recurrent_connections,
    neuron_params["eta"], readout_neuron_params["eta"],
    threshold_beta, ratio_of_LIF, forced_w_fb)
print("\n", experiment_label, "\n")

runtime = cycle_time * num_repeats
pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
if neuron_pop_size:
    neuron_res = neuron.get_data('all')
readout_res = readout_pop.get_data(['v', 'gsyn_exc', 'gsyn_inh'])  # ('all')

total_error = 0.0
cycle_error = [0.0 for i in range(num_repeats)]
cycle_classification = [-1 for i in range(cycle_time)]
test_classification = []
confusion_matrix = [[0. for i in range(output_size)] for i in range(output_size)]
for cycle in range(num_repeats):
    for time_index in range(cycle_time):
        instantaneous_error = np.abs(float(
            readout_res.segments[0].filter(name='gsyn_inh')[0][time_index+(cycle*cycle_time)][0]))
        cycle_error[cycle] += instantaneous_error
        total_error += instantaneous_error
        voltages = [0.0 for i in range(output_size)]
        for n_out in range(output_size):
            v_mem = np.abs(float(readout_res.segments[0].filter(name='v')[0][time_index + (cycle * cycle_time)][n_out]))
            voltages[n_out] = v_mem
        cycle_classification[time_index] = voltages.index(max(voltages))
    test_classification.append([labels[cycle], max(set(cycle_classification), key=cycle_classification.count)])  # mode
    confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
# for i in range(output_size):
#     total_tests = sum(confusion_matrix[i])
#     for j in range(output_size):
#         confusion_matrix[i][j] /= total_tests

correct_or_not = [int(i == j) for [i, j] in test_classification]

if neuron_pop_size:
    new_connections_in = []#in_proj.get('weight', 'delay').connections[0]#[]
    for partition in in_proj.get('weight', 'delay').connections:
        for conn in partition:
            new_connections_in.append(conn)
    new_connections_in.sort(key=lambda x:x[1])
    from_list_in.sort(key=lambda x:x[1])
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
from_list_out.sort(key=lambda x:x[1])
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
    from_list_rec.sort(key=lambda x:x[1])
    connection_diff_rec = []
    for i in range(len(from_list_out)):
        connection_diff_rec.append(new_connections_rec[i][2] - from_list_rec[i][2])
    print("Recurrent connections\noriginal\n", np.array(from_list_out))
    print("new\n", np.array(new_connections_out))
    print("diff\n", np.array(connection_diff_out))

# pynn.end()
# print("job done")

print(experiment_label)
print("cycle_error =", cycle_error)
print(experiment_label)
print("total error =", total_error)
print("classification = ", test_classification)
print("correct or not = ", correct_or_not)
print("\\", "|\t", end="")
for i in range(output_size):
    print("{:5}\t|\t".format(i), end="")
print("")
class_count = 0
for test_label in confusion_matrix:
    print(class_count, "|\t", end="")
    for choice in test_label:
        print("{:5}\t|\t".format(round(choice, 3)), end="")
    print("")
    class_count += 1
print("average classification = ", np.average(correct_or_not))
print("weighted average classification = ", np.average(correct_or_not, weights=[i for i in range(num_repeats)]))
print(experiment_label)
print("average error = ", np.average(cycle_error))
print("weighted average", np.average(cycle_error, weights=[i for i in range(num_repeats)]))
print("minimum error = ", np.min(cycle_error))
print("minimum iteration = ", cycle_error.index(np.min(cycle_error)), "- with time stamp =", cycle_error.index(np.min(cycle_error)) * 1024)

fig, axs = plt.subplots(2, 1)
df_cm = pd.DataFrame(confusion_matrix, range(output_size), range(output_size))
ave_corr10 = moving_average(cycle_error, 10)
ave_corr60 = moving_average(cycle_error, 60)
axs[0].scatter([i for i in range(num_repeats)], cycle_error)
axs[0].plot([i + 5 for i in range(len(ave_corr10))], ave_corr10, 'r')
axs[0].plot([i + 30 for i in range(len(ave_corr60))], ave_corr60, 'r')
axs[0].set_title(experiment_label)
axs[1] = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}) # font size
# plt.figure()
# plt.scatter([i for i in range(num_repeats)], cycle_error)
# plt.title(experiment_label)
plt.show()

if neuron_pop_size:
    start_time = runtime-cycle_time*10
    end_time = runtime
    plt.figure()
    Figure(
        Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True,
              xticks=True, xlim=(0, runtime-start_time)),

        Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True,
              xticks=True, xlim=(start_time, end_time)),

        Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

        Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

        title="neuron data for {}".format(experiment_label)
    )
    plt.show()
else:
    plt.figure()
    Figure(
        Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(runtime-cycle_time*3, runtime)),

        Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True,
              xlim=(runtime-cycle_time*3, runtime)),

        Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True,
              xlim=(0, runtime)),

        Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True,
              xlim=(0, runtime)),

        title="neuron data for {}".format(experiment_label)
    )
    plt.show()

print("hold")
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
'''
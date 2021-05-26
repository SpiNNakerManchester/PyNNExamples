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

def shd_to_split_array(file_name):
    data_labels = []
    data_spikes = []
    h5data = tables.open_file(file_name, mode='r')
    units = h5data.root.spikes.units
    times = h5data.root.spikes.times
    labels = h5data.root.labels
    data_count = 0
    for idx, label in enumerate(labels):
        if label < 10:
            spike_array = [[] for i in range(700)]
            data_labels.append(label)
            spike_times = times[idx]
            neuron_idx = units[idx]
            for spike, unit in zip(spike_times, neuron_idx):
                spike_array[unit].append((spike*1000))
            data_spikes.append(spike_array.copy())
    return data_spikes, data_labels

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

np.random.seed(272727)

cycle_time = 1000
num_repeats = 1000
# pynn.setup(1.0)

# file_name = '/data/mbaxrap7/Heidelberg speech/shd_train.h5'
# print("formatting data")
# spike_times, labels = shd_to_split_array(file_name)
# print("pickling")
# filename = "shd_testing_english_individual.pickle"
# outfile = open(filename, 'wb')
# pickle.dump([spike_times, labels], outfile)
# outfile.close()
infile = open("../shd_training_english.pickle", 'rb')
spike_times, labels = pickle.load(infile)
infile.close()

reg_rate = 0.0000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = False
synapse_eta = 0.001
hidden_eta_modifier = 0.#2
base_weight_in = 0.0
base_weight_out = 0.
base_weight_rec = 0.0
max_weight = 8
layers = 1
threshold_beta = 0.3
ratio_of_LIF = 0.5
output_size = 10
forced_w_fb = False
confusion_matrix_cutoff = 0.8


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

neuron_pop_size = 20
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
    neuron = []
    for i in range(layers):
        neuron.append(pynn.Population(neuron_pop_size,
                                      pynn.extra_models.EPropAdaptive(**neuron_params),
                                      label='eprop_pop{}'.format(i)))

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
                              neuron[0],
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
    out_proj = pynn.Projection(neuron[-1],
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')
    for i in range(layers):
        if i > 0:
            from_list_l, max_syn_per_output = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec,
                                                                      base_weight=base_weight_in)
            out_proj = pynn.Projection(neuron[i-1],
                                       neuron[i],
                                       # pynn.OneToOneConnector(),
                                       pynn.FromListConnector(from_list_l),
                                       synapse_type=eprop_learning_output,
                                       label='input_connections',
                                       receptor_type='input_connections')
        neuron[i].record('all')
        learning_proj = pynn.Projection(readout_pop,
                                        neuron[i],
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

    from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec,
                                                           offset=input_size,
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
#     neuron.record('spikes')
neuron[0].record(['gsyn_exc', 'v', 'gsyn_inh'])
readout_pop.record('all')

runtime = 200
pynn.run(runtime)
# pynn.run(runtime/2)
in_spikes = input_pop.get_data('spikes')
neuron_res = []
if neuron_pop_size:
    for i in range(layers):
        neuron_res.append(neuron[i].get_data('all'))
readout_res = readout_pop.get_data(['v', 'gsyn_exc', 'gsyn_inh'])  # ('all')

start_time = 0#runtime-cycle_time*10
end_time = runtime
plt.figure()
Figure(
    Panel(neuron_res[0].segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(neuron_res[0].segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(neuron_res[0].segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(neuron_res[0].segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True,
          xticks=True, xlim=(0, runtime-start_time)),

    Panel(neuron_res[0].segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True,
          xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

    title="neuron data for {}".format('test')
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
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

def max_syn_connector(pre_pop_size, post_pop_size, max_syn, base_weight=0.0):
    connections = []
    for i in range(post_pop_size):
        select_from = [j for j in range(pre_pop_size)]
        delay_index = 0
        for k in range(max_syn):
            syndex = np.random.choice(select_from)
            del select_from[select_from.index(syndex)]
            connections.append([syndex, i, weight_distribution(pre_pop_size)+base_weight, delay_index])
            delay_index += 1
    return connections

def prCyan(skk, end='/n'):
    print("\033[96m {}\033[00m" .format(skk), end=end)

def print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                 confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeat=0):
    # print(cycle_error)
    print(experiment_label)
    print(experiment_label)
    # print("cycle_error =", cycle_error)
    print("correct or not = ", correct_or_not)
    print("\\", "|\t", end="")
    for i in range(output_size):
        print("{:5}\t|\t".format(i), end="")
    print("")
    class_count = 0
    for i, test_label in enumerate(confusion_matrix):
        print(class_count, "|\t", end="")
        for j, choice in enumerate(test_label):
            if i == j:
                prCyan("{:4}\t|\t".format(round(choice, 3)), end="")
            else:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
        print("")
        class_count += 1
    print("")
    print("\\", "|\t", end="")
    for i in range(output_size):
        print("{:5}\t|\t".format(i), end="")
    print("")
    class_count = 0
    for i, test_label in enumerate(final_confusion_matrix):
        print(class_count, "|\t", end="")
        for j, choice in enumerate(test_label):
            if i == j:
                prCyan("{:4}\t|\t".format(round(choice, 3)), end="")
            else:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
        print("")
        class_count += 1
    for i in range(int(len(correct_or_not)/window_cycles)):
        if i*window_cycles in cue_break:
            print("increased classes")
        print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
              np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]),
              '({})'.format(np.average(correct_or_not[max(((i + 1) * window_cycles)-100, 0):(i + 1) * window_cycles])))
    print("average classification = ", np.average(correct_or_not))
    print("weighted average classification = ", np.average(correct_or_not,
                                                           weights=[i for i in range(len(correct_or_not))]))
    print("classification = ", test_classification)
    print(experiment_label)
    print(cue_break)
    print("current classes = ", len(cue_break) + output_size)
    # print("average error = ", np.average(cycle_error))
    # print("weighted average", np.average(cycle_error, weights=[i for i in range(len(cycle_error))]))
    # print("minimum error = ", np.min(cycle_error))
    # print("minimum iteration = ", cycle_error.index(np.min(cycle_error)), "- with time stamp =",
    #       cycle_error.index(np.min(cycle_error)) * 1024)
    print("iteration: ", current_window, "/", len(new_labels), " - repeat #", repeat)

def plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                        output_size,
                        address_string, test_label, save_flag=False, cue_break=[], plot_flag=False,
                        learning_threshold=0.75, no_classes=10):
    fig, axs = plt.subplots(2, 2)
    df_cm = pd.DataFrame(confusion_matrix, range(output_size+1), range(output_size+1))
    f_df_cm = pd.DataFrame(final_confusion_matrix, range(output_size+1), range(output_size+1))
    ave_err10 = moving_average(cycle_error, 10)
    ave_err100 = moving_average(cycle_error, 100)
    ave_err1000 = moving_average(cycle_error, 1000)
    axs[0][0].scatter([i for i in range(len(cycle_error))], cycle_error)
    axs[0][0].plot([i + 5 for i in range(len(ave_err10))], ave_err10, 'r')
    axs[0][0].plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][0].plot([i + 500 for i in range(len(ave_err1000))], ave_err1000, 'g')
    axs[0][0].plot([0, len(cycle_error)], [900, 900], 'g')
    axs[0][0].set_xlim([0, len(cycle_error)])
    axs[0][0].set_ylim([0, 1000])
    axs[0][0].set_title("cycle error")
    for iteration_break in cue_break:
        axs[0][0].axvline(x=iteration_break, color='b')
    ave_corr10 = moving_average(correct_or_not, 10)
    ave_err100 = moving_average(correct_or_not, 100)
    ave_err1000 = moving_average(correct_or_not, 1000)
    axs[0][1].scatter([i for i in range(len(correct_or_not))], correct_or_not)
    axs[0][1].plot([i + 5 for i in range(len(ave_corr10))], ave_corr10, 'r')
    axs[0][1].plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][1].plot([i + 500 for i in range(len(ave_err1000))], ave_err1000, 'g')
    axs[0][1].plot([0, len(correct_or_not)], [0.5, 0.5], 'r')
    random_chance = 1. / float(no_classes)
    axs[0][1].plot([0, len(correct_or_not)], [random_chance, random_chance], 'r')
    axs[0][1].plot([0, len(correct_or_not)], [learning_threshold, learning_threshold], 'g')
    for iteration_break in cue_break:
        axs[0][1].axvline(x=iteration_break, color='b')
    axs[0][1].set_xlim([0, len(correct_or_not)])
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_title("classification error")
    for iteration_break in cue_break:
        axs[0][1].axvline(x=iteration_break, color='b')
    axs[1][0] = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, ax=axs[1][0])  # font size
    axs[1][0].set_title("full confusion matrix")
    axs[1][1] = sn.heatmap(f_df_cm, annot=True, annot_kws={"size": 8}, ax=axs[1][1])  # font size
    axs[1][1].set_title("window confusion matrix")
    plt.suptitle(test_label, fontsize=16)
    # plt.title(experiment_label)
    if plot_flag:
        plt.show()

    # plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # manager.full_screen_toggle()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    # print(manager.window.maxsize())
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_flag:
        plt.savefig(address_string + test_label + " learning curve.png", bbox_inches='tight', dpi=200)
    # plt.show()
    plt.close()

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

def create_tests_and_labels(selected_classes):
    test_idx = [i for i in range(min(max_tests, 399*len(selected_classes)))]
    class_idx = [[np.random.randint(10), np.random.randint(10)] for i in range(700)]
    new_spikes = [[] for i in range(700)]
    new_labels = [-1 for i in range(len(test_idx))]
    np.random.shuffle(test_idx)
    count = -1
    for spike, label in zip(spike_times, labels):
        if label in selected_classes:
            count += 1
            new_labels[test_idx[count]] = label
            for idx, neuron in enumerate(spike):
                if label in class_idx[idx]:
                    new_spikes[idx] = new_spikes[idx] + list(map(lambda x: x + (1000*test_idx[count]), neuron))
        if count+1 >= max_tests or count+1 >= len(test_idx):
            break
    return new_labels, new_spikes


def collect_tests_and_labels(selected_classes):
    test_idx = [i for i in range(min(max_tests, 399*len(selected_classes)))]
    new_spikes = [[] for i in range(700)]
    new_labels = [-1 for i in range(len(test_idx))]
    np.random.shuffle(test_idx)
    count = -1
    for spike, label in zip(spike_times, labels):
        if label in selected_classes:
            count += 1
            new_labels[test_idx[count]] = label
            for idx, neuron in enumerate(spike):
                new_spikes[idx] = new_spikes[idx] + list(map(lambda x: x + (1000*test_idx[count]), neuron))
        if count+1 >= max_tests or count+1 >= len(test_idx):
            break
    return new_labels, new_spikes

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
max_tests = 1001
# pynn.setup(1.0)

# file_name = '/data/mbaxrap7/Heidelberg speech/shd_train.h5'
# print("formatting data")
# spike_times, labels = shd_to_split_array(file_name)
# print("pickling")
# filename = "shd_testing_english_individual.pickle"
# outfile = open(filename, 'wb')
# pickle.dump([spike_times, labels], outfile)
# outfile.close()
# infile = open("../shd_training_english.pickle", 'rb')
# spike_times, labels = pickle.load(infile)
infile = open("./../shd_testing_english_individual.pickle", 'rb')
spike_times, labels = pickle.load(infile)
infile.close()

reg_rate = 0.0000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = False
synapse_eta = 0.0001
batch_length = 1
hidden_eta_modifier = 0.#2
base_weight_in = 0.0
base_weight_out = 0.
base_weight_rec = 0.0
max_weight = 16
max_shd_conn = 256
layers = 1
threshold_beta = 0.3
ratio_of_LIF = 0.5
output_size = 10
forced_w_fb = False
confusion_matrix_cutoff = 0.8
selected_classes = [i for i in range(output_size)]
# new_labels, new_spike_times = collect_tests_and_labels(selected_classes)
new_labels, new_spike_times = create_tests_and_labels(selected_classes)

test_label = 'eta({}) {} - mw {}'.format(batch_length, synapse_eta, max_weight)

pynn.setup(timestep=1, max_delay=1000)


input_size = 700
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": new_labels,#[0:num_repeats],
    "eta": synapse_eta + 0.0,
    "update_ready": cycle_time*batch_length
    }
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourceArray,
                            # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            # {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, input_split,
                            #                                            input_speed_up, input_size, use_old=False)},
                            {'spike_times': new_spike_times},

                            label='input_pop')


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


from_list_max = max_syn_connector(input_size, output_size, max_shd_conn)
out_proj = pynn.Projection(input_pop,
                           readout_pop,
                           # pynn.OneToOneConnector(),
                           pynn.FromListConnector(from_list_max),
                           synapse_type=eprop_learning_output,
                           label='input_connections',
                           receptor_type='input_connections')

input_pop.record('spikes')
readout_pop.record('all')

full_runtime = 1000 * len(new_labels)
runned_time = 0
current_iter = 0
window_size = 10*1000
window_cycles = 10
final_confusion_matrix = [[0. for i in range(output_size+1)] for j in range(output_size+1)]
confusion_matrix = [[0. for i in range(output_size+1)] for j in range(output_size+1)]
long_confusion_decay = 0.95
short_confusion_decay = 0.5
correct_or_not = []
cycle_error = []

while runned_time <= full_runtime - window_size:
    pynn.run(window_size)
    readout_res = readout_pop.get_data(['gsyn_exc', 'v', 'gsyn_inh'])#, clear=True)

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            confusion_matrix[i][j] *= long_confusion_decay
            final_confusion_matrix[i][j] *= short_confusion_decay
    test_classification = []
    for cycle in range(window_cycles):
        correct_or_not.append([])
        # cycle_classification = [-1 for i in range(cycle_time)]
        ce = [0.0 for i in range(output_size)]
        for time_index in range(cycle_time):
            instantaneous_error = np.abs(float(
                readout_res.segments[0].filter(name='gsyn_inh')[0][time_index + ((cycle+current_iter) * cycle_time)][0]))
            cycle_error.append(instantaneous_error)
            # softmaxes = [0.0 for i in range(output_size)]
            for n_out in range(output_size):
                ce[n_out] += float(readout_res.segments[0].filter(name='v')[0][time_index + ((cycle+current_iter) * cycle_time)][n_out])
                # ce[n_out] = np.exp(max(min(8.75, v_mem), -8.75))
            # ce_sum = sum(ce)
            # for i in range(len(ce)):
            #     softmaxes[i] += ce[i] / ce_sum
        if sum(ce) == 0:
            choice = output_size
        else:
            choice = ce.index(max(ce))
        test_classification.append(
            [new_labels[cycle+current_iter], choice])  # mode
        # print("current label = ", cycle+(current_window*window_cycles))
        confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
        final_confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
        correct_or_not[-1] = int(test_classification[-1][0] == test_classification[-1][1])
    runned_time += window_size
    current_iter += window_cycles
    print_status(current_iter, test_label, [], test_classification, correct_or_not,
                 confusion_matrix, final_confusion_matrix, new_labels, [], repeat=0)

address_string = './../shd_graphs/'
plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                        output_size,
                        address_string, test_label, save_flag=True, cue_break=[], plot_flag=False,
                        learning_threshold=0.75, no_classes=10)

# pynn.run(runtime/2)
in_spikes = input_pop.get_data('spikes')
# readout_res = readout_pop.get_data(['v', 'gsyn_exc', 'gsyn_inh'])  # ('all')

start_time = 0#runned_time-cycle_time*10
end_time = runned_time
plt.figure()
Figure(
    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

    title="neuron data for {}".format(test_label)
)

figure = plt.gcf()  # get current figure
figure.set_size_inches(16, 9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./../shd_graphs/' + test_label + " full learning curve.png", bbox_inches='tight', dpi=200)
plt.close()

start_time = runned_time-cycle_time*5
end_time = runned_time
plt.figure()
Figure(
    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(start_time, end_time)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(start_time, end_time)),

    title="neuron data for {}".format(test_label)
)

figure = plt.gcf()  # get current figure
figure.set_size_inches(16, 9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./../shd_graphs/' + test_label + " final learning curve.png", bbox_inches='tight', dpi=200)
plt.close()
# plt.show()

# weights =

print("hold")
pynn.end()
print("job done")
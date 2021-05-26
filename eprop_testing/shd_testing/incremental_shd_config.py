import numpy as np
import pickle
import sys
import random

np.random.seed(27)

cycle_time = 1000
num_repeats = 1000
max_tests = 1001
no_syn_classes = 9.5
class_list = [i for i in range(10)]
if no_syn_classes % 1 == 0:
    class_idx = [random.sample(class_list, no_syn_classes) for i in range(700)]
else:
    class_idx = []
    prob = no_syn_classes - float(int(no_syn_classes))
    for i in range(700):
        if np.random.random() > prob:
            class_idx.append(random.sample(class_list, int(no_syn_classes)))
        else:
            class_idx.append(random.sample(class_list, int(no_syn_classes) + 1))

free_label = "batchin{} ".format(no_syn_classes)
read_in_arg = True

reg_rate = 0.0000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = False
# recurrent_connections = True
readout_eta = 0.001
synapse_eta = 0.001
lr_decay = 0.99
batch_size = 1
firing_reg = True
# firing_reg = False
target_rate = 5
v_mem_lr = 1
# v_mem_lr = 0.001
firing_lr = 1
hidden_eta_modifier = -.0
readout_eta_multiplier = 1
if firing_reg:
    base_weight_in = 0.0
else:
    base_weight_in = 0.03
base_weight_out = 0.
base_weight_rec = 0.0
max_weight = 4
neuron_pop_size = 256
layers = 1
threshold_beta = 0.3
# threshold_beta = 3
ratio_of_LIF = 0.5
output_size = 10
forced_w_fb = False
w_fb_type = 3
fb_multiplier = 1.
confusion_matrix_cutoff = 0.8
long_confusion_decay = 0.95
short_confusion_decay = 0.5

window_cycles = 50
learning_threshold = 0.7
repeat_on_fail = True
no_class_start = 10
class_progress = 1
random_class_order = False
record_data = False
plot_lc_per_iteration = True
plot_weights = True

class_order = [output_size-1-i for i in range(output_size)]
if random_class_order:
    np.random.shuffle(class_order)

if read_in_arg:
    synapse_eta = float(sys.argv[1])
    readout_eta = float(sys.argv[2])
    synapse_eta = readout_eta
    v_mem_lr = float(sys.argv[3])
    firing_lr = float(sys.argv[4])
    w_fb_type = int(sys.argv[5])
    recurrent_connections = bool(int(sys.argv[6]))
    fb_multiplier = float(sys.argv[7])
    batch_size = int(sys.argv[8])
    print("collected variables", sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
          recurrent_connections, fb_multiplier)
else:
    print("no variables were collected - running from config")

infile = open("./../shd_testing_english_individual.pickle", 'rb')
spike_times, labels = pickle.load(infile)
infile.close()

input_size = 700
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": labels,#[0:num_repeats],
    "eta": readout_eta * readout_eta_multiplier,
    "window_size": cycle_time*batch_size,
    # "update_ready": cycle_time
    }

beta = []
w_fb = []
for i in range(neuron_pop_size):
    if i < neuron_pop_size * ratio_of_LIF:
    # if np.random.random() < ratio_of_LIF:
        beta.append(0)
        # beta.append(threshold_beta)
    else:
        beta.append(threshold_beta)
    if forced_w_fb:
        feedback_weights = [0. for j in range(output_size)]
        feedback_weights[np.random.choice([2, 5])] = 1.
    else:
        if w_fb_type == 0:
            feedback_weights = [np.random.random()*2 for j in range(output_size)]
        elif w_fb_type == 1:
            feedback_weights = [(np.random.random()*2)-1 for j in range(output_size)]
        elif w_fb_type == 2:
            feedback_weights = [np.random.randn() / np.sqrt(neuron_pop_size) for j in range(output_size)]
        elif w_fb_type == 3:
            feedback_weights = [np.random.randn() for j in range(output_size)]
        elif w_fb_type == 4:
            feedback_weights = [0. for j in range(output_size)]
            feedback_weights[np.random.randint(0, 10)] = 3.
        elif w_fb_type == 5:
            feedback_weights = [-3. for j in range(output_size)]
            feedback_weights[np.random.randint(0, 10)] = 3.
        else:
            print("incorrect feedback choice")
            Exception
        feedback_weights = (np.array(feedback_weights) * fb_multiplier).tolist()
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
    "target_rate": target_rate*firing_reg,#[10 + np.random.randn() for i in range(neuron_pop_size)],
    "eta": synapse_eta + hidden_eta_modifier,
    "tau_err": 1000*1.,
    "tau_a": cycle_time,
    "window_size": cycle_time*batch_size,
    "input_synapses": input_size,
    "rec_synapses": recurrent_connections * neuron_pop_size,
    "number_of_cues": 1,
    "v_mem_lr": v_mem_lr,
    "firing_lr": firing_lr
    # "scalar": 1
    }


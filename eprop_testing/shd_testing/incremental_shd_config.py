import numpy as np
import pickle

np.random.seed(272727)

cycle_time = 1000
num_repeats = 1000

reg_rate = 0.0000
p_connect_in = 1.
p_connect_rec = 1.
p_connect_out = 1.
recurrent_connections = False
synapse_eta = 0.2
batch_size = 20
hidden_eta_modifier = -.0
readout_eta_multiplier = 1
firing_reg = True
target_rate = 10
base_weight_in = 0.03
base_weight_out = 0.
base_weight_rec = 0.0
max_weight = 8
neuron_pop_size = 100
layers = 1
threshold_beta = 0.3
ratio_of_LIF = 0.5
output_size = 10
forced_w_fb = False
confusion_matrix_cutoff = 0.8
confusion_decay = 0.99

window_cycles = 10
learning_threshold = 0.7
no_class_start = 2
class_progress = 1
random_class_order = True
record_data = True
plot_membranes = False

class_order = [i for i in range(output_size)]
if random_class_order:
    np.random.shuffle(class_order)

infile = open("./../shd_testing_english_individual.pickle", 'rb')
spike_times, labels = pickle.load(infile)
infile.close()

input_size = 700
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": labels,#[0:num_repeats],
    "eta": synapse_eta * readout_eta_multiplier,
    "window_size": cycle_time*batch_size,
    # "update_ready": cycle_time
    }

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
        feedback_weights[np.random.choice([2, 5])] = 1.
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
    "target_rate": target_rate*firing_reg,#[10 + np.random.randn() for i in range(neuron_pop_size)],
    "eta": synapse_eta + hidden_eta_modifier,
    "tau_err": 1000*1.,
    "tau_a": 2*cycle_time,
    "window_size": cycle_time*batch_size,
    "input_synapses": input_size,
    "rec_synapses": recurrent_connections * neuron_pop_size,
    "number_of_cues": 1,
    # "scalar": 1
    }


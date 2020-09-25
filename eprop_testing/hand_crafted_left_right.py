import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from PyNN8Examples.eprop_testing.frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager
from PyNN8Examples.eprop_testing.plot_graph import draw_graph_from_list, plot_learning_curve

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size) #+ 0.5
    # base_weight = 0
    return base_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
                neuron_syn_count += 1
                conn = [i, j, weight_distribution(pre_pop_size), delay_count]
                delay_count += 1
                connections.append(conn)
        if neuron_syn_count > max_syn_per_neuron:
            max_syn_per_neuron = neuron_syn_count
    return connections, max_syn_per_neuron

def range_connector(pre_min, pre_max, post_min, post_max, weight=1.5, delay_offset=0):
    connections = []
    for j in range(int(post_min), int(post_max)):
        # delay = delay_offset
        for i in range(int(pre_min), int(pre_max)):
            nd_weight = weight_distribution(pre_max-pre_min)
            connections.append([i, j, weight+nd_weight, i+delay_offset])
            # delay += 1
    return connections

np.random.seed(272727)

number_of_cues = 1
cycle_time = (number_of_cues*150)+1000+150
num_repeats = 1000
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
recurrent_connections = False
synapse_eta = 0.01
tau_a = 2500#[cycle_time - 150 + (np.random.randn() * 200) for i in range(100)]
input_split = 100
window_cycles = 2
window_size = cycle_time*window_cycles
threshold_beta = .3

max_weight = 8.0
in_weight = 0.55
prompt_weight = 0.55
rec_weight = 0#-0.5
out_weight = 0
weight_string = "i{}-p{}-r{}-o{}".format(in_weight, prompt_weight, rec_weight, out_weight)
pynn.setup(timestep=1)


input_size = 40
readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "poisson_pop_size": input_size / 4,
    "rate_on": 100,
    "rate_off": 0,
    # "tau_m": tau_a,
    "w_fb": [1, -1, 0],
    "eta": synapse_eta * 10.,
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

neuron_pop_size = 4*25
ratio_of_LIF = 0.5
beta = []
for i in range(neuron_pop_size):
    if i < neuron_pop_size/2:
    # if i % 2 == 0:
        beta.append(0)
    else:
        beta.append(threshold_beta)
neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0,
    # "w_fb": [np.random.random() for i in range(neuron_pop_size)], # best it seems
    # "w_fb": [(np.random.random() * 2) - 1. for i in range(neuron_pop_size)],
    "w_fb": [4*np.random.random() - 4*np.random.random() for i in range(neuron_pop_size)],  ## for both feedback weights
    # "w_fb": [-3]*(neuron_pop_size/2) + [3]*(neuron_pop_size/2),
    # "w_fb": [3]*int(neuron_pop_size/4) + [-3]*int(neuron_pop_size/4) + [3]*int(neuron_pop_size/4) + [-3]*int(neuron_pop_size/4),
    # "B": 0.0,
    "beta": beta,
    "target_rate": 10,
    "tau_a": tau_a,
    "eta": synapse_eta * 5,#/ 20.,
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

SpynnakerExternalDevicePluginManager.add_edge(readout_pop._get_vertex, input_pop._get_vertex, "CONTROL")

start_w = [weight_distribution(neuron_pop_size*input_size) for i in range(input_size)]
eprop_learning_neuron = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

# from_list_in, max_syn_per_input = probability_connector(input_size, neuron_pop_size, p_connect_in)
ps = int(readout_neuron_params["poisson_pop_size"])
# from_list_in = range_connector(0, ps, 0, neuron_pop_size/2, weight=in_weight)  # connect 1/2st 2 left
# from_list_in += range_connector(ps, ps*2, neuron_pop_size/2, neuron_pop_size, weight=in_weight)  # connect 2/2nd 2 right
# from_list_in = range_connector(0, ps*2, 0, neuron_pop_size, weight=in_weight)  # connect all cues to pop
# from_list_in += range_connector(ps*2, ps*3, 0, neuron_pop_size, delay_offset=0, weight=prompt_weight)  # connect all 2 prompt
from_list_in = range_connector(0, ps*2, 0, neuron_pop_size, delay_offset=0, weight=in_weight)  # connect all 2 prompt
from_list_in += range_connector(ps*2, ps*4, 0, neuron_pop_size, delay_offset=0, weight=prompt_weight)  # connect all 2 prompt
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
        w_min=-max_weight, w_max=max_weight, reg_rate=0.0))

# from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
# from_list_out, max_syn_per_output = probability_connector(neuron_pop_size, 2, p_connect_out)
# from_list_out = range_connector(0, neuron_pop_size/2, 1, 2, weight=out_weight)  # connect 1/2st 2 right output
# from_list_out += range_connector(neuron_pop_size/2, neuron_pop_size, 0, 1, weight=out_weight)  # connect 2/2nd 2 left output
# from_list_out += range_connector(0, neuron_pop_size/2, 0, 1, weight=-out_weight)  # connect 1/2st -2 left output
# from_list_out += range_connector(neuron_pop_size/2, neuron_pop_size, 1, 2, weight=-out_weight)  # connect 2/2nd -2 right output
from_list_out = range_connector(0, neuron_pop_size, 0, 2, weight=out_weight)  # connect all
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
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    # from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec, offset=0)
    # from_list_rec = range_connector(0, neuron_pop_size/2, neuron_pop_size/2, neuron_pop_size, weight=rec_weight, delay_offset=100)  # inhibitory connections between 1/2s
    # from_list_rec += range_connector(neuron_pop_size/2, neuron_pop_size, 0, neuron_pop_size/2, weight=rec_weight, delay_offset=100)  # inhibitory connections between 1/2s
    from_list_rec = range_connector(0, neuron_pop_size, 0, neuron_pop_size, weight=rec_weight, delay_offset=100)  # recurrent connections
    recurrent_proj = pynn.Projection(neuron,
                                     neuron,
                                     pynn.FromListConnector(from_list_rec),
                                     synapse_type=eprop_learning_recurrent,
                                     label='recurrent_connections',
                                     receptor_type='recurrent_connections')

input_pop.record('spikes')
neuron.record('spikes')
neuron.record(['gsyn_exc', 'v', 'gsyn_inh'], indexes=[i for i in range(int((neuron_pop_size/2)-5), int((neuron_pop_size/2)+5))])
readout_pop.record('all')

runtime = cycle_time * num_repeats

experiment_label = "eta:{}/{} - size:{}/{} - weights:{} - p_conn:{}/{}/{} - rec:{} - cycle:{}/{}/{} xavier b:{}".format(
    readout_neuron_params["eta"], neuron_params["eta"], input_size, neuron_pop_size, weight_string, p_connect_in, p_connect_rec, p_connect_out, recurrent_connections, cycle_time, window_size, runtime, threshold_beta)
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

new_connections_in = []#in_proj.get('weight', 'delay').connections[0]#[]
for partition in in_proj.get('weight', 'delay').connections:
    for conn in partition:
        new_connections_in.append(conn)
new_connections_in.sort(key=lambda x:x[0])
new_connections_in.sort(key=lambda x:x[1])
from_list_in.sort(key=lambda x:x[0])
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
new_connections_out.sort(key=lambda x:x[0])
new_connections_out.sort(key=lambda x:x[1])
from_list_out.sort(key=lambda x:x[0])
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
    new_connections_rec.sort(key=lambda x:x[0])
    new_connections_rec.sort(key=lambda x:x[1])
    from_list_rec.sort(key=lambda x:x[0])
    from_list_rec.sort(key=lambda x:x[1])
    connection_diff_rec = []
    for i in range(len(from_list_rec)):
        connection_diff_rec.append(new_connections_rec[i][2] - from_list_rec[i][2])
    print("Recurrent connections\noriginal\n", np.array(from_list_rec))
    print("new\n", np.array(new_connections_rec))
    print("diff\n", np.array(connection_diff_rec))

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
print(experiment_label)

# plt.figure()
# Figure(
#     Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),
#
#     title="neuron data for {}".format(experiment_label)
# )
# plt.show()

plot_start = runtime-(cycle_time*15)
plot_end = runtime
plt.figure()
Figure(
    Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(plot_start, plot_end)),

    title="neuron data for {}".format(experiment_label)
)
plt.show()

print("plotted neuron data")

fig, axs = plt.subplots(2, 1)
axs[0].set_title(experiment_label)
axs[0].scatter([i for i in range(len(correct_or_not))], correct_or_not)
axs[1].scatter([i for i in range(len(cycle_error))], cycle_error)
axs[1].plot([0, len(cycle_error)], [75, 75], 'r')
plt.show()

print("plotted curves")

pynn.end()
print("job done")

'''
draw_graph_from_list(new_connections_in, new_connections_rec, new_connections_out)

base_string = 'connection_lists/good 1 cue 20n recF'
np.save(base_string+' in.npy', new_connections_in)
np.save(base_string+' out.npy', new_connections_out)
np.save(base_string+' rec.npy', new_connections_rec)

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
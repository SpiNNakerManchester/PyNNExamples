import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size)
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

np.random.seed(272727)

cycle_time = 1024
num_repeats = 800 # 200
pynn.setup(1.0)

target_data = []
for i in range(1024):
            target_data.append(#1)
                0 + 2 * np.sin(2 * i * 2* np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2* np.pi / 1024))
                )

synapse_eta = 0.01

readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": target_data,
    "eta": synapse_eta
    }

pynn.setup(timestep=1)

input_size = 100
input_pop = pynn.Population(input_size,
                            pynn.SpikeSourceArray,
                            # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                            {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, 7., 7., input_size)},
                            label='input_pop')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.SinusoidReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

reg_rate = 0.00
eprop_learning_neuron = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=reg_rate))

from_list_in, max_syn_per_neuron = probability_connector(input_size, 1, 1.)
if max_syn_per_neuron > 250:
    Exception
else:
    print("max number fo synapses per neuron:", max_syn_per_neuron)
in_proj = pynn.Projection(input_pop,
                          readout_pop,
                          pynn.FromListConnector(from_list_in),
                          # pynn.AllToAllConnector(),
                          synapse_type=eprop_learning_neuron,
                          label='input_connections',
                          receptor_type='input_connections')

input_pop.record('spikes')
readout_pop.record('all')

experiment_label = "eta:{} - in size:{} - reg_rate: {}".format(readout_neuron_params["eta"], input_size, reg_rate)
print("\n", experiment_label, "\n")

runtime = cycle_time * num_repeats
pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
readout_res = readout_pop.get_data('all')

total_error = 0.0
cycle_error = [0.0 for i in range(num_repeats)]
for cycle in range(num_repeats):
    for time_index in range(1024):
        instantaneous_error = np.abs(float(
            readout_res.segments[0].filter(name='v')[0][time_index+(cycle*1024)][0]) - target_data[time_index])
        cycle_error[cycle] += instantaneous_error
        total_error += instantaneous_error

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
print(experiment_label)
print("cycle_error =", cycle_error)
print("total error =", total_error)
print("average error = ", np.average(cycle_error))
print("weighted average", np.average(cycle_error, weights=[i for i in range(num_repeats)]))
print("minimum error = ", np.min(cycle_error))
print("minimum iteration = ", cycle_error.index(np.min(cycle_error)))

plt.figure()
Figure(

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    title="neuron data for {}".format(experiment_label)
)
plt.show()

plt.figure()
plt.scatter([i for i in range(num_repeats)], cycle_error)
plt.title(experiment_label)
plt.show()

pynn.end()
print("job done")

'''
plt.figure()
plt.scatter([i for i in range(num_repeats)], cycle_error)
plt.title(experiment_label)
plt.show()
'''
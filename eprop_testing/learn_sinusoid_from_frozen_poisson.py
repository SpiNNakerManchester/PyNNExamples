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

np.random.seed(272727)

cycle_time = 1024
num_repeats = 200
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0.8,
    "v_rest": 0,
    "w_fb": 0.75
    }

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
    "eta": 0.01
    }

pynn.setup(timestep=1, max_delay=250)

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

# start_w = [(-0.5) / float(input_size*3) for i in range(input_size*3)]
# start_w = [0 for i in range(input_size*3)]
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

from_list = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
in_proj = pynn.Projection(input_pop,
                          # neuron,
                          readout_pop,
                          pynn.FromListConnector(from_list),
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

experiment_label = "eta: {} - in size:{}".format(readout_neuron_params["eta"], input_size)

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
new_connections = []
for conn in in_proj.get('weight', 'delay').connections[0]:
    if conn[2] > 3:
        new_connections.append([conn[0], conn[1], conn[2]-16, conn[3]])
    else:
        new_connections.append([conn[0], conn[1], conn[2], conn[3]])
connection_diff = []
for i in range(len(from_list)):
    connection_diff.append(new_connections[i][2] - from_list[i][2])
print "original\n", np.array(from_list)
print "new\n", np.array(new_connections)
print "diff\n", np.array(connection_diff)
print experiment_label
print "cycle_error =", cycle_error
print "total error =", total_error

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
plt.scatter([1739.792622420481, 1709.5202466031203, 1553.4356064017663, 1416.4611380004644, 1296.884425553637, 1194.575622946901, 1107.4903712199666, 1033.3835733187398, 971.0615212114659, 917.5133821256777, 872.6510081750777, 834.5469131202292, 801.932674085516, 774.9009456814987, 750.1415036479369, 729.6718713370992, 711.7966887307455, 695.9602492993286, 681.0463025990683, 667.6629473587994, 655.1788626995324, 644.2886597336307, 635.1718852422105, 627.580679006996, 620.2473888815699, 612.6768968519833, 606.322878448528, 600.5606061879085, 594.9049025460067, 590.3165570869079, 585.928947121106, 581.7914075857282, 577.8478734542942, 573.9005069692499, 570.2323602781323, 566.9547954787932, 563.5905659961875, 560.5811679842368, 557.9392042159176, 555.1969595994593, 552.4119156301078, 549.6176570604082, 547.2449373430011, 545.09023468779, 542.9563983030412, 540.7090468143625, 538.6339804357523, 536.551477050181, 534.5158960960372, 532.5783979620755, 530.7991372597328, 529.3332156311888, 527.5736331116575, 526.0769687214514, 524.4955884459596, 522.6853104495779, 520.7073037035847, 519.0256202926344, 517.2003068373479, 515.4950684255847, 514.0947884900232, 512.6683798676718, 511.30226238544725, 509.91720766153156, 508.629168621346, 507.28459464673665, 506.1595595993376, 505.0124112046502, 503.7347607616274, 502.5661122235975, 501.247681309569, 499.8701919955723, 498.5544368876105, 497.26920048088226, 496.0205642565987, 494.7760495942485, 493.55579462040606, 492.3615918288879, 491.31921419064423, 490.1703845647036, 488.9975355860119, 487.84398840548664, 486.7731965026004, 485.7425479488428, 484.8627736282796, 484.038425699403, 483.21525583918657, 482.56062135331445, 481.71157503867664, 480.9726149884655, 480.03081703859186, 479.1709425611602, 478.3041092203702, 477.5888913548276, 476.9374040358187, 476.32121311710847, 475.7890564964918, 475.1812203423324, 474.54878441446164, 473.9732929852731], [i for i in range(num_repeats)])
plt.show()
'''
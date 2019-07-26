import sys

import numpy as np
import pylab as plt

import spynnaker8 as sim
import time as local_time

sys.path.append("../")


def spike_raster_plot_8(
        spikes, plt, duration, ylim, scale_factor=0.001, title=None,
        filepath=None, file_format='pdf', file_name='', xlim=None,
        onset_times=None, pattern_duration=None, markersize=3,
        marker_colour='black', alpha=1., subplots=None, legend_strings=None):

    if len(spikes) > 0:
        neuron_index = 1
        spike_ids = []
        spike_times = []

        for times in spikes:
            for time in times:
                spike_ids.append(neuron_index)
                spike_times.append(time)
            neuron_index += 1
        scaled_times = [
            spike_time * scale_factor for spike_time in spike_times]

        # plot results
        if subplots is None:
            plt.figure(title)
            plt.xlabel("time (s)")
        else:
            ax = plt.subplot(subplots[0], subplots[1], subplots[2])
            ax.set_title(title)
            if subplots[2] == subplots[0]:
                plt.xlabel("time (s)")
            else:
                ax.set_xticklabels([])
        plt.plot(scaled_times, spike_ids, '.', markersize=markersize,
                 markerfacecolor=marker_colour, markeredgecolor='none',
                 markeredgewidth=0, alpha=alpha)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)
        plt.ylabel("neuron ID")

        if onset_times is not None:
            # plot block of translucent colour per pattern
            ax = plt.gca()
            pattern_legend = []
            legend_labels = []
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            # labels = ['A','B','C',]
            for i, pattern in enumerate(onset_times):
                pattern_legend.append(
                    plt.Line2D(
                        [0], [0], color=colours[i % 8], lw=4, alpha=0.2))
                legend_labels.append("s{}".format(i+1))
                for onset in pattern:
                    x_block = (onset, onset + scale_factor * pattern_duration)
                    ax.fill_between(
                        x_block, ylim, alpha=0.2, facecolor=colours[i % 8],
                        lw=0.5)
            plt.legend(
                pattern_legend, legend_labels,
                bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=len(onset_times), mode="expand", borderaxespad=0.)
        if xlim is not None:
            plt.xlim(xlim)
        if legend_strings is not None and (
                subplots is None or subplots[2] == 1):
            plt.legend(
                legend_strings, bbox_to_anchor=(0.1, 1.25), loc='upper center',
                ncol=len(legend_strings), markerscale=10.)
        if filepath is not None:
            if subplots is None or subplots[2] == subplots[0]:
                plt.savefig(
                    filepath + '/' +
                    file_name + '{}.'.format(title) + file_format)


# ============================================================================
# Simulation parameters
# ============================================================================

duration = 1000.
n_input = 10000.
n_per_core = 255

input_size = n_per_core * np.ceil(n_input / n_per_core)
target_pop_size = n_per_core * np.ceil((n_input * 2. / 3) / n_per_core)
inh_pop_size = n_per_core * np.ceil((n_input * 1. / 3) / n_per_core)

input_spikes = np.load('./input_spikes.npy').tolist()

spike_raster_plot_8(
    input_spikes, plt, duration / 1000.0, ylim=input_size + 1,
    title="input_activity")

list_file_name = './conn_list_{}input_scaled_id.npz'.format(input_size)

connection_list_file = np.load(list_file_name)
source_target_list = connection_list_file['source_target_list']
source_inh_list = connection_list_file['source_inh_list']
target_target_list = connection_list_file['target_target_list']
target_inh_list = connection_list_file['target_inh_list']
inh_inh_list = connection_list_file['inh_inh_list']
inh_target_list = connection_list_file['inh_target_list']

timestep = 1.

time_start = local_time.time()
sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, n_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp, n_per_core)

pop_size = max([input_size, target_pop_size, inh_pop_size])
source_pop = sim.Population(
    pop_size, sim.SpikeSourceArray(spike_times=input_spikes),
    label='input_pop')
target_pop = sim.Population(
    pop_size, sim.IF_cond_exp, {},
    label='target_pop_fixed_weight_scale_cond')
inh_pop = sim.Population(
    pop_size, sim.IF_cond_exp, {},
    label='inh_pop_fixed_weight_scale_cond')

target_pop.record('spikes')
inh_pop.record('spikes')
source_target_proj = sim.Projection(
    source_pop, target_pop, sim.FromListConnector(source_target_list),
    synapse_type=sim.StaticSynapse())
source_inh_proj = sim.Projection(
    source_pop, inh_pop, sim.FromListConnector(source_inh_list),
    synapse_type=sim.StaticSynapse())
target_lat_proj = sim.Projection(
    target_pop, target_pop, sim.FromListConnector(target_target_list),
    synapse_type=sim.StaticSynapse())
target_inh_proj = sim.Projection(
    target_pop, inh_pop, sim.FromListConnector(target_inh_list),
    synapse_type=sim.StaticSynapse())
inh_lat_proj = sim.Projection(
    inh_pop, inh_pop, sim.FromListConnector(inh_inh_list),
    synapse_type=sim.StaticSynapse(), receptor_type='inhibitory')
inh_target_proj = sim.Projection(
    inh_pop, target_pop, sim.FromListConnector(inh_target_list),
    synapse_type=sim.StaticSynapse(), receptor_type='inhibitory')

max_period = 5000.
num_recordings = int((duration / max_period) + 1)

for i in range(num_recordings):
    sim.run(duration / num_recordings)

target_data = target_pop.get_data(['spikes'])
output_spikes = target_data.segments[0].spiketrains

sim.end()

print(
    "simulation of {}s complete in {}s".format(
        duration / 1000., local_time.time() - time_start))

spike_raster_plot_8(
    output_spikes, plt, duration / 1000., pop_size + 1, 0.001,
    title="output pop activity")

non_zero_spikes = [
    train for train in output_spikes if len(train) > 0]
print(
    "non zero output spikes length:{} target n_neurons:{}".format(
        len(non_zero_spikes), target_pop_size))

plt.show()

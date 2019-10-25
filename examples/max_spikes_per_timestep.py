import spynnaker8 as sim

# Initialise simulator
sim.setup(timestep=1)


# Spike input
poisson_spike_source = sim.Population(250, sim.SpikeSourcePoisson(
    rate=50, duration=5000), label='poisson_source')

spike_source_array = sim.Population(250, sim.SpikeSourceArray,
                                    {'spike_times': [1000]},
                                    label='spike_source')


# Neuron Parameters
cell_params_exc = {
    'tau_m': 20.0, 'cm': 1.0, 'v_rest': -65.0, 'v_reset': -65.0,
    'v_thresh': -50.0, 'tau_syn_E': 5.0, 'tau_syn_I': 15.0,
    'tau_refrac': 0.3, 'i_offset': 0}

cell_params_inh = {
    'tau_m': 20.0, 'cm': 1.0, 'v_rest': -65.0, 'v_reset': -65.0,
    'v_thresh': -50.0, 'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
    'tau_refrac': 0.3, 'i_offset': 0}

# Neuronal populations
pop_exc = sim.Population(500, sim.IF_curr_exp(**cell_params_exc),
                         label='excitatory_pop')

pop_inh = sim.Population(125, sim.IF_curr_exp(**cell_params_inh),
                         label='inhibitory_pop')


# Generate random distributions from which to initialise parameters
rng = sim.NumpyRNG(seed=98766987, parallel_safe=True)

# Initialise membrane potentials uniformly between threshold and resting
pop_exc.set(v=sim.RandomDistribution('uniform',
                                     [cell_params_exc['v_reset'],
                                      cell_params_exc['v_thresh']],
                                      rng=rng))

# Distribution from which to allocate delays
delay_distribution = sim.RandomDistribution('uniform', [1, 10], rng=rng)

# Spike input projections
spike_source_projection = sim.Projection(spike_source_array, pop_exc,
    sim.FixedProbabilityConnector(p_connect=0.05),
    synapse_type=sim.StaticSynapse(weight=0.1, delay=3),
    receptor_type='excitatory')

# Poisson source projections
poisson_projection_exc = sim.Projection(poisson_spike_source, pop_exc,
    sim.FixedProbabilityConnector(p_connect=0.2),
    synapse_type=sim.StaticSynapse(weight=0.06, delay=3),
    receptor_type='excitatory')
poisson_projection_inh = sim.Projection(poisson_spike_source, pop_inh,
    sim.FixedProbabilityConnector(p_connect=0.2),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=3),
    receptor_type='excitatory')

# Recurrent projections
exc_exc_rec = sim.Projection(pop_exc, pop_exc,
    sim.FixedProbabilityConnector(p_connect=0.1),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=3),
    receptor_type='excitatory')
# exc_exc_one_to_one_rec = sim.Projection(pop_exc, pop_exc,
#     sim.OneToOneConnector(),
#     synapse_type=sim.StaticSynapse(weight=0.03, delay=delay_distribution),
#     receptor_type='excitatory')
inh_inh_rec = sim.Projection(pop_inh, pop_inh,
    sim.FixedProbabilityConnector(p_connect=0.1),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=3),
    receptor_type='inhibitory')

# Projections between neuronal populations
exc_to_inh = sim.Projection(pop_exc, pop_inh,
    sim.FixedProbabilityConnector(p_connect=0.2),
    synapse_type=sim.StaticSynapse(weight=0.06, delay=3),
    receptor_type='excitatory')
inh_to_exc = sim.Projection(pop_inh, pop_exc,
    sim.FixedProbabilityConnector(p_connect=0.2),
    synapse_type=sim.StaticSynapse(weight=0.06, delay=3),
    receptor_type='inhibitory')


# Specify output recording
pop_exc.record('all')
# pop_inh.record('spikes')
pop_inh.record('all')


# Run simulation
sim.run(simtime=5000)


# Extract results data
exc_data = pop_exc.get_data('spikes')
inh_data = pop_inh.get_data('spikes')

all_exc_pop_data = pop_exc.get_data('all')
all_inh_pop_data = pop_inh.get_data('all')

exc_v_data = all_exc_pop_data.segments[0].filter(name='v')[0]
exc_v_data = (((exc_v_data * 2**4) * 5) / 1000000) # convert to time in ms
exc_gsyn_exc_data = all_exc_pop_data.segments[0].filter(name='gsyn_exc')[0]
exc_gsyn_inh_data = all_exc_pop_data.segments[0].filter(name='gsyn_inh')[0]

inh_v_data = all_inh_pop_data.segments[0].filter(name='v')[0]
inh_v_data = (((inh_v_data * 2**4) * 5) / 1000000)
inh_gsyn_exc_data = all_inh_pop_data.segments[0].filter(name='gsyn_exc')[0]
inh_gsyn_inh_data = all_inh_pop_data.segments[0].filter(name='gsyn_inh')[0]

# from pyNN.utility.plotting import Figure, Panel
import matplotlib as mtpltlb

mtpltlb.rcParams.update({
    'font.size': 20,
    'font.family': 'Times New Roman'})
import matplotlib.pyplot as plt

# F = Figure(
#     Panel(gsyn_inh_data[:,1:3], ylabel="gsyn_inh"),
#     Panel(gsyn_exc_data[:,1:3], ylabel="gsyn_exc"),
#     Panel(v_data[:,1:3], ylabel="v"),)


# exc_max_spikes_per_tick = exc_gsyn_exc_data[:,254:256]
# exc_max_restarts = exc_gsyn_inh_data[:,254:256]
#
inh_max_spikes_per_tick = inh_gsyn_exc_data[:,0]
inh_max_restarts = inh_gsyn_inh_data[:,0]

exc_pop_mean_incoming_spikes = (exc_gsyn_exc_data[:,254].mean() + exc_gsyn_exc_data[:,255].mean())/2
inh_pop_mean_incoming_spikes = inh_gsyn_exc_data[:,0].mean()

exc_times_mean = (exc_v_data[:,254].mean() + exc_v_data[:,499].mean())/2
inh_times_mean = inh_v_data[:,124].mean()

plt.figure()
plt.suptitle("Excitatory Population")
plt.subplot(2,1,1)
plt.title("Spikes received per timestep (exc pop)", loc='right')
plt.plot(exc_gsyn_exc_data[:,254], label='Core A')
plt.plot(exc_gsyn_exc_data[:,499], label='Core B')
plt.plot(5000*[exc_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(exc_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)

plt.subplot(2,1,2)
plt.title("Total timestep sim time (exc pop)", loc='right')
plt.plot(exc_v_data[:,254], label='Core A')
plt.plot(exc_v_data[:,499], label='Core B')
plt.plot(5000*[exc_times_mean], color='green', label='mean = {}'.format(exc_times_mean))
plt.plot(5000*[1.0], color='purple', label='real-time')
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0, 2.25))
plt.ylabel('Sim Time (ms)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show(block=False)


plt.figure()
plt.suptitle("Excitatory Population")
plt.subplot(2,1,1)
plt.title("Spikes received per timestep (exc pop)", loc='right')
plt.plot(exc_gsyn_exc_data[:,254], label='Core A')
plt.plot(exc_gsyn_exc_data[:,499], label='Core B')
plt.plot(5000*[exc_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(exc_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)

plt.suptitle("Inhibitory Population")
plt.subplot(2,1,2)
plt.title("Spikes received per timestep (inh pop)", loc='right')
plt.plot(inh_max_spikes_per_tick,color='red')
plt.plot(5000*[inh_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(inh_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)


# plt.figure()
# plt.suptitle("Excitatory Population")
# plt.subplot(2,1,1)
# plt.title("Spikes received per timestep (inh pop)", loc='right')
# plt.plot(exc_gsyn_exc_data[:,254], label='Core A')
# plt.plot(exc_gsyn_exc_data[:,255], label='Core B')
# plt.plot(5000*[exc_pop_mean_incoming_spikes], color='green', label='mean= {}'.format(exc_pop_mean_incoming_spikes))
# plt.legend(loc='upper right')
# plt.xlim((0,5000))
# plt.ylim((0,60))
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show(block=False)




plt.figure()
plt.suptitle("Excitatory Population")
plt.subplot(2,1,1)
plt.title("Spikes received per timestep (exc pop)", loc='right')
plt.plot(exc_gsyn_exc_data[:,254], label='Core A')
plt.plot(exc_gsyn_exc_data[:,499], label='Core B')
plt.plot(5000*[exc_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(exc_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)

plt.subplot(2,1,2)
plt.title("Pipeline restarts per timestep (exc pop)", loc='right')
plt.plot(exc_gsyn_inh_data[:,254], label='Core A')
plt.plot(exc_gsyn_inh_data[:,499], label='Core B')
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,70))
plt.ylabel('Count')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show(block=False)


plt.figure()
plt.suptitle("Inhibitory Population")
plt.subplot(2,1,1)
plt.title("Spikes received per timestep (inh pop)", loc='right')
plt.plot(inh_max_spikes_per_tick,color='red')
plt.plot(5000*[inh_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(inh_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)

plt.subplot(2,1,2)
plt.title("Pipeline restarts per timestep (inh pop)", loc='right')
plt.plot(inh_max_restarts, color='red')
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,70))
plt.ylabel('Count')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show(block=False)


plt.figure()
plt.suptitle("Inhibitory Population")
plt.subplot(2,1,1)
plt.title("Spikes received per timestep (inh pop)", loc='right')
plt.plot(inh_max_spikes_per_tick,color='red')
plt.plot(5000*[inh_pop_mean_incoming_spikes], color='green', label='mean = {}'.format(inh_pop_mean_incoming_spikes))
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0,60))
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)

plt.subplot(2,1,2)
plt.title("Total timestep sim time (inh pop)", loc='right')
plt.plot(inh_v_data[:,124], color='red' )
plt.plot(5000*[inh_times_mean], color='green', label='mean = {}'.format(inh_times_mean))
plt.plot(5000*[1.0], color='purple', label='real-time')
plt.legend(loc='upper right')
plt.xlim((0,5000))
plt.ylim((0, 2.25))
plt.ylabel('Sim Time (ms)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show(block=False)


plt.show()

# Exit simulation
sim.end()


######## From Peter ######
from spynnaker8.utilities.neo_convertor import convert_spikes


sim_time = 5000

exc_spikes = convert_spikes(exc_data)
inh_spikes = convert_spikes(inh_data)

import matplotlib as mlib
import numpy as np

mlib.rcParams.update({'font.size': 25})

def instant_rates(spikes, simtime=sim_time, chunk_size=1, N_layer=500):
    per_neuron_instaneous_rates = np.empty((N_layer, int(np.ceil(simtime / chunk_size))))
    for neuron_index in np.arange(N_layer):

        firings_for_neuron = spikes[
            spikes[:, 0] == neuron_index]
        for chunk_index in np.arange(per_neuron_instaneous_rates.shape[
                                         1]):
            per_neuron_instaneous_rates[neuron_index, chunk_index] = \
                np.count_nonzero(
                    np.logical_and(
                        firings_for_neuron[:, 1] >= (
                                chunk_index * chunk_size),
                        firings_for_neuron[:, 1] < (
                                (chunk_index + 1) * chunk_size)
                    )
                ) / (1.0 * chunk_size)
    instaneous_rates = np.sum(per_neuron_instaneous_rates,
                              axis=0)  # / float(N_layer)  # uncomment this if you want mean firing rate

    return instaneous_rates


def plot_spikes(exc_spikes, inh_spikes, title, simtime=sim_time):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1],
                                                                                        'hspace': .1})
    ax1.set_xlim((0, simtime))
    ax1.scatter([i[1] for i in exc_spikes], [i[0] for i in exc_spikes], s=1,
                marker="*")

    ax1.scatter([i[1] for i in inh_spikes], [i[0] + 500 for i in inh_spikes],
                s=1, marker="*", c='r')
    ax1.set_ylabel('Neuron ID')
    ax1.set_title(title)
    # Include a histogram of mean firing activity
    chunk_size = 1  # ms

    inh_N_layer = 125
    inh_instant_rates_1_ms = instant_rates(inh_spikes, simtime, chunk_size, inh_N_layer)
    ax2.bar(np.arange(0, simtime, chunk_size), inh_instant_rates_1_ms, width=chunk_size, color='r')
    inh_avg_firing_rate = inh_spikes.shape[0] / float(simtime)
    ax2.axhline(inh_avg_firing_rate,
#                 linestyle='--',
                color='green',
                label="avg spikes emitted per ms: {}".format(inh_avg_firing_rate))
    ax2.legend(loc='upper right')

    N_layer = 500
    instant_rates_1_ms = instant_rates(exc_spikes, simtime, chunk_size, N_layer)
    ax3.bar(np.arange(0, simtime, chunk_size), instant_rates_1_ms, width=chunk_size, color='C0')
    exc_avg_firing_rate = exc_spikes.shape[0] / float(simtime)
    ax3.axhline(exc_avg_firing_rate,
#                 linestyle='--',
                color='green',
                label="avg spikes emitted per ms: {}".format(exc_avg_firing_rate))
    ax3.legend(loc='upper right')
    ax3.set_xlabel('Time(ms)')

    # Saving the figure
    plt.savefig(title + ".pdf", bbox_inches="tight")
    plt.savefig(title + ".svg", bbox_inches="tight")
    plt.savefig(title + ".png", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


plot_spikes(exc_spikes, inh_spikes, "Random Balanced Network")


print "Job done"
import spynnaker8 as sim
import numpy
import pylab as plt
#from signal_prep import *


def cell_voltage_plot_8(v, plt, duration_ms, time_step_ms,scale_factor=0.001, id=0, title=''):
    times = range(0,int(duration_ms),int(time_step_ms))
    scaled_times = [time*scale_factor for time in times]
    membrane_voltage = v[id]
    plt.figure(title + str(id + 1))
    plt.plot(scaled_times, membrane_voltage)

def spike_raster_plot_8(spikes,plt,duration,ylim,scale_factor=0.001,title=None):
    if len(spikes) > 0:
        neuron_index = 1
        spike_ids = []
        spike_times = []

        for times in spikes:
            for time in times:
                spike_ids.append(neuron_index)
                spike_times.append(time)
            neuron_index+=1
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]

        ##plot results
        plt.figure(title)
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)
        plt.ylabel("neuron ID")
        plt.xlabel("time (s)")

# Population parameters
model = sim.IF_curr_exp
cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

stim_size = 1.
noise_rate = 10.
duration = 1. * 1000.
w2s =2.
w_max = w2s/2.
start_weight = w_max/2.
a_plus = 0.03125
a_minus = 0.85 * a_plus

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

stim_times = [i*100. for i in range(int(duration))]

ext_stim = sim.Population(
    stim_size, sim.SpikeSourcePoisson(rate=noise_rate, duration=duration),
    label="stim_poisson_{}Hz".format(noise_rate))
#ext_stim = sim.Population(stim_size, sim.SpikeSourceArray(spike_times=stim_times))
ext_stim.record("spikes")

target_pop = sim.Population(1,sim.IF_curr_exp,cell_params,label="target")
target_pop.record(["spikes",'v'])
"""
# Plastic Connection between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=16.7, tau_minus=33.7, A_plus=a_plus, A_minus=a_minus),
    weight_dependence=sim.AdditiveWeightDependence(
        w_min=0.0, w_max=w_max), weight=start_weight)"""

stdp_proj=sim.Projection(
    ext_stim, target_pop, sim.OneToOneConnector(), receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=w2s))
#synapse_type=stdp_model)

sim.run(duration)

target_data =target_pop.get_data(["spikes","v"])
stim_data = ext_stim.get_data("spikes")

sim.end()

mem_v = target_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, 1.,scale_factor=0.001)
spike_raster_plot_8(stim_data.segments[0].spiketrains,plt,duration/1000.,stim_size+1,0.001)
spike_raster_plot_8(target_data.segments[0].spiketrains,plt,duration/1000.,2,0.001)

plt.show()
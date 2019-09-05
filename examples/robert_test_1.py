import pylab as plt
import numpy as np

import spynnaker8 as sim
from signal_prep import *
from spinnak_ear.spinnak_ear_pynn_model.spinnaker_ear_model import SpiNNakEar
from spynnaker8.utilities import neo_convertor

# ===========================================================================
# Neuron parameters
# ===========================================================================
bushy_params_cond = {
    'tau_syn_E': 2.,
    'v_reset': -60.,
    'v_rest': -60.,
    'v_thresh': -40.
}
w2s_b = 0.3

# ============================================================================
# Simulation parameters
# ============================================================================
moc_spikes = [[],[],[]]#[[10.], [20], [30, 31, 32, 33]]
moc_spikes_2 = [[],[],[]]#[[110.], [120], [130]]
Fs = 50e3#22e3#100000.#
dBSPL=50
freq = 1000
tone_duration = 0.05#0.2
silence_duration = 0.01#0.1 #0.075#

tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_stereo = np.asarray([tone,tone_r])

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
}

test_file = "tone_{}Hz_stereo".format(freq)#"timit"#
binaural_audio = sounds_dict[test_file]
# what the duration was, but needs to adjust for lowest common time step
duration = np.ceil((binaural_audio[0].size/Fs)*1000.)

#scale = 0.03
scale = 1./30
an_pop_size = int(scale * 30e3)

# ===========================================================================
# SpiNNaker setup
# ===========================================================================
sim.setup(timestep=1.0, time_scale_factor=10)

# ===========================================================================
# Populations
# ===========================================================================
left_ear = SpiNNakEar(
    audio_input=binaural_audio[0], fs=Fs, ear_index=0, scale=scale)
an_pop_size = left_ear.calculate_n_atoms()
spinnakear_pop_left = sim.Population(
    left_ear.calculate_n_atoms(), left_ear, label="spinnakear_pop_left")
spinnaker_pop_left.record(["inner_ear_spike_probability", 'moc'])
right_ear = SpiNNakEar(
    audio_input=binaural_audio[1], fs=Fs, ear_index=1, scale=scale)
spinnakear_pop_right = sim.Population(
    right_ear.calculate_n_atoms(), right_ear, label="spinnakear_pop_right")
spinnakear_pop_right.record(["inner_ear_spike_probability", 'moc'])

moc_pop = sim.Population(
    3, sim.SpikeSourceArray(spike_times=moc_spikes), label="moc_pop")
moc_pop_2 = sim.Population(
    3, sim.SpikeSourceArray(spike_times=moc_spikes_2), label="moc_pop_2")

target_pop_size = spinnakear_pop_right.outgoing_neurons()
if spinnakear_pop_left.outgoing_neurons() != target_pop_size:
    raise Exception("different sizes")

target_pop = sim.Population(target_pop_size, sim.IF_cond_exp, {},
                            label="target_fixed_weight_scale_cond")
target_pop.record(['spikes'])

# ==========================================================================
# Projections
# ==========================================================================
moc_ohc_connections = [(0, 0), (1, 1), (2, 2)]
moc_ohc_connections_2 = [(0, 5), (1, 6), (2, 7)]
moc_projection = sim.Projection(
    moc_pop, spinnakear_pop_left, sim.FromListConnector(moc_ohc_connections),
    synapse_type=sim.StaticSynapse(weight=1.))
moc_projection2 = sim.Projection(
    moc_pop, spinnakear_pop_right,
    sim.FromListConnector(moc_ohc_connections_2),
    synapse_type=sim.StaticSynapse(weight=1.))

one_to_one_list = [(i, i) for i in range(target_pop_size)]
target_projection_left = sim.Projection(
    spinnakear_pop_left, target_pop, sim.FromListConnector(one_to_one_list),
    synapse_type=sim.StaticSynapse(weight=0.05))
target_projection_right = sim.Projection(
    spinnakear_pop_right, target_pop, sim.FromListConnector(one_to_one_list),
    synapse_type=sim.StaticSynapse(weight=0.05))

sim.run(duration)

ear_left_data = spinnakear_pop_left.get_data()
#ear_spikes_left = ear_left_data.segments[0].spiketrains
#ear_spikes_left = neo_convertor.convert_spiketrains(ear_spikes_left)
#ear_moc_left = ear_left_data.segments[0].filter(name='moc')[0]

ear_right_data = spinnakear_pop_right.get_data()
#ear_spikes_right = ear_right_data.segments[0].spiketrains
#ear_moc_right = ear_right_data.segments[0].filter(name='moc')[0]

target_data = target_pop.get_data(['spikes'])
#target_spikes = target_data.segments[0].spiketrains

sim.end()
'''
spike_raster_plot_8(
    ear_spikes_left, plt, duration / 1000., an_pop_size + 1, 0.001,
    title="ear pop activity left")
spike_raster_plot_8(
    ear_spikes_right, plt, duration / 1000., an_pop_size + 1, 0.001,
    title="ear pop activity right")
spike_raster_plot_8(
    target_spikes, plt, duration / 1000., target_pop_size + 1, 0.001,
    title="target pop activity") 


legend_string = [str(i) for i in range(int(an_pop_size / 10))]
plt.figure("MOC left")
for moc_signal in ear_moc_left:
    x = np.linspace(0, duration, len(moc_signal))
    plt.plot(x, moc_signal)
plt.xlabel("time (ms)")
plt.legend(legend_string)

plt.figure("MOC right")
for moc_signal in ear_moc_right:
    x = np.linspace(0, duration, len(moc_signal))
    plt.plot(x, moc_signal)
plt.xlabel("time (ms)")
plt.legend(legend_string)
plt.show()'''
np.savez_compressed('./ear_' + test_file + '_{}an_fibres_{}dB_{}s'.format
                    (an_pop_size,dBSPL,int(duration / 1000.)), 
                    ear_data=np.asarray([ear_left_data,ear_right_data]), 
                    sg_data=sg_data,Fs=Fs,stimulus=binaural_audio)



import numpy as np

import spynnaker8 as sim
from signal_prep import *
from spinn_utilities.default_ordered_dict import DefaultOrderedDict
from spinnak_ear.spinnak_ear_pynn_model.spinnaker_ear_model import SpiNNakEar

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
moc_spikes = [[10.]]
moc_spikes_2 = [[110.], [120], [130]]
Fs = 50e3#22e3#100000.#
dBSPL=50
freq = 1000
tone_duration = 0.05#0.2
silence_duration = 0.01#0.1 #0.075#

tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,
                       silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,
                         silence=True,silence_duration=silence_duration)
tone_stereo = np.asarray([tone,tone_r])

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
}

test_file = "tone_{}Hz_stereo".format(freq)#"timit"#
binaural_audio = sounds_dict[test_file]
# what the duration was, but needs to adjust for lowest common time step
#duration = np.ceil((binaural_audio[0].size/Fs)*1000.)
duration = 72

#scale = 0.03
#scale = 1./30
scale = 10./30e3
an_pop_size = int(scale * 30e3)

# ===========================================================================
# SpiNNaker setup
# ===========================================================================
sim.setup(timestep=1.0, time_scale_factor=100)

# ===========================================================================
# Populations
# ===========================================================================
left_ear = SpiNNakEar(
    audio_input=binaural_audio[0], fs=Fs, ear_index=0, scale=scale,
    pole_freqs=[[1000]])
an_pop_size = left_ear.calculate_n_atoms()
spinnakear_pop_left = sim.Population(
    left_ear.calculate_n_atoms(), left_ear, label="spinnakear_pop_left")
spinnakear_pop_left.record(["inner_ear_spike_probability", 'moc'])
right_ear = SpiNNakEar(
    audio_input=binaural_audio[1], fs=Fs, ear_index=1, scale=scale,
    pole_freqs=[[1000]])
spinnakear_pop_right = sim.Population(
    right_ear.calculate_n_atoms(), right_ear, label="spinnakear_pop_right")
spinnakear_pop_right.record(["inner_ear_spike_probability", 'moc'])

moc_pop = sim.Population(
    1, sim.SpikeSourceArray(spike_times=moc_spikes), label="moc_pop")
#moc_pop_2 = sim.Population(
#    3, sim.SpikeSourceArray(spike_times=moc_spikes_2), label="moc_pop_2")

target_pop_size = spinnakear_pop_right.outgoing_neurons()
if spinnakear_pop_left.outgoing_neurons() != target_pop_size:
    raise Exception("different sizes")

target_pop = sim.Population(target_pop_size, sim.IF_cond_exp, {},
                            label="target_fixed_weight_scale_cond")
target_pop.record(['spikes'])

# ==========================================================================
# Projections
# ==========================================================================
moc_ohc_connections = [(0, 0)]
moc_ohc_connections_2 = [(0, 5), (0, 6), (0, 7)]
moc_projection = sim.Projection(
    moc_pop, spinnakear_pop_left, sim.FromListConnector(moc_ohc_connections),
    synapse_type=sim.StaticSynapse(weight=1.0))
moc_projection2 = sim.Projection(
    moc_pop, spinnakear_pop_right,
    sim.FromListConnector(moc_ohc_connections_2),
    synapse_type=sim.StaticSynapse(weight=1.0))

one_to_one_list = [(i, i) for i in range(
    spinnakear_pop_left.outgoing_neurons())]
target_projection_left = sim.Projection(
    spinnakear_pop_left, target_pop, sim.FromListConnector(one_to_one_list),
    synapse_type=sim.StaticSynapse(weight=0.05))
target_projection_right = sim.Projection(
    spinnakear_pop_right, target_pop, sim.FromListConnector(one_to_one_list),
    synapse_type=sim.StaticSynapse(weight=0.05))

sim.run(duration)

ear_left_data_moc = spinnakear_pop_left.spinnaker_get_data(["moc"])
ear_left_data_prob = spinnakear_pop_left.spinnaker_get_data([
    "inner_ear_spike_probability"])

ear_right_data_prob = spinnakear_pop_right.spinnaker_get_data([
    "inner_ear_spike_probability"])
ear_right_data_moc = spinnakear_pop_right.spinnaker_get_data([
    "moc"])

target_data = target_pop.get_data(['spikes'])
target_spikes = target_data.segments[0].spiketrains

left_moc_in_robert_format = list()
left_prob_in_robert_format = list()
right_moc_in_robert_format = list()
right_prob_in_robert_format = list()

print("starting sort")
prob_data = DefaultOrderedDict(list)
samples = len(spinnakear_pop_left._vertex.get_value("audio_input"))
position = 0
print("len is {}".format(len(ear_left_data_prob)))
for element in ear_left_data_prob:
    if not numpy.isnan(element[2]):
        prob_data[int(element[0])].append(element[2])
    else:
        prob_data[int(element[0])].append(0)
    position += 1

for fiber_id in prob_data:
    left_prob_in_robert_format.append(prob_data[fiber_id])

print("starting sort")
moc_data = DefaultOrderedDict(list)
for element in ear_left_data_moc:
    time = element[0]
    if element[1] < len(spinnakear_pop_left._vertex.get_value("audio_input")):
        moc_data[time].append(element[2])
for time in moc_data:
    left_moc_in_robert_format.append(moc_data[time])

print("starting sort")
prob_data = DefaultOrderedDict(list)
for element in ear_right_data_prob:
    if not numpy.isnan(element[2]):
        prob_data[int(element[0])].append(element[2])
    else:
        prob_data[int(element[0])].append(0)
    position += 1
for fiber_id in prob_data:
    right_prob_in_robert_format.append(prob_data[fiber_id])

print("starting sort")
moc_data = DefaultOrderedDict(list)
for element in ear_right_data_moc:
    time = element[0]
    if element[1] < len(spinnakear_pop_right._vertex.get_value("audio_input")):
        moc_data[time].append(element[2])
for time in moc_data:
    right_moc_in_robert_format.append(moc_data[time])

print("eee")

np.savez_compressed('./ear_' + test_file + '_{}an_fibres_{}dB_{}s'.format
                    (an_pop_size,dBSPL,int(duration / 1000.)), 
                    ear_data=np.asarray(
                        [{"moc": left_moc_in_robert_format,
                          "prob": left_prob_in_robert_format},
                         {"moc": right_moc_in_robert_format,
                          "prob": right_prob_in_robert_format}]),
                    Fs=Fs, stimulus=binaural_audio)


sim_data = np.load('./ear_tone_1000Hz_stereo_112.0an_fibres_50dB_0s.npz',
                   allow_pickle=True)
vrr = sim_data['ear_data']


sim.end()

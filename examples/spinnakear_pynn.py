import pylab as plt
import numpy as np

import spynnaker8 as sim
from signal_prep import *
from spinnak_ear.spinnaker_ear_model import SpiNNakEar
from spinnak_ear.spinnaker_ear_model import spinnakear_size_calculator
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint
from elephant.statistics import isi,cv

#================================================================================================
# Neuron parameters
#================================================================================================
bushy_params_cond = {#'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               #'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 2.,#2.5,#
               #'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.,#-70.0,
               'v_rest': -60.,
               'v_thresh': -40.
               }
w2s_b = 0.3#
#================================================================================================
# Simulation parameters
#================================================================================================
# moc_spikes = [[10.],[],[]]#,102,104]]
# moc_spikes = [[10.,15.,20.,100.,105.],[24.,95.,102.],[5.,35.,80.]]#,102,104]]
moc_spikes = [[10.],[20],[30,31,32,33,]]#,102,104]]
moc_spikes_2 = [[110.],[120],[130]]
# for _ in range(3):
#     moc_spikes.append([t for t in range(0,150,3)])
Fs = 50e3#100000.#
dBSPL=60
wav_directory = '../../OME_SpiNN/'

freq = 3000
tone_duration = 0.05
tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_stereo = np.asarray([tone,tone_r])
click = generate_signal(signal_type='click',fs=Fs,dBSPL=dBSPL,duration=0.0002,plt=None,silence=True,silence_duration=0.075)

# binaural_audio = np.asarray([np.tile(tone_1,5),np.tile(tone_1,4)])
# binaural_audio = np.asarray([np.tile(tone_1,5)])
binaural_audio = tone_stereo#np.asarray([click,click])#

duration = (binaural_audio[0].size/Fs)*1000.#max(input_spikes[0])

# plt.figure("audio stimulus")
# x = np.linspace(0,duration/1000,len(binaural_audio[0]))
# plt.plot(x,binaural_audio[0])
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
an_pop_size = spinnakear_size_calculator(scale=0.03)
spinnakear_pop_left = sim.Population(an_pop_size,SpiNNakEar(audio_input=binaural_audio[0],fs=Fs,n_channels=an_pop_size/10,ear_index=0),label="spinnakear_pop_left")
spinnakear_pop_left.record(['spikes','moc'])
spinnakear_pop_right = sim.Population(an_pop_size,SpiNNakEar(audio_input=binaural_audio[1],fs=Fs,n_channels=an_pop_size/10,ear_index=0),label="spinnakear_pop_right")
spinnakear_pop_right.record(['spikes','moc'])

moc_pop = sim.Population(3,sim.SpikeSourceArray(spike_times=moc_spikes),label="moc_pop")
moc_pop_2 = sim.Population(3,sim.SpikeSourceArray(spike_times=moc_spikes_2),label="moc_pop_2")

#================================================================================================
# Projections
#================================================================================================
moc_ohc_connectons = [(0,0),(1,1),(2,2)]
moc_ohc_connectons_2 = [(0,5),(1,6),(2,7)]
moc_projection = sim.Projection(moc_pop,spinnakear_pop_left,sim.FromListConnector(moc_ohc_connectons),synapse_type=sim.StaticSynapse(weight=1.))
moc_projection = sim.Projection(moc_pop,spinnakear_pop_right,sim.FromListConnector(moc_ohc_connectons_2),synapse_type=sim.StaticSynapse(weight=1.))

sim.run(duration)

ear_left_data = spinnakear_pop_left.get_data()
ear_spikes_left = ear_left_data['spikes']
ear_moc_left = ear_left_data['moc']

ear_right_data = spinnakear_pop_right.get_data()
ear_spikes_right = ear_right_data['spikes']
ear_moc_right = ear_right_data['moc']
sim.end()

spike_raster_plot_8(ear_spikes_left, plt, duration / 1000., an_pop_size + 1, 0.001, title="ear pop activity left")
spike_raster_plot_8(ear_spikes_right, plt, duration / 1000., an_pop_size + 1, 0.001, title="ear pop activity right")
# spike_raster_plot_8(output_spikes_left, plt, duration / 1000., an_pop_size + 1, 0.001, title="output pop activity left")
# spike_raster_plot_8(output_spikes_right, plt, duration / 1000., an_pop_size + 1, 0.001, title="output pop activity right")

legend_string = [str(i) for i in range(an_pop_size/10)]
plt.figure("MOC left")
for moc_signal in ear_moc_left:
    x = np.linspace(0,duration,len(moc_signal))
    plt.plot(x,moc_signal)
plt.xlabel("time (ms)")
plt.legend(legend_string)

plt.figure("MOC right")
for moc_signal in ear_moc_right:
    x = np.linspace(0,duration,len(moc_signal))
    plt.plot(x,moc_signal)
plt.xlabel("time (ms)")
plt.legend(legend_string)
plt.show()
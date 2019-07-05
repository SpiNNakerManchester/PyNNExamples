import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import spynnaker8 as sim
import sys
sys.path.append("../")
from signal_prep import *
from spinnak_ear.spinnakear import SpiNNakEar,spinnakear_size_calculator
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint
from elephant.statistics import isi,cv
#================================================================================================
# Simulation parameters
#================================================================================================
Fs = 50e3#100000.#
dBSPL=100
wav_directory = '../../OME_SpiNN/'
freq = 3000
tone_duration = 0.05
silence_duration = 0.1 #0.075#

tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_stereo = np.asarray([tone,tone_r])
timit_l = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,silence_duration=silence_duration,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=0)
[_,signal] = wavfile.read(wav_directory+'10788_edit.wav')
signal = numpy.float64(signal[:,0])
max_val=numpy.mean(signal**2)**0.5
timit_r = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,silence_duration=silence_duration,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=1,max_val=max_val)
timit = numpy.asarray([timit_l,timit_r])

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
                "timit":timit
}

test_file = "tone_{}Hz_stereo".format(freq)#"timit"#
binaural_audio = sounds_dict[test_file]#tone_stereo#np.asarray([click,click])#
duration = (binaural_audio[0].size/Fs)*1000.#max(input_spikes[0])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.)
#================================================================================================
# Populations
#================================================================================================
an_pop_size = 100
n_ears = 2
record_vars = ['spikes']
spinnakear_pops = [[] for _ in range(n_ears)]
spinnakear_objs = [[] for _ in range(n_ears)]
ear_data = [[] for _ in range(n_ears)]

for ear_index in range(n_ears):
    spinnakear_objs[ear_index]= SpiNNakEar(audio_input=binaural_audio[ear_index],fs=Fs,n_channels=an_pop_size/10,ear_index=ear_index)
    spinnakear_pops[ear_index] = sim.Population(an_pop_size,spinnakear_objs[ear_index],label="spinnakear_pop {}".format(ear_index))
    spinnakear_pops[ear_index].record(record_vars)

sim.run(duration)

for ear_index in range(n_ears):
    # ear_data[ear_index] = spinnakear_pops[ear_index].get_data(record_vars)
    # spike_raster_plot_8(np.flipud(ear_data[ear_index]['spikes']), plt, duration / 1000., an_pop_size + 1, 0.001, title="ear pop activity {}".format(ear_index))
    profile_dict=spinnakear_objs[ear_index].get_profile_data(['ihc','ome','drnl'])
    plt.figure("profile {}".format(ear_index))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    plt.subplot(gs[0])
    plt.plot(np.linspace(0,duration,binaural_audio[ear_index].size),binaural_audio[ear_index])
    plt.title("{}ms {}Hz tone burst at {}dBSPL".format(int(tone_duration*1000),freq,dBSPL))
    plt.ylabel("stimulus air pressure (Pa)")
    plt.xlabel("time (ms)")
    plt.subplot(gs[1])
    vs = ['ome', 'drnl', 'ihc']
    profile_legend=[]
    colours = ['g', 'b','r']
    for ci,v in enumerate(vs):
        a = profile_dict[v]
        profile_legend.append(plt.Line2D([0], [0], color=colours[ci], lw=2))
        for i in range(len(a)):
            plt.plot(1e3 * (a[i] / 8.),color=colours[ci])
    # plt.plot(np.ones(len(a[0])) * (1. / Fs)*1e6,color='k',linestyle='dashed')
    # profile_legend.append(plt.Line2D([0], [0], color='k', lw=2,linestyle='dashed'))
    # vs.append('real time')
    plt.legend(profile_legend,vs)
    plt.xlabel("segment index")
    plt.ylabel("execution time ($\mu$s)")
sim.end()


plt.show()
# results_directory = '/tmp/rob_test_results'
# np.savez_compressed(results_directory+'/ear_' + test_file + '_{}an_fibres_{}dB_{}s'.format
#                          (an_pop_size,dBSPL,int(duration/1000.)),ear_data=ear_data,Fs=Fs,stimulus=binaural_audio)



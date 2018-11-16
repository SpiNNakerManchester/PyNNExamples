from signal_prep import *
import numpy as np
import matplotlib.pylab as plt
from pyNN.random import NumpyRNG, RandomDistribution

input_dir = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# an_data = np.load(input_dir + "/spinnakear_13.5_1_kHz_1s_50dB_vrr.npz")
# an_data = np.load(input_dir + "/spinnakear_13.5_1_kHz_aiu_60s_50dB_1000fibres.npz")
# cn_data = np.load(input_dir + "/cn_13.5_1kHz_50dB_0.1ms_timestep_1s.npz")
cn_data = np.load(input_dir + "/chopper_13.5_1kHz_50dB_5000an_fibres_0.1ms_timestep_75s.npz")

# drnl = an_data['drnl']
#
# plt.plot(drnl[2920:2970].T)
#
# plt.figure()
# plt.plot(drnl[2860:2910].T)
# plt.figure()
# plt.plot(drnl[2980:3030].T)
# # plt.figure()
# # plt.plot(drnl[940:960].T)
#
# plt.show()

an_spikes = cn_data['an_spikes']
# ch_spikes = cn_data['ch_spikes']
# on_spikes = cn_data['on_spikes']
onset_times = cn_data['onset_times']
t_spikes = cn_data['t_spikes']


# spikes = an_spikes[959]#ch_spikes[959]
# isi = [(spikes[i] - spikes[i-1])/0.4 for i,_ in enumerate(spikes) if i>0]
# plt.figure()
# plt.hist(isi,bins=1000)
# plt.show()

# spike_raster_plot_8(an_spikes,plt,1.,len(an_spikes)+1,title="an pop activity")
# spike_raster_plot_8(on_spikes,plt,1.,len(on_spikes)+1,title="on pop activity")
spike_raster_plot_8(t_spikes,plt,75.,len(t_spikes)+1,title="ch pop activity")
# spike_raster_plot_8(octopus_spikes,plt,75.,len(octopus_spikes)+1,title="octopus pop activity")
# plt.show()
# an_psth_spikes = repeat_test_spikes_gen(an_spikes,500,onset_times)
# t_psth_spikes = repeat_test_spikes_gen(t_spikes,850,onset_times,test_duration=100.)
# psth_plot_8(plt,numpy.arange(len(t_psth_spikes[1])),t_psth_spikes[1],bin_width=0.001,duration=0.2,title="PSTH_T")

psth_index = 1
chosen_neurons =[850]
# chosen_neurons =np.random.choice(len(t_spikes),size=10,replace=False)
legend_string=[]
max_psth = 0
for neuron in chosen_neurons:
    t_psth_spikes = repeat_test_spikes_gen(t_spikes,test_neuron_id=neuron,onset_times=onset_times,test_duration=100.)
    psths=psth_plot_8(plt,numpy.arange(len(t_psth_spikes[psth_index])),t_psth_spikes[psth_index],bin_width=0.0001,
                      duration=0.1,title="PSTH_T stimulus {}".format(psth_index))
    if max(psths)>max_psth:
        max_psth = max(psths)
# psth_plot_8(plt,numpy.arange(len(an_psth_spikes[0])),an_psth_spikes[0],bin_width=0.001,duration=0.2,title="PSTH_AN")
plt.ylim((0,max_psth))
plt.legend(chosen_neurons)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.io import savemat, loadmat
from signal_prep import spike_raster_plot_8

# mat_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/MAP1_14j_2017 - Copy/userPrograms"
# mat_file = loadmat(mat_directory+"/timit_50dB_an")
# mat_an =[np.nonzero(neuron)[0]*(1000./48e3) for neuron in mat_file['ANoutput']]
# n_fibre_types = 3
# split_size = len(mat_an)/n_fibre_types
# split_spikes = [mat_an[i*split_size:i*split_size+split_size] for i in range(n_fibre_types)]
# mat_an=[val for tup in zip(*split_spikes) for val in tup]
# spike_raster_plot_8(mat_an, plt, 0.5, len(mat_an) + 1, 0.001,markersize=1)

n_ears = 2
test_n_fibres=[300,3000,30000]#
results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
for n,n_fibres in enumerate(test_n_fibres):
    results_file = np.load(results_directory+'/ear_timit_{}an_fibres_50dB_2s.npz'.format(n_fibres))
    ear_data = results_file['ear_data']
    stimulus = results_file['stimulus']
    Fs = results_file['Fs']
    # for ear_index,data in enumerate(ear_data):
    ear_index = 0
    data=ear_data[ear_index]
    plt.figure("ear {}".format(ear_index))
    duration = 0.5#stimulus[ear_index].size / Fs
    spike_raster_plot_8(np.flipud(data['spikes']), plt, duration, n_fibres + 1, 0.001,#xlim=(1,1.2),
                        title="{} fibres".format(n_fibres), markersize=1,subplots=(len(test_n_fibres),1,n+1))

# n_fibres=10000
# results_file = np.load(results_directory+'/ear_timit_{}an_fibres_50dB_2s.npz'.format(n_fibres))
# ear_data = results_file['ear_data']
# ear_labels = ['left ear','right ear']
# plt.figure()
# for ear_index,data in enumerate(ear_data):
#     duration = stimulus[ear_index].size / Fs
#     spike_raster_plot_8(np.flipud(data['spikes']), plt, duration, n_fibres + 1, 0.001,#xlim=(1,1.5),
#                         title="{}".format(ear_labels[ear_index]), markersize=1,subplots=(2,1,ear_index+1))

plt.show()
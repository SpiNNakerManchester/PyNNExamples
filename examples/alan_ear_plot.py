import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# results_file = '/ear_tone_1000Hz_stereo_1000an_fibres_0dB_0s.npz'
# results_file = '/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_50dB_0s_moc_True_lat_True.npz'
# results_file = '/ear_tone_1000Hz_stereo_100an_fibres_50dB_0s.npz'
results_file = '/ear_tone_1000Hz_stereo_112.0an_fibres_50dB_0s.npz'


sim_data = np.load(results_directory+results_file)

binaural_audio = sim_data['stimulus']
Fs = sim_data['Fs']
tone_duration = 0.1
freq=1000
dBSPL = 100
# profile_data = sim_data['profile_data']

plt.figure()
# vrr = sim_data['ear_data'][0]['debug']
vrr = np.asarray(sim_data['ear_data'][0]['prob'])
mid_point = 0#len(vrr)/2
vrr_chan = vrr[mid_point:mid_point+10]
plt.plot(vrr_chan.T)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

results_file = './ear_tone_1000Hz_stereo_112.0an_fibres_50dB_0s.npz'


sim_data = np.load(results_file, allow_pickle=True)

binaural_audio = sim_data['stimulus']
Fs = sim_data['Fs']
tone_duration = 0.1
freq=1000
dBSPL = 100

plt.figure()
# vrr = sim_data['ear_data'][0]['debug']
vrr = np.asarray(sim_data['ear_data'][0]['prob'])
mid_point = 0#len(vrr)/2
vrr_chan = vrr[mid_point:mid_point+10]
#vrr_chan = vrr[0]
x = vrr_chan.T
plt.plot(vrr_chan.T)
plt.show()

# coding=utf-8
import spynnaker8 as p
import numpy as np
import math
import unittest as unitt
from pyNN.utility.plotting import Figure, Panel, plot_spiketrains
from pyNN.random import NumpyRNG, RandomDistribution
from neo.core import Unit, Segment
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import elephant.statistics as stat
import quantities as qt
import elephant.conversion as qv
import elephant.signal_processing as esp

# ===============================
# === Define parameters =========
# ===============================

p.setup(0.1)    # 0.1ms time step
runtime = 1000  # Run time [ms]

rate = 6            # Frequency of random stimulation [Hz]
stim_dur = 700      # Duration of random stimulation [ms]

Thal_w = 0.2             # Thal-L4 synapse weight
Thal_delay = 1           # Refractory period of Thal-L4 synapse

pconn = 0.1      # connection probability
rngseed = 98766987
parallel_safe = True
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

exc_conn = p.FixedProbabilityConnector(pconn, rng=rng)
inh_conn = p.FixedProbabilityConnector(pconn, rng=rng)

Thal_n = 28

L4_exc_n = 347
L4_inh_n = 61

L23_exc_n = 450
L23_inh_n = 79

t = np.arange(runtime)

#  ==== Distributions ========

cm_dist = p.RandomDistribution(
    distribution='normal_clipped', mu=1, sigma=0.5, high=1.5, low=0.5)

uniformDistr = RandomDistribution('uniform', [-0.65, -0.55], rng=rng)

# === Neuron parameters ====

L2_3_cell_params = {
        'tau_m': 30.0,
        'cm': 0.16,
        'v_rest': -65.0,
        'v_reset': -72.0,
        'v_thresh': -40.0,
        'tau_syn_E': 5.0,
        'tau_syn_I': 15.0,
        'tau_refrac': 15,
        'i_offset': 0}

L4_cell_params = {
        'tau_m': 35.0,
        'cm': 0.12,
        'v_rest': -65.0,
        'v_reset': -66.0,
        'v_thresh': -40.0,
        'tau_syn_E': 5.0,
        'tau_syn_I': 15.0,
        'tau_refrac': 15,
        'i_offset': 0}

# === Randomise weights and delays [limits mu +/- 2*sigma] ====
# === Uniform for layers L2/3 & L4 =========
# === Excitatory ====

w_exc = 0.1
d_exc = p.RandomDistribution(
    distribution='normal_clipped', mu=1.5, sigma=0.75, high=3, low=0.1)

# === Inhibitory ========

w_inh = 0.15
d_inh = p.RandomDistribution(
    distribution='normal_clipped', mu=0.75, sigma=0.375, high=1.5, low=0.1)

# ====================================
# === Create neuron populations ======
# ====================================

# === Thalamic poisson source =====

src_Thal = p.Population(
        Thal_n, p.SpikeSourcePoisson(rate=rate, duration=stim_dur),
        label="expoisson")

# === L2/3 Excitatory and Inhibitory populations ======

L23_exc_cells = p.Population(
    L23_exc_n, p.IF_curr_exp(**L2_3_cell_params), label="L2/3 Excitatory_Cells")
L23_inh_cells = p.Population(
    L23_inh_n, p.IF_curr_exp(**L2_3_cell_params), label="L2/3 Inhibitory_Cells")

# === L4 Excitatory and Inhibitory populations ======

L4_exc_cells = p.Population(
    L4_exc_n, p.IF_curr_exp(**L4_cell_params), label="L4 Excitatory_Cells")
L4_inh_cells = p.Population(
    L4_inh_n, p.IF_curr_exp(**L4_cell_params), label="L4 Inhibitory_Cells")

# === Create synapses ========

connections = {
    'Thal-L4e' : p.Projection(
        src_Thal, L4_exc_cells, p.FixedProbabilityConnector(0.25),
        p.StaticSynapse(weight=Thal_w, delay=Thal_delay), receptor_type="excitatory"),
    'Thal-L4i' : p.Projection(
        src_Thal, L4_inh_cells, p.FixedProbabilityConnector(0.25),
        p.StaticSynapse(weight=Thal_w, delay=Thal_delay), receptor_type="excitatory"),
    # === Layer 4 intra ======
    'e-e' : p.Projection(
        L4_exc_cells, L4_exc_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=d_exc)),
    'e-i' : p.Projection(
        L4_exc_cells, L4_inh_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=d_exc)),
    'i-e' : p.Projection(
        L4_inh_cells, L4_exc_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=d_inh)),
    'i-i' : p.Projection(
        L4_inh_cells, L4_inh_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=d_inh)),
    # Layer 4-2/3 inter ========
    'L4e-L23e': p.Projection(
        L4_exc_cells, L23_exc_cells, p.FixedProbabilityConnector(0.1),
        synapse_type=p.StaticSynapse(weight=0.2, delay=d_exc),receptor_type='excitatory'),
    'L4e-L23i': p.Projection(
        L4_exc_cells, L23_inh_cells, p.FixedProbabilityConnector(0.1),
        synapse_type=p.StaticSynapse(weight=0.2, delay=d_exc), receptor_type='excitatory'),
    # === Layer 2/3 intra ======
    'e-e': p.Projection(
        L23_exc_cells, L23_exc_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=d_exc)),
    'e-i': p.Projection(
        L23_exc_cells, L23_inh_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=d_exc)),
    'i-e': p.Projection(
        L23_inh_cells, L23_exc_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=d_inh)),
    'i-i': p.Projection(
        L23_inh_cells, L23_inh_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=d_inh)),}

# === Data handling ====================
# === Record data for plots ====

src_Thal.record('spikes')

L4_exc_cells.record('spikes')
L4_inh_cells.record('spikes')

L23_exc_cells.record('spikes')
L23_inh_cells.record('spikes')

p.run(runtime)

# === Gather data for plots ===

pre_spikes_slow = src_Thal.get_data()
Thal_spikes = pre_spikes_slow.segments[0].spiketrains

L4_exc_data = L4_exc_cells.get_data()
L4_inh_data = L4_inh_cells.get_data()
L23_exc_data = L23_exc_cells.get_data()
L23_inh_data = L23_inh_cells.get_data()

# === Concatenate spiketrains
L4ei = L4_exc_data.segments[0].spiketrains + L4_inh_data.segments[0].spiketrains
L23ei = L23_exc_data.segments[0].spiketrains + L23_inh_data.segments[0].spiketrains

# === Create colourcode arrays
L4_exc_col = np.array([0,0,1]*L4_exc_n).reshape(L4_exc_n, -1)
L4_inh_col = np.array([1,0,0]*L4_inh_n).reshape(L4_inh_n, -1)

L23_exc_col = np.array([0,0,1]*L23_exc_n).reshape(L23_exc_n, -1)
L23_inh_col = np.array([1,0,0]*L23_inh_n).reshape(L23_inh_n, -1)

L4_colorCodes = np.append(L4_exc_col, L4_inh_col, axis=0)
L23_colorCodes = np.append(L23_exc_col, L23_inh_col, axis=0)

# === Bin concatenated spiketrains
binsize = 1 * qt.ms
Thal_binned_spikes = stat.time_histogram(Thal_spikes, binsize=binsize, t_start=0 * qt.ms, t_stop=runtime * qt.ms)
L4_binned_spikes = stat.time_histogram(L4ei, binsize=binsize, t_start=0 * qt.ms, t_stop=runtime * qt.ms)
L23_binned_spikes = stat.time_histogram(L23ei, binsize=binsize, t_start=0 * qt.ms, t_stop=runtime * qt.ms)

# === Bin individual spiketrains for heatmap
h_binsize = 50 * qt.ms
L4i_binned_heatmap = qv.BinnedSpikeTrain(L4_inh_data.segments[0].spiketrains, binsize=h_binsize).to_array()
L4e_binned_heatmap = qv.BinnedSpikeTrain(L4_exc_data.segments[0].spiketrains, binsize=h_binsize).to_array()
L23i_binned_heatmap = qv.BinnedSpikeTrain(L23_exc_data.segments[0].spiketrains, binsize=h_binsize).to_array()
L23e_binned_heatmap = qv.BinnedSpikeTrain(L23_inh_data.segments[0].spiketrains, binsize=h_binsize).to_array()

# == Assign Neo binned data to ndarray
L4fft_data = np.array(L4_binned_spikes)
L4fft_data = L4fft_data.reshape(1000,)
L23fft_data = np.array(L23_binned_spikes)
L23fft_data = L23fft_data.reshape(1000,)

# === Fourier transform [James]

L4fft = np.abs(np.fft.fft(L4fft_data))**2
L23fft = np.abs(np.fft.fft(L23fft_data))**2

# === Fourier transform [Oliver]
L23FT = np.fft.fft(L23fft_data)/(runtime/2) # Compute and normalise FT
L23FT = L23FT[range(runtime/2)] # FT is symmetric so only need half

# Extract frequencies to print
freqs = np.fft.fftfreq(runtime)


# === Create plots =========

# === Raster plot
plt.figure(1, figsize=(12,8))
plt.subplot(311)
plt.eventplot(pre_spikes_slow.segments[0].spiketrains, colors=[0,0,0])
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Neuron id')
plt.title('Thalamus')
plt.subplot(312)
plt.eventplot(L4ei, colors=L4_colorCodes)
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Neuron id')
plt.title('Layer 4 e/i')
plt.subplot(313)
plt.eventplot(L23ei, colors=L23_colorCodes)
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Neuron id')
plt.title('Layer 2/3 e/i')

# === Plot binned spiketrains
plt.figure(2, figsize=(12,8))
plt.subplot(311)
plt.plot(Thal_binned_spikes)
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Firing rate [sp./cell/sec.]')
plt.title('Thalamic spike rate')
plt.minorticks_on()
plt.subplot(312)
plt.plot(L4_binned_spikes)
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Firing rate [sp./cell/sec.]')
plt.title('Layer 4 spike rate')
plt.minorticks_on()
plt.subplot(313)
plt.plot(L23_binned_spikes)
plt.xlim(0,runtime)
plt.xlabel('Time [s]')
plt.ylabel('Firing rate [sp./cell/sec.]')
plt.title('Layer 2/3 spike rate')
plt.minorticks_on()

# === Plot heatmaps
plt.figure(3, figsize=(8,12))
# plt.suptitle('Column heatmap')
plt.subplot(411)
plt.imshow(L4i_binned_heatmap, cmap='jet', interpolation='none', aspect='auto')
plt.title('Layer 4 inhibitory')
plt.subplot(412)
plt.imshow(L4e_binned_heatmap, cmap='jet', interpolation='none', aspect='auto')
plt.title('Layer 4 excitatory')
plt.subplot(413)
plt.imshow(L23i_binned_heatmap, cmap='jet', interpolation='none', aspect='auto')
plt.title('Layer 2/3 inhibitory')
plt.subplot(414)
plt.imshow(L23e_binned_heatmap, cmap='jet', interpolation='none', aspect='auto')
plt.title('Layer 2/3 excitatory')
plt.tight_layout()

# Plot FFT [James]
plt.figure(4, figsize=(12,8))
plt.subplot(211)
plt.plot(L4fft)
plt.minorticks_on()
plt.xlim(0, (runtime/2))
plt.xlabel('Harmonic frequency [Hz]')
# plt.suptitle('Power spectrum')
plt.subplot(212)
plt.plot(L23fft)
plt.minorticks_on()
plt.xlim(0, (runtime/2))
plt.xlabel('Harmonic frequency [Hz]')
# plt.ylim(0, 2000)

# Plot FFT [Oliver]
plt.figure(5, figsize=(12,8))
fig, ax = plt.subplots(2,1)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].plot(t,L23fft_data)
ax[1].plot(t[range(runtime/2)],abs(L23FT[range(runtime/2)])) # plot half as symmetric
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('|FT|')
plt.subplots_adjust(hspace=0.4)

plt.show()

# print abs(FT)
print 'FT frequencies'
for f in range(len(L23FT)):
    if (abs(L23FT[f]) > 0.001):
        print freqs[f]*runtime

# === Clean-up and end simulation =====

p.end()

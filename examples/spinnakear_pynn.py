import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pyNN.utility.plotting import Figure, Panel
from spinnakear import SpiNNakEar
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint

#================================================================================================
# Simulation parameters
#================================================================================================
moc_spikes = [[10.,15.,20.,100.,105.]]#,102,104]]
Fs = 100000.#22050.
dBSPL=20
tone_1 = generate_signal(freq=1000,dBSPL=dBSPL,duration=0.05,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
binaural_audio = np.asarray([np.tile(tone_1,5)])
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
an_pop_size = 3000
spinnakear_pop = sim.Population(an_pop_size,SpiNNakEar(audio_input=binaural_audio,fs=Fs,n_channels=an_pop_size/10),label="spinnakear_pop")
# spinnakear_pop.size = an_pop_size TODO:set spinnakear pop size to 60 (not 39 (n_atoms))
moc_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=moc_spikes),label="moc_pop")
an_pop = sim.Population(an_pop_size,sim.IF_cond_exp,{},label="an_pop")
an_pop.record('spikes')
#================================================================================================
# Projections
#================================================================================================

moc_ohc_connectons = [(0,0),(0,1)]
# moc_projection = sim.Projection(moc_pop,spinnakear_pop,sim.FromListConnector(moc_ohc_connectons),synapse_type=sim.StaticSynapse(weight=1.))
spinnakear_an_connections = []
for id in range(an_pop_size):
    spinnakear_an_connections.append((id,id))
an_projection = sim.Projection(spinnakear_pop,an_pop,sim.FromListConnector(spinnakear_an_connections),synapse_type=sim.StaticSynapse(weight=0.1))

duration = (binaural_audio.size/Fs)*1000.#max(input_spikes[0])

sim.run(duration)

target_data = an_pop.get_data(['spikes'])
output_spikes = target_data.segments[0].spiketrains

sim.end()

spike_raster_plot_8(output_spikes, plt, duration / 1000., an_pop_size + 1, 0.001, title="output pop activity")

plt.show()

# Figure(
#     # plot data for postsynaptic neuron
#     Panel(cd_data.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",legend=False,
#           yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",legend=False,
#            yticks=True,xticks=True, xlim=(0, duration)),
#     # Panel(cd_data.segments[0].filter(name='gsyn_inh')[0],
#     #       ylabel="gsyn inhibitory (mV)",legend=False,
#     #        yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].spiketrains,marker='.',
#           yticks=True,markersize=3,
#                  markerfacecolor='black', markeredgecolor='none',
#                  markeredgewidth=0,xticks=True, xlim=(0, duration)),
#     Panel(input_data.segments[0].spiketrains, marker='.',
#           yticks=True, markersize=3,
#           markerfacecolor='black', markeredgecolor='none',
#           markeredgewidth=0, xticks=True, xlim=(0, duration))
# )

# plt.show()
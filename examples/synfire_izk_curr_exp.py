"""
Synfirechain-like example
"""
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 500
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
nNeurons = 200  # number of neurons in each population
p.set_number_of_neurons_per_core(p.Izhikevich, nNeurons / 2)

cell_params_izk = {'a': 0.02,
                   'b': 0.2,
                   'c': -65,
                   'd': 8,
                   'v_init': -75,
                   'u_init': 0,
                   'tau_syn_E': 2,
                   'tau_syn_I': 2,
                   'i_offset': 0
                   }

populations = list()
projections = list()

weight_to_spike = 30
delay = 1

loopConnections = list()
for i in range(0, nNeurons):
    singleConnection = ((i, (i + 1) % nNeurons, weight_to_spike, delay))
    loopConnections.append(singleConnection)

injectionConnection = [(0, 0)]
spikeArray = {'spike_times': [[50]]}
populations.append(
    p.Population(nNeurons, p.Izhikevich(**cell_params_izk), label='pop_1'))
populations.append(
    p.Population(1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1'))

projections.append(p.Projection(
    populations[0], populations[0], p.FromListConnector(loopConnections),
    p.StaticSynapse(weight=weight_to_spike, delay=delay)))
projections.append(p.Projection(
    populations[1], populations[0], p.FromListConnector(injectionConnection),
    p.StaticSynapse(weight=weight_to_spike, delay=1)))

populations[0].record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])

p.run(runtime)

# get data (could be done as one, but can be done bit by bit as well)
data = populations[0].get_data(['v', 'gsyn_exc', 'spikes', 'gsyn_inh'])

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[populations[0].label], yticks=True, xlim=(0, runtime)),
    Panel(data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[populations[0].label], yticks=True, xlim=(0, runtime)),
    Panel(data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[populations[0].label], yticks=True, xlim=(0, runtime)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()

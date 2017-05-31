"""
Synfirechain-like example
"""
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

from pyNN.random import RandomDistribution

p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
nNeurons = 200  # number of neurons in each population
max_delay = 50
run_time = max_delay * nNeurons
p.set_number_of_neurons_per_core(p.IF_curr_exp, nNeurons / 2)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

weight_to_spike = 2.0
delay = RandomDistribution("uniform", low=1, high=max_delay)

loopConnections = list()
for i in range(0, nNeurons):
    delay_value = delay.next()
    singleConnection = (i, ((i + 1) % nNeurons), weight_to_spike, delay_value)
    loopConnections.append(singleConnection)

injectionConnection = [(0, 0, weight_to_spike, 1)]
spikeArray = {'spike_times': [[0]]}
main_pop = p.Population(
    nNeurons, p.IF_curr_exp(**cell_params_lif), label='pop_1')
input_pop = p.Population(
    1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1')

p.Projection(main_pop, main_pop, p.FromListConnector(loopConnections))
p.Projection(input_pop, main_pop, p.FromListConnector(injectionConnection))

main_pop.record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])

print "Running for {} ms".format(run_time)
p.run(run_time)
# get data (could be done as one, but can be done bit by bit as well)
data = main_pop.get_data(['v', 'gsyn_exc', 'spikes', 'gsyn_inh'])

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    # membrane potential of the postsynaptic neuron
    Panel(data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, run_time)),
    Panel(data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, run_time)),
    Panel(data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, run_time)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()

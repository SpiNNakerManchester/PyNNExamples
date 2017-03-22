"""
Synfirechain-like example
"""
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

run_time = 6000
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
nNeurons = 10  # number of neurons in each population

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

weight_to_spike = 10.0
delay = 2
second_spike_start = delay * nNeurons
space_between_inputs = delay * nNeurons * 2

connections = list()
reverseConnections = list()
for i in range(0, nNeurons - 1):
    connections.append((i, (i + 1) % nNeurons, weight_to_spike, delay))
    reverseConnections.append(((i + 1) % nNeurons, i, weight_to_spike, delay))

injectionConnection_1 = [(0, 0, weight_to_spike, 1)]
injectionConnection_2 = [(1, nNeurons - 1, weight_to_spike, 1)]
input_1 = [i for i in xrange(0, run_time, space_between_inputs)]
input_2 = [i for i in xrange(second_spike_start, run_time,
                             space_between_inputs)]
spikeArray = {'spike_times': [input_1, input_2],
              'max_on_chip_memory_usage_for_spikes_in_bytes': 640,
              'space_before_notification': 320}
main_pop = p.Population(
    nNeurons, p.IF_curr_exp(**cell_params_lif), label='pop_1')
second_main_pop = p.Population(
    nNeurons, p.IF_curr_exp(**cell_params_lif), label='pop_2')
input_pop = p.Population(
    2, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1')

p.Projection(
    main_pop, main_pop, p.FromListConnector(connections),
    p.StaticSynapse(weight=weight_to_spike, delay=delay))
p.Projection(
    second_main_pop, second_main_pop, p.FromListConnector(reverseConnections),
    p.StaticSynapse(weight=weight_to_spike, delay=delay))

p.Projection(
    input_pop, main_pop, p.FromListConnector(injectionConnection_1),
    p.StaticSynapse(weight=weight_to_spike, delay=1))

p.Projection(
    input_pop, second_main_pop, p.FromListConnector(injectionConnection_2),
    p.StaticSynapse(weight=weight_to_spike, delay=1))

main_pop.record("spikes")
second_main_pop.record("spikes")

p.run(run_time)

# get data (could be done as one, but can be done bit by bit as well)
spikes1 = main_pop.get_data('spikes')
spikes2 = second_main_pop.get_data("spikes")

Figure(
    # raster plot of the pre_synaptic neuron spike times
    Panel(spikes1.segments[0].spiketrains,
          ylabel="spikes from first pop",
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    # membrane potential of the post_synaptic neuron
    Panel(spikes2.segments[0].spiketrains,
          ylabel="spikes from second pop",
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    title="large data Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()

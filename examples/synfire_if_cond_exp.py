"""
Synfirechain-like example
"""
import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 3500
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
nNeurons = 832  # number of neurons in each population
#p.set_number_of_neurons_per_core(p.IF_curr_exp, 33)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0,
                   'e_rev_E': 0.,
                   'e_rev_I': -80.
                   }

weight_to_spike = 0.035
delay = 2

loopConnections = list()
for i in range(0, nNeurons):
    singleConnection = ((i, (i + 1) % nNeurons, weight_to_spike, delay))
    loopConnections.append(singleConnection)

injectionConnection = [(0, 0)]
spikeArray = {'spike_times': [[2]]}
main_pop = p.Population(
    nNeurons, p.IF_cond_exp(**cell_params_lif), label='pop_1', in_partitions=6, out_partitions=6)
input_pop = p.Population(
    1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1')

p.Projection(
    main_pop, main_pop, p.FromListConnector(loopConnections),
    p.StaticSynapse(weight=weight_to_spike, delay=delay), receptor_type="excitatory")
p.Projection(
    input_pop, main_pop, p.FromListConnector(injectionConnection),
    p.StaticSynapse(weight=weight_to_spike, delay=1), receptor_type="excitatory")

main_pop.record(['spikes', 'gsyn_exc', 'gsyn_inh', 'v'])

p.run(runtime)

#time.sleep(3000)

# get data (could be done as one, but can be done bit by bit as well)
v = main_pop.get_data('v')
gsyn_exc = main_pop.get_data('gsyn_exc')
gsyn_inh = main_pop.get_data('gsyn_inh')
spikes = main_pop.get_data('spikes')

figure_filename = "results.png"
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, runtime)),
    Panel(gsyn_exc.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, runtime)),
    Panel(gsyn_inh.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[main_pop.label], yticks=True, xlim=(0, runtime)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)

for n in range(len(spikes.segments[0].spiketrains)):
    print "Neuron: " + str(n) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[n])


#ind = 0
#for elem in v.segments[0].filter(name='v')[0]:
#    print str(ind) + "\n\n\n\n" + str(elem)
#    ind += 1

plt.show()
print(figure_filename)

p.end()

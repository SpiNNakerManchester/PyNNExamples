import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 100
nNeurons = 65
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, nNeurons / 2)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0,
                   "v" : -40
                   }

weight_to_spike = 0.035

population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1')
input = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='input')

p.Projection(input, population, p.FromListConnector([(0, 0), (1, 0), (64, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=2))

population.record(['v', 'spikes'])

p.run(runtime)

v = population.get_data('v')
spikes = population.get_data('spikes')

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[population.label], yticks=True, xlim=(0, runtime)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)

plt.show()

p.end()
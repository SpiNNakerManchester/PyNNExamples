import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 5000
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
nNeurons = 1  # number of neurons in each population
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
                   }

weight_to_spike = 0.035
delay = 17

spikeArray = {'spike_times': [[0, 1000, 1500]]}

pop1 = p.Population(2, p.IF_curr_exp(**cell_params_lif), label="pop1")
input = p.Population(2, p.SpikeSourceArray(**spikeArray), label='input1')

p.Projection(input, pop1, p.FromListConnector([(0, 0, 0.035, 17)]), p.StaticSynapse(weight=weight_to_spike, delay=delay))

pop1.record(['v', 'spikes'])

p.run(runtime)

v = pop1.get_data('v')
#gsyn_exc = main_pop.get_data('gsyn_exc')
#gsyn_inh = main_pop.get_data('gsyn_inh')
spikes = pop1.get_data('spikes')

figure_filename = "results.png"
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop1.label], yticks=True, xlim=(0, runtime)),
    #Panel(gsyn_exc.segments[0].filter(name='gsyn_exc')[0],
    #      ylabel="gsyn excitatory (mV)",
    #      data_labels=[main_pop.label], yticks=True, xlim=(0, runtime)),
    #Panel(gsyn_inh.segments[0].filter(name='gsyn_inh')[0],
    #      ylabel="gsyn inhibitory (mV)",
    #      data_labels=[main_pop.label], yticks=True, xlim=(0, runtime)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()
print(figure_filename)

p.end()
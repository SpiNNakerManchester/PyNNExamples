import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 1000
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0.8,
    "v_rest": 0
    }


input_pop = pynn.Population(1,
                            pynn.SpikeSourceArray,
                            {'spike_times': [200]},
                            label='input_pop')

neuron = pynn.Population(1,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

in_proj = pynn.Projection(input_pop,
                          neuron,
                          pynn.OneToOneConnector(),
                          pynn.StaticSynapse(weight=[-0.5], delay=[0]),
                          receptor_type='input_connections')


input_pop.record('spikes')
neuron.record('all')

pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
neuron_res = neuron.get_data('all')

Figure(
    Panel(in_spikes.segments[0].spiketrains,
          ylabel="Input Spikes",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(neuron_res.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(neuron_res.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
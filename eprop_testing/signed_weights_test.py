import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 100
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0
    }


spike_source = pynn.Population(2,
                               pynn.SpikeSourceArray,
                               {'spike_times': [10]},
                               label='Spike Source')

neuron = pynn.Population(2,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

proj = pynn.Projection(spike_source, neuron,
                       pynn.OneToOneConnector(),
                       pynn.StaticSynapse(weight=[-0.5, 2] , delay=[1, 5]),
                       label='exc_proj')


neuron.record('all')

pynn.run(runtime)

res = neuron.get_data('all')
weights = proj.get('weight', 'list', with_address=False)
delays = proj.get('delay', 'list', with_address=False)

for i in range(len(weights)):
    print(weights[i], delays[i])

Figure(
    Panel(res.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 100000
dt = 1

pynn.setup(dt)

n_neurons = 16

neuron_params = {
    "v": 0,
    "target_rate": 10
    }



neuron = pynn.Population(n_neurons, 
                         pynn.extra_models.EPropAdaptive(**neuron_params), 
                         label='eprop_pop')

poisson_src = pynn.Population(n_neurons, 
                              pynn.SpikeSourcePoisson(rate=
                                                      10
#                                                       [15.0, 5.0]
                                                      ), 
                              label='Poisson Src')

proj = pynn.Projection(
    poisson_src,
    neuron,
    pynn.OneToOneConnector(),
    pynn.StaticSynapse(weight=5.0, delay=dt), 
    # weight set to cause postsynaptic neuron to fire
    receptor_type='excitatory'
    )

poisson_src.record('spikes')
neuron.record('all')

pynn.run(runtime)

res = neuron.get_data('all')
poisson_spikes = poisson_src.get_data('spikes')

Figure(
    Panel(poisson_spikes.segments[0].spiketrains,
          ylabel="Input Spikes",
          data_labels=poisson_src.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="Global Rate Approx (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=poisson_src.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
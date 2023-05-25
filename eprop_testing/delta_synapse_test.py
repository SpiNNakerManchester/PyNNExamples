import pyNN.spiNNaker as pynn
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


inp_spike_source = pynn.Population(2,
                               pynn.SpikeSourceArray,
                               {'spike_times': [10]},
                               label='Spike Source')

pseudo_rec_spike_source = pynn.Population(2,
                               pynn.SpikeSourceArray,
                               {'spike_times': [80]},
                               label='Spike Source')

neuron = pynn.Population(2,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

inp_proj = pynn.Projection(inp_spike_source, neuron,
                       pynn.OneToOneConnector(),
                       # note that delays are now fixed to one in terms of spikes,
                       # but the synaptic word field indexes the synapse array
                       pynn.StaticSynapse(weight=[-0.5, 10] , delay=[1, 1]),
                       receptor_type='input_connections')


rec_proj = pynn.Projection(pseudo_rec_spike_source, neuron,
                       pynn.OneToOneConnector(),
                       # note that delays are now fixed to one in terms of spikes,
                       # but the synaptic word field indexes the synapse array
                       pynn.StaticSynapse(weight=[0.5, -10] , delay=[1, 1]),
                       receptor_type='recurrent_connections')



neuron.record('all')

pynn.run(runtime)

res = neuron.get_data('all')
weights = inp_proj.get('weight', 'list', with_address=False)
delays = inp_proj.get('delay', 'list', with_address=False)

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
    Panel(res.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
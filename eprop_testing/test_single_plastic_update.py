import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 2000
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0,
    "v_rest": 0
    }


spike_source = pynn.Population(2,
                               pynn.SpikeSourceArray,
                               {'spike_times': [1025]},
                               label='Spike Source')

neuron = pynn.Population(2,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')


start_w = [-0.5, 2]
eprop_learning = pynn.STDPMechanism(
    timing_dependence=pynn.extra_models.TimingDependenceEprop(),
    weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
        w_min=-2.0, w_max=2.0, reg_rate=1.5), 
    weight=start_w, delay=[0, 0])

proj = pynn.Projection(spike_source, neuron,
                       pynn.OneToOneConnector(),
                       synapse_type=eprop_learning,
                       label='input_connections',
                       receptor_type='input_connections')

neuron.record('all')

pynn.run(runtime)

res = neuron.get_data('all')

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
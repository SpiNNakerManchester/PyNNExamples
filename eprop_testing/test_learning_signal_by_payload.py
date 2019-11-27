import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 1024
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0.8,
    "v_rest": 0
    }

target_data = []
for i in range(1024):
            target_data.append(
                5 + 2 * np.sin(2 * i * 2* np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2* np.pi / 1024))
                )

readout_neuron_params = {
    "v": 0,
    "v_thresh": 30, # controls firing rate of error neurons
    "target_data": target_data,
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
                          pynn.StaticSynapse(weight=[-0.5], delay=[174]),
                          receptor_type='excitatory')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.SinusoidReadout(
                            **readout_neuron_params
                           ), 
                       label="readout_pop" 
                       )

input_pop.record('spikes')
neuron.record('all')
readout_pop.record('all')

pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
neuron_res = neuron.get_data('all')
readout_res = readout_pop.get_data('all')

Figure(
    Panel(in_spikes.segments[0].spiketrains,
          ylabel="Input Spikes",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
#     Panel(neuron_res.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",
#           data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
#     Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",
#           data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
#     Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0],
#           xlabel="Time (ms)", xticks=True,
#           ylabel="gsyn inhibitory (mV)",
    Panel(readout_res.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=readout_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=readout_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=readout_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(readout_res.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=readout_pop.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
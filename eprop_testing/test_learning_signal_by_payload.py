import pyNN.spiNNaker as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 1024
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 0.8,
    "v_rest": 0,
    "w_fb": 0.75
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
                            {'spike_times': [200, 210]},
                            label='input_pop')

neuron = pynn.Population(1,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

in_proj = pynn.Projection(input_pop,
                          neuron,
                          pynn.OneToOneConnector(),
                          pynn.StaticSynapse(weight=[-0.5], delay=[0]),
                          receptor_type='input_connections')

# Output population
readout_pop = pynn.Population(3, # HARDCODED 1
                       pynn.extra_models.SinusoidReadout(
                            **readout_neuron_params
                           ),
                       label="readout_pop"
                       )

learning_proj = pynn.Projection(readout_pop,
                          neuron,
                          pynn.OneToOneConnector(),
                          pynn.StaticSynapse(weight=[-0.5], delay=[0]),
                          receptor_type='learning_signal')

input_pop.record('spikes')
neuron.record('all')
readout_pop.record('all')

pynn.run(runtime)
in_spikes = input_pop.get_data('spikes')
neuron_res = neuron.get_data('all')
readout_res = readout_pop.get_data('all')



# Plot rec neuron output
plt.figure()
plt.tight_layout()

plt.subplot(4, 1, 1)
plt.plot(neuron_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')

plt.subplot(4, 1, 2)
plt.plot(neuron_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')

plt.subplot(4, 1, 3)
plt.plot(neuron_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')

plt.subplot(4,1,4)
plt.plot(in_spikes.segments[0].spiketrains, label='in_spikes')

# Plot Readout output
plt.figure()
plt.tight_layout()

plt.subplot(3, 1, 1)
plt.plot(readout_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')

plt.subplot(3, 1, 2)
plt.plot(readout_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')

plt.subplot(3, 1, 3)
plt.plot(readout_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')


plt.show()

pynn.end()
print("job done")
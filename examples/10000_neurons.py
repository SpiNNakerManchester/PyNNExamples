import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 100
nNeurons = 10000
p.setup(timestep=0.1)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 64)

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


population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1')
input = p.Population(1, p.SpikeSourceArray(spike_times=[1, 8, 16, 50]), label='input')


p.Projection(input, population, p.FromListConnector([(0, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=1))

for i in range(200, 270):
    p.Projection(input, population, p.FromListConnector([(0, i)]), p.StaticSynapse(weight=weight_to_spike, delay=1))

p.Projection(input, population, p.FromListConnector([(0, 5000)]), p.StaticSynapse(weight=weight_to_spike, delay=1))
p.Projection(input, population, p.FromListConnector([(0, 8000)]), p.StaticSynapse(weight=weight_to_spike, delay=1))

population.record(['v', 'spikes'])

p.run(runtime)

v = population.get_data('v')
spikes = population.get_data('spikes')

for n in range(len(spikes.segments[0].spiketrains)):
    if len(spikes.segments[0].spiketrains[n]) > 0:
        print "Neuron: " + str(n) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[n])


p.end()
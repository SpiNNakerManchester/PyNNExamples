import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def self_connection():

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
                       }

    weight_to_spike = 0.035

    population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1', in_partitions=2, out_partitions=2)
    input = p.Population(2, p.SpikeSourceArray(spike_times=[0, 8, 16, 50]), label='input')

    p.Projection(input, population, p.FromListConnector([(0, 64)]), p.StaticSynapse(weight=weight_to_spike, delay=2))
    p.Projection(population, population, p.FromListConnector([(64, 1)]), p.StaticSynapse(weight=weight_to_spike, delay=2))

    population.record(['v', 'spikes'])

    p.run(runtime)

    v = population.get_data('v')
    spikes = population.get_data('spikes')

    p.end()

    if str(spikes.segments[0].spiketrains[64]) == "[ 4. 11. 18. 53.] ms" and \
        str(spikes.segments[0].spiketrains[1]) == "[ 8. 14. 20. 30. 57.] ms":
        return True
    else:
        print "Neuron: " + str(1) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[1])
        print "Neuron: " + str(64) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[64])
        return False


if __name__ == "__main__":
    if self_connection():
        print "PASSED!!!"
    else:
        print "FAILED"
import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def single_neuron():

    runtime = 100
    nNeurons = 65
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

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

    population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1')
    input = p.Population(1, p.SpikeSourceArray(spike_times=[0, 8, 16, 50]), label='input')

    p.Projection(input, population, p.FromListConnector([(0, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=2))

    population.record(['v', 'spikes'])

    p.run(runtime)

    v = population.get_data('v')
    spikes = population.get_data('spikes')

    p.end()

    if str(spikes.segments[0].spiketrains[0]) == "[ 4. 11. 18. 53.] ms":
        return True
    else:
        return False


    #plt.show()

if __name__ == "__main__":
    if single_neuron():
        print "PASSED!!!"
    else:
        print "FAILED"
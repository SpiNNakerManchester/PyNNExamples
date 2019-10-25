import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def single_neuron_microsec():

    runtime = 200
    nNeurons = 2
    p.setup(timestep=1)
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
    delay = 3

    population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1')
    input = p.Population(nNeurons, p.SpikeSourceArray(spike_times=[8, 16, 50]), label='input')

    #input = p.Population(250, p.SpikeSourcePoisson(
    #    rate=50, duration=5000), label='poisson_source')

    l = []
    l1 = []
    for i in range(100):
        l.append((i, i))
        if (100 - i > 0):
            l1.append((100-i, i))

    # Generate random distributions from which to initialise parameters
    rng = p.NumpyRNG(seed=98766987, parallel_safe=True)

    # Distribution from which to allocate delays
    #delay_distribution = p.RandomDistribution('uniform', [1, 10], rng=rng)

    p.Projection(input, population, p.FixedProbabilityConnector(p_connect=0.5), p.StaticSynapse(weight=weight_to_spike, delay=delay), receptor_type='excitatory')
    #p.Projection(input, population, p.FromListConnector([(0, 0), (0, 1), (1, 1)]), p.StaticSynapse(weight=weight_to_spike, delay=delay), receptor_type='excitatory')
    #p.Projection(population, population, p.FromListConnector(l1), p.StaticSynapse(weight=weight_to_spike, delay=delay), receptor_type='excitatory')

    # Poisson source projections
    #poisson_projection_exc = p.Projection(input, population,
    #    p.FixedProbabilityConnector(p_connect=0.2),
    #    synapse_type=p.StaticSynapse(weight=0.06, delay=delay),
    #    receptor_type='excitatory')

    population.record(['v', 'spikes'])

    p.run(runtime)

    v = population.get_data('v')
    spikes = population.get_data('spikes')

    p.end()

    if str(spikes.segments[0].spiketrains[0]) == "[11. 16. 19. 24. 53. 58.] ms":
        return True
    else:
        return False

if __name__ == "__main__":
    if single_neuron_microsec() is True:
        print "PASSED!"
    else:
        print "FAILED"

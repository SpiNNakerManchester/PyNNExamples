import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def single_neuron_double_spikes():
    runtime = 10
    nNeurons = 70
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
    #p.set_number_of_neurons_per_core(p.IF_curr_exp, nNeurons / 2)

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

    # weight_to_spike = 0.035
    weight_to_spike = 1
    delay = 3

    population = p.Population(nNeurons, p.IF_curr_exp(**cell_params_lif), label='population')
    input = p.Population(1, p.SpikeSourceArray(spike_times=[1, 5, 7]), label='input1')

    p.Projection(input, population, p.FixedProbabilityConnector(p_connect=0.99), p.StaticSynapse(weight=weight_to_spike, delay=2), receptor_type="excitatory")
    # p.Projection(input, population, p.FromListConnector([(0, 0, 0.1, 1)]), p.StaticSynapse(weight=weight_to_spike, delay=2), receptor_type="inhibitory")
    p.Projection(input, population, p.FixedProbabilityConnector(p_connect=0.99), p.StaticSynapse(weight=weight_to_spike, delay=4), receptor_type="inhibitory")

    population.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    ge = population.get_data('gsyn_exc')
    gi = population.get_data('gsyn_inh')

    t = 0

    # Golden values used for test purposes. DON'T CHANGE THE DELAYS AND WEIGHTS!
    golden_exc = ["0.0 uS", "0.0 uS", "0.0 uS", "0.906341552734375 uS", "0.742034912109375 uS",
                  "0.607513427734375 uS", "0.49737548828125 uS", "1.31353759765625 uS",
                  "1.075408935546875 uS", "1.78680419921875 uS"]
    golden_inh = ["0.0 uS", "0.0 uS", "0.0 uS", "0.0 uS", "0.0 uS", "0.906341552734375 uS",
                  "0.742034912109375 uS", "0.607513427734375 uS", "0.49737548828125 uS", "1.31353759765625 uS"]

    for i in range(len(ge.segments[0].filter(name='gsyn_exc')[0])):
        for j in range(len(ge.segments[0].filter(name='gsyn_exc')[0][i])):
            a = str(ge.segments[0].filter(name='gsyn_exc')[0][i][j])
            b = str(gi.segments[0].filter(name='gsyn_inh')[0][i][j])
            if golden_exc[t] != a or \
               golden_inh[t] != b:
                print a
                print b
                print golden_exc[t]
                print golden_inh[t]
                return False
        t += 1

    p.end()

    return True


if __name__ == "__main__":
    if single_neuron_double_spikes() is True:
        print "PASSED"
    else:
        print "FAILED"

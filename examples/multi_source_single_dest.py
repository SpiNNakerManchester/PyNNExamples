import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def multi_source_single_dest():

    runtime = 100
    p.setup(timestep=0.1)
    nNeurons = 65  # number of neurons in each population

    cell_params_lif = {'cm': 0.25,
                       'i_offset': 0.0,
                       'tau_m': 20.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50.0
                       }

    input = list()
    projections = list()

    weight_to_spike = 2.0


    for i in range(10):
        input.append(p.Population(1, p.SpikeSourceArray(spike_times=[0, 8, 16, 50]), label='input'))

    population = p.Population(nNeurons, p.IF_curr_exp(**cell_params_lif), label='population')

    for i in range(10):
        projections.append(p.Projection(input[i], population, p.FromListConnector([(0, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=4)))

    population.record(["spikes"])

    p.run(runtime)

    spikes = population.get_data('spikes')

    p.end()

    if str(spikes.segments[0].spiketrains[0]) == "[ 4.5  7.8 12.  14.9 18.7 21.4 24.6 29.2 54.2 57.4 61.9] ms":
        return True
    return False

if __name__ == "__main__":
    if multi_source_single_dest():
        print "PASSED!!!"
    else:
        print "FAILED"

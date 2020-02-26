import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def multi_layer():

    runtime = 100
    nNeurons = 80
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

    source = p.Population(1, p.SpikeSourceArray(spike_times=[0, 1, 7]), label='source_array')
    population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population', in_partitions=2, out_partitions=2)
    input = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='input', in_partitions=2, out_partitions=2)

    p.Projection(source, input, p.FromListConnector([(0, 0), (0, 1), (0, 64), (0, 65), (0, 70)]), p.StaticSynapse(weight=weight_to_spike, delay=2))
    p.Projection(input, population, p.FromListConnector([(0, 2), (1, 3), (64, 4), (65, 5), (70, 6), (64, 71), (70, 73)]), p.StaticSynapse(weight=weight_to_spike, delay=2))

    population.record(['v', 'spikes'])
    input.record(['spikes'])

    p.run(runtime)

    v = population.get_data('v')
    mid = input.get_data('spikes')
    spikes = population.get_data('spikes')

    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime)),
        # membrane potential of the postsynaptic neuron
        Panel(v.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        title="Simple synfire chain example",
        annotations="Simulated with {}".format(p.name())
    )

    #plt.show()

    midlist = [0, 1, 64, 65, 70]
    endlist = [2, 3, 4, 5, 6, 71, 73]

    success = True
    for i in midlist:
        print "Neuron " + str(i) + " spiked at " + str(mid.segments[0].spiketrains[i])
        if str(mid.segments[0].spiketrains[i]) != "[ 3.  9. 16.] ms" and success is True:
            success = False

    for i in endlist:
        print "Neuron " + str(i) + " spiked at " + str(spikes.segments[0].spiketrains[i])
        if str(spikes.segments[0].spiketrains[i]) != "[ 7. 12. 18. 28.] ms" and success is True:
            success = False

    p.end()

    return success


if __name__ == "__main__":
    if multi_layer() == True:
        print "PASSED"
    else:
        print "FAILED"
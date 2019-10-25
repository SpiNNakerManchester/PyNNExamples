import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def single_neuron_del_ext():
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
    delay = 19

    population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='population_1')
    input = p.Population(1, p.SpikeSourceArray(spike_times=[1, 8, 16, 50]), label='input')

    p.Projection(input, population, p.FromListConnector([(0, 0), (0, 66)]), p.StaticSynapse(weight=weight_to_spike, delay=delay))

    population.record(['v', 'spikes'])

    p.run(runtime)

    v = population.get_data('v')
    spikes = population.get_data('spikes')

    figure_filename = "results.png"
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
        # membrane potential of the postsynaptic neuron
        Panel(v.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime), xticks=True),
        title="Single neuron",
        annotations="Simulated with {}".format(p.name())
    )

    for n in range(len(spikes.segments[0].spiketrains)):
        print "Neuron: " + str(n) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[n])

    plt.show()
    print(figure_filename)

    p.end()

    return True


if __name__ == "__main__":
    if single_neuron_del_ext() is True:
        print "PASSED!!!"
    else:
        print "FAILED"
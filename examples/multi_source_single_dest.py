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

    population.record(['spikes', 'gsyn_exc', 'gsyn_inh', 'v'])

    p.run(runtime)

    spikes = population.get_data('spikes')
    v = population.get_data('v')
    gsyn_exc = population.get_data('gsyn_exc')
    gsyn_inh = population.get_data('gsyn_inh')

    figure_filename = "results.png"
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime)),
        # membrane potential of the postsynaptic neuron
        Panel(v.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(gsyn_exc.segments[0].filter(name='gsyn_exc')[0],
              ylabel="gsyn excitatory (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(gsyn_inh.segments[0].filter(name='gsyn_inh')[0],
              ylabel="gsyn inhibitory (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        title="Simple synfire chain example",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    p.end()

    print str(spikes.segments[0].spiketrains[0])

    if str(spikes.segments[0].spiketrains[0]) == "[ 4.1  6.6  9.3 12.1 14.5 17.1 20.  22.4 25.  28.  31.9 41.1 54.1 56.6 59.3 62.6 67.6] ms":
    # Older values with ring buffer shift = 2 "[ 4.5  7.8 12.  14.9 18.7 21.4 24.6 29.2 54.2 57.4 61.9] ms":
        return True
    return False

if __name__ == "__main__":
    if multi_source_single_dest():
        print "PASSED!!!"
    else:
        print "FAILED"

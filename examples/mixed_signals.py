import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def mixed_signals():
    runtime = 100
    nNeurons = 2
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
                       'e_rev_E': 0.,
                       'e_rev_I': -80.
                       }

    # weight_to_spike = 0.035
    weight_to_spike = 0.1
    delay = 3

    exc_population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='exc_population', in_partitions=[1, 1], out_partitions=1, n_targets=1)
    inh_population = p.Population(nNeurons, p.IF_cond_exp(**cell_params_lif), label='inh_population', in_partitions=[1, 1], out_partitions=1, n_targets=1)
    input = p.Population(1, p.SpikeSourceArray(spike_times=[1, 8, 16, 50]), label='input1')
    input2 = p.Population(1, p.SpikeSourceArray(spike_times=[3, 15, 49]), label='input2')

    p.Projection(input, exc_population, p.FixedProbabilityConnector(p_connect=0.5), p.StaticSynapse(weight=weight_to_spike, delay=2))
    p.Projection(input2, inh_population, p.FixedProbabilityConnector(p_connect=0.5), p.StaticSynapse(weight=weight_to_spike, delay=2))

    p.Projection(inh_population, exc_population, p.FromListConnector([(0, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=2), receptor_type="inhibitory")
    p.Projection(exc_population, inh_population, p.FromListConnector([(0, 0)]), p.StaticSynapse(weight=weight_to_spike, delay=2), receptor_type="excitatory")

    exc_population.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])
    inh_population.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    ve = exc_population.get_data('v')
    spikes = exc_population.get_data('spikes')
    gee = exc_population.get_data('gsyn_exc')
    gei = exc_population.get_data('gsyn_inh')


    vi = inh_population.get_data('v')
    ispikes = inh_population.get_data('spikes')
    gie = inh_population.get_data('gsyn_exc')
    gii = inh_population.get_data('gsyn_inh')

    figure_filename = "results.png"
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
        # membrane potential of the postsynaptic neuron
        Panel(ve.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[exc_population.label], yticks=True, xlim=(0, runtime), xticks=True),
        Panel(ispikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
        # membrane potential of the postsynaptic neuron
        Panel(vi.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[inh_population.label], yticks=True, xlim=(0, runtime), xticks=True),
        Panel(gee.segments[0].filter(name='gsyn_exc')[0],
              ylabel="gsyn excitatory (mV)",
              data_labels=[exc_population.label], yticks=True, xlim=(0, runtime)),
        Panel(gei.segments[0].filter(name='gsyn_inh')[0],
              ylabel="gsyn inhibitory (mV)",
              data_labels=[exc_population.label], yticks=True, xlim=(0, runtime)),
        Panel(gie.segments[0].filter(name='gsyn_exc')[0],
              ylabel="gsyn excitatory (mV)",
              data_labels=[inh_population.label], yticks=True, xlim=(0, runtime)),
        Panel(gei.segments[0].filter(name='gsyn_inh')[0],
              ylabel="gsyn inhibitory (mV)",
              data_labels=[inh_population.label], yticks=True, xlim=(0, runtime)),
        title="Single neuron",
        annotations="Simulated with {}".format(p.name())
    )

    for n in range(len(spikes.segments[0].spiketrains)):
        print("Neuron: " + str(n) + " spiked at timestep: " + str(spikes.segments[0].spiketrains[n]))
        print("Neuron (inh): " + str(n) + " spiked at timestep: " + str(ispikes.segments[0].spiketrains[n]))

    plt.show()
    print(figure_filename)

    p.end()

    if str(spikes.segments[0].spiketrains[0]) == "[ 3. 10. 18. 52.] ms" and \
        str(ispikes.segments[0].spiketrains[0]) == "[ 5.  8. 12. 15. 18. 21. 24. 29. 51. 54. 57. 62.] ms":
        return True
    else:
        return False

if __name__ == "__main__":
    if mixed_signals() is True:
        print("PASSED!!!")
    else:
        print("FAILED")
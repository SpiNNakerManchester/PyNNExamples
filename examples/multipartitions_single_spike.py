import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def multipart_base_test(parts):

    runtime = 50
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    cell_params_lif_input = {'cm': 0.25,
                       'i_offset': 0.0,
                       'tau_m': 20.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50.0,
                       'v': -40.0
                       }

    cell_params_lif_dest = {'cm': 0.25,
                             'i_offset': 0.0,
                             'tau_m': 20.0,
                             'tau_refrac': 2.0,
                             'tau_syn_E': 5.0,
                             'tau_syn_I': 5.0,
                             'v_reset': -70.0,
                             'v_rest': -65.0,
                             'v_thresh': -50.0
                             }

    weight_to_spike = 3

    input_population = p.Population(64 * parts, p.IF_curr_exp(**cell_params_lif_input), label='input', in_partitions=parts, out_partitions=parts)
    destination_population = p.Population(64 * parts, p.IF_curr_exp(**cell_params_lif_dest), label='dest', in_partitions=parts, out_partitions=parts)

    conn = [((64 * (parts - 1) + 1), (64 * (parts - 1) + 1))]

    p.Projection(input_population, destination_population, p.FromListConnector(conn),
                 p.StaticSynapse(weight=weight_to_spike, delay=2), receptor_type="excitatory")

    destination_population.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    v = destination_population.get_data('v')
    spikes = destination_population.get_data('spikes')
    g_syne = destination_population.get_data('gsyn_exc')
    g_syn = destination_population.get_data('gsyn_inh')

    figure_filename = "results.png"
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime)),
        # membrane potential of the postsynaptic neuron
        Panel(v.segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[destination_population.label], yticks=True, xlim=(0, runtime)),
        Panel(g_syne.segments[0].filter(name='gsyn_exc')[0],
              ylabel="gsyn excitatory (mV)",
              data_labels=[destination_population.label], yticks=True, xlim=(0, runtime)),
        Panel(g_syn.segments[0].filter(name='gsyn_inh')[0],
              ylabel="gsyn inhibitory (mV)",
              data_labels=[destination_population.label], yticks=True, xlim=(0, runtime)),
        title="Multi partitions base test",
        annotations="Simulated with {}".format(p.name())
    )

    p.end()

    plt.show()

    print str(64 * (parts - 1) + 1) + " " + str(spikes.segments[0].spiketrains[64 * (parts - 1) + 1])

if __name__ == "__main__":

    for i in range(13, 14):
        print("\n\n\n\n\n\n----------------TESTING WITH " + str(i) + " PARTITIONS----------------\n\n\n\n\n\n")
        multipart_base_test(i)
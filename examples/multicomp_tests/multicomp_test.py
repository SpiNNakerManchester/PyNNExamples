import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def multicomp_test():

    runtime = 100
    nNeurons = 65
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    # cell_params_tcmp = {"u_thresh": -50,
    #                    "u_reset": -70,
    #                    "u_rest": -65,
    #                    "i_offset": 0,
    #                    "v": -65
    #                    }

    weight_to_spike = 1

    population = p.Population(1, p.extra_models.IFCurrExpTwoComp(), label='population_1')
    input_e = p.Population(1, p.SpikeSourceArray(spike_times=[1, 2, 20, 21]), label='input_e')
    input_i = p.Population(1, p.SpikeSourceArray(spike_times=[4, 5, 23, 24]), label='input_i')

    p.Projection(input_e, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike, delay=1), receptor_type="soma_exc")
    #p.Projection(input_i, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike, delay=1), receptor_type="soma_inh")

    population.record(['v', 'spikes', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v')
    spikes = population.get_data('spikes')
    v = population.get_data('gsyn_exc')

    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(spikes.segments[0].spiketrains,
              yticks=True, markersize=0.2, xlim=(0, runtime)),
        # membrane potential of the postsynaptic neuron
        Panel(u.segments[0].filter(name='v')[0],
              ylabel="Soma Membrane potential (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(v.segments[0].filter(name='gsyn_exc')[0],
              ylabel="Dendrite membrane potential",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        title="multicompartment example",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    p.end()

    if str(spikes.segments[0].spiketrains[0]) == "[ 4. 11. 18. 53.] ms":
        return True
    else:
        return False


    #plt.show()

if __name__ == "__main__":
    if multicomp_test():
        print "PASSED!!!"
    else:
        print "FAILED"
import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def dendrite_comp_stimulus():

    runtime = 50
    nNeurons = 65
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    # cell_params_tcmp = {"u_thresh": -50,
    #                    "u_reset": -70,
    #                    "u_rest": -65,
    #                    "i_offset": 0,
    #                    "v": -65
    #                    }

    weight_to_spike = 100

    population = p.Population(1, p.extra_models.IFCurrExpTwoComp(), label='population_1')
    input = p.Population(1, p.SpikeSourceArray(spike_times=[1, 2, 3, 4, 5, 6]), label='input')
    input2 = p.Population(1, p.SpikeSourceArray(spike_times=[]), label='input2')

    p.Projection(input, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike, delay=1), receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike, delay=1), receptor_type="dendrite_inh")

    population.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v')
    spikes = population.get_data('spikes')
    v = population.get_data('gsyn_exc')
    rate = population.get_data('gsyn_inh')

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
        Panel(rate.segments[0].filter(name='gsyn_inh')[0],
              ylabel="Rate",
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
    if dendrite_comp_stimulus():
        print "PASSED!!!"
    else:
        print "FAILED"
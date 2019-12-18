import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def rate_based_test():

    runtime = 20
    nNeurons = 65
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    # cell_params_tcmp = {"u_thresh": -50,
    #                    "u_reset": -70,
    #                    "u_rest": -65,
    #                    "i_offset": 0,
    #                    "v": -65
    #                    }

    weight_to_spike = 10
    r = 150
    to_send = list()
    for _ in range(1, 20):
        to_send.append(r)
        r += 150

    population1 = p.Population(1, p.extra_models.IFExpRateTwoComp(starting_rate=0), label='population_1')
    source = p.Population(1, p.RateSourceArray(rate_times=[i for i in range(1, 20)], rate_values=to_send), label='rate_source')
    #source2 = p.Population(1, p.RateSourceArray(rate_times=[1, 2], rate_values=[3, 4]), label='rate_source2')

    p.Projection(source, population1, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_exc")
    #p.Projection(source2, population1, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
    #             receptor_type="dendrite_exc")

    population1.record(['v', 'gsyn_exc', 'gsyn_inh'])
    #population2.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population1.get_data('v')
    v = population1.get_data('gsyn_exc')
    rate = population1.get_data('gsyn_inh')

    #u2 = population2.get_data('v')
    #v2 = population2.get_data('gsyn_exc')
    #rate2 = population2.get_data('gsyn_inh')

    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u.segments[0].filter(name='v')[0],
              ylabel="Soma Membrane potential (mV)",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        Panel(v.segments[0].filter(name='gsyn_exc')[0],
              ylabel="Dendrite membrane potential",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        Panel(rate.segments[0].filter(name='gsyn_inh')[0],
              ylabel="Rate",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        title="multicompartment example",
        annotations="Simulated with {}".format(p.name())
    )

    plt.grid(True)

    plt.show()

    p.end()

    return False


if __name__ == "__main__":
    if rate_based_test():
        print "PASSED!!!"
    else:
        print "FAILED"
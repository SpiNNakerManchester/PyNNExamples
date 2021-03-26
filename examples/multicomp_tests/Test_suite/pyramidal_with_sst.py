import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def test(apical_learning_rate=0.05, basal_learning_rate=0.11875, sst_learning_rate=0.11875, top_down_learning_rate=0.05,
         g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8,
         top_down_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         bottom_up_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         top_down_rates=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2],
         bottom_up_rates=[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2, 2, 2, 2]):

    runtime = 5000

    p.setup(timestep=1)

    exc_apical_weight = 1
    inh_apical_weight = 0.3
    basal_weight = 0.11
    sst_dend_weight = 0.4
    sst_soma_weight = 0

    top_down = p.Population(1, p.RateSourceArray(rate_times=[i for i in range(5000)], rate_values=[0 if i < 200 else 1 for i in range(5000)], looping=4,
                                                    partitions=1), label='top_down_input')
    bottom_up = p.Population(1, p.RateSourceArray(rate_times=[i for i in range(5000)], rate_values=[0.5 for _ in range(5000)], looping=4,
                                                    partitions=1), label='bottom_up_input')

    pyramidal = p.Population(1, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
                                label='pyramidal', in_partitions=[1, 1, 1, 0], out_partitions=1)
    interneuron = p.Population(1, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D),
                                label='interneuron', in_partitions=[1, 1, 0, 0], out_partitions=1)
    top_down_neuron = p.Population(1, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D),
                                label='top_down_neuron', in_partitions=[1, 1, 0, 0], out_partitions=1)

    basal_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10000, w_max=10000,
                                                                   learning_rates=(basal_learning_rate, 0,
                                                                                   0, 0)),
        weight=basal_weight)

    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10000, w_max=10000,
                                                                   learning_rates=(apical_learning_rate, 0,
                                                                                   0, 0)),
        weight=inh_apical_weight)

    sst_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10000, w_max=10000,
                                                                                     learning_rate=sst_learning_rate),
        weight=sst_dend_weight)

    upper_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10000, w_max=10000,
                                                                                     learning_rate=top_down_learning_rate),
        weight=sst_dend_weight)

    p.Projection(top_down_neuron, pyramidal, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    p.Projection(top_down_neuron, interneuron, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    p.Projection(interneuron, pyramidal, p.AllToAllConnector(), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(bottom_up, pyramidal, p.AllToAllConnector(),  synapse_type=basal_plasticity,
                 receptor_type="basal_exc")

    p.Projection(pyramidal, interneuron, p.AllToAllConnector(), synapse_type=sst_plasticity,
                 receptor_type='dendrite_exc')

    p.Projection(pyramidal, top_down_neuron, p.AllToAllConnector(), synapse_type=upper_plasticity,
                 receptor_type='dendrite_exc')
    p.Projection(top_down, top_down_neuron, p.AllToAllConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    pyramidal.record(['v', 'gsyn_exc', 'gsyn_inh'])
    interneuron.record(['v', 'gsyn_exc'])
    top_down_neuron.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = pyramidal.get_data('v').segments[0].filter(name='v')[0]

    va = pyramidal.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = pyramidal.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    usst = interneuron.get_data('v').segments[0].filter(name='v')[0]

    vsst = interneuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    utop = top_down_neuron.get_data('v').segments[0].filter(name='v')[0]

    vtop = top_down_neuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(utop, ylabel="Top-down somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vtop, ylabel="Top-down dendritic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(usst, ylabel="SST somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vsst, ylabel="SST dendritic potential",
              data_labels=[top_down_neuron.label], yticks=True, xlim=(0, runtime)),
        title="Top-Down + SST neurons",
        annotations="Simulated with {}".format(p.name()),
    )

    plt.grid(True)

    plt.show()


    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u, ylabel="Soma Membrane potential (mV)",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(va, ylabel="Apical dendritic potential",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(vb, ylabel="Basal dendritic potential",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(usst, ylabel="SST somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(utop, ylabel="Top-down somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        title="Pyramidal + SST neurons",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    p.end()

    return True


def success_desc():
    return "Pyramidal model and sst neuron test PASSED"


def failure_desc():
    return "Pyramidal model and sst neuron test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())

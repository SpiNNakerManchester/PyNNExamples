import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def test(apical_learning_rate=0.015625, basal_learning_rate=0.25, sst_learning_rate=0.015625,
         g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8,
         top_down_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         bottom_up_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         top_down_rates=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2],
         bottom_up_rates=[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2, 2, 2, 2]):

    runtime = 3000

    p.setup(timestep=1)

    exc_apical_weight = 1
    inh_apical_weight = 0.25
    basal_weight = 0.125
    sst_dend_weight = 0.5
    sst_soma_weight = 0

    top_down = p.Population(10, p.RateSourceArray(rate_times=[i for i in range(3000)], rate_values=[0 if i < 500 else 10 for i in range(3000)], looping=4),
                            label='top_down_input')
    input = p.Population(784, p.RateSourceArray(rate_times=[i for i in range(3000)], rate_values=[1 for _ in range(3000)], looping=4),
                             label='bottom_up_input')

    pyramidalL1 = p.Population(500, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='pyramidalL1')
    interneuronL1 = p.Population(500, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom), label='interneuronL1')
    pyramidalL2 = p.Population(500, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='pyramidalL2')
    interneuronL2 = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom), label='interneuronL2')
    output_neurons = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom), label='top_down_neurons')

    basal_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-100, w_max=100,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=basal_weight)

    apical_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-100, w_max=100,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=inh_apical_weight)

    sst_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-100, w_max=100,
                                                                                     learning_rate=sst_learning_rate),
        weight=sst_dend_weight)

    basal_plasticityL2 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-100, w_max=100,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=basal_weight)

    apical_plasticityL2 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-100, w_max=100,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=inh_apical_weight)

    sst_plasticityL2 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-100, w_max=100,
                                                                                     learning_rate=sst_learning_rate),
        weight=sst_dend_weight)

    basal_plasticityOUT = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-100, w_max=100,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=basal_weight)

    # LAYER 1
    p.Projection(input, pyramidalL1, p.AllToAllConnector(), synapse_type=basal_plasticityL1,
                 receptor_type="basal_exc")

    p.Projection(pyramidalL1, interneuronL1, p.AllToAllConnector(), synapse_type=sst_plasticityL1,
                 receptor_type='dendrite_exc')

    p.Projection(interneuronL1, pyramidalL1, p.AllToAllConnector(), synapse_type=apical_plasticityL1,
                 receptor_type="apical_inh")

    p.Projection(pyramidalL2, pyramidalL1, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")

    p.Projection(pyramidalL2, interneuronL1, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    # LAYER 2
    p.Projection(pyramidalL1, pyramidalL2, p.AllToAllConnector(), synapse_type=basal_plasticityL2,
                 receptor_type='basal_exc')

    p.Projection(pyramidalL2, interneuronL2, p.AllToAllConnector(), synapse_type=sst_plasticityL2,
                 receptor_type='dendrite_exc')

    p.Projection(interneuronL2, pyramidalL2, p.AllToAllConnector(), synapse_type=apical_plasticityL2,
                 receptor_type="apical_inh")

    p.Projection(output_neurons, pyramidalL2, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")

    p.Projection(output_neurons, interneuronL2, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')


    # OUTPUT LAYER
    p.Projection(pyramidalL2, output_neurons, p.AllToAllConnector(), synapse_type=basal_plasticityOUT,
                 receptor_type='basal_exc')

    p.Projection(top_down, output_neurons, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    output_neurons.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = output_neurons.get_data('v').segments[0].filter(name='v')[0]

    v = output_neurons.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]


    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u, ylabel="somatic potential",
              data_labels=[output_neurons.label], yticks=True, xlim=(0, runtime)),
        Panel(v, ylabel="dendritic potential",
              data_labels=[output_neurons.label], yticks=True, xlim=(0, runtime)),
        title="MNIST output",
        annotations="Simulated with {}".format(p.name())
    )

    plt.grid(True)

    plt.show()

    p.end()

    return True


def success_desc():
    return "MNIST microcircuit test PASSED"


def failure_desc():
    return "MNIST microcircuit test FAILED"


if __name__ == "__main__":

    if test():
        print success_desc()
    else:
        print failure_desc()

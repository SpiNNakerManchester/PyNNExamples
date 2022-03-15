import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import random


def test(g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8):

    runtime = 105000

    # learning rates
    lPP32 = 0.0005
    lPP21 = 0.0025
    lPP10 = 0.00125
    lPI22 = 0.00125
    lPI11 = 0.005
    lIP22 = lPI22/3
    lIP11 = lPI11/3

    input_pop_size = 1

    p.setup(timestep=1)

    # Weights
    soma_weight = 0

    # Input values
    values = [40 if i < 100 else 0 if i >= 100 and i < 200 else 15 for i in range(300)]
    rate_vals = []

    for _ in range(400):
        rate_vals.extend(values)

    # Input population
    input_population = [p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(len(rate_vals))],
                                                          rate_values=rate_vals, looping=4,
                                                          partitions=1), label='bottom_up_input') for _ in
                        range(input_pop_size)]

    # # Input population
    # input_population = p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(len(rate_vals))],
    #                                                      rate_values=rate_vals, looping=1,
    #                                                      partitions=1), label='bottom_up_input')

    # Network 1
    input1 = p.Population(3, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
                          label='student_in', in_partitions=[1, 1, 1, 0], out_partitions=1, noise=1)

    # ininter1 = p.Population(4, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
    #                         label='student_int1', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=1)
    #
    # hidden1 = p.Population(4, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
    #                        label='student_hid1', in_partitions=[1, 1, 1, 0], out_partitions=1, noise=1)

    hiinter1 = p.Population(2, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
                            label='student_int2', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=1)

    out1 = p.Population(2, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
                        label='student_out', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=1)

    # Network 2
    input2 = p.Population(3, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
                          label='teacher_in', in_partitions=[1, 1, 1, 0], out_partitions=1, noise=0)# was True

    # ininter2 = p.Population(4, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
    #                         label='teacher_int1', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=0)
    #
    # hidden2 = p.Population(4, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
    #                        label='teacher_hid1', in_partitions=[1, 1, 1, 0], out_partitions=1, noise=0)

    hiinter2 = p.Population(2, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
                            label='teacher_int2', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=0)

    out2 = p.Population(2, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom, teach=False),
                        label='teacher_out', in_partitions=[1, 1, 0, 0], out_partitions=1, noise=0)

    # Plasticity rules
    basal_plast11 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lPP21, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    dend_plast11 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPI11),
        weight=random.uniform(-1, 1))

    apical_plast11 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lIP11, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    dend_plast_out12 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPP32),
        weight=random.uniform(-1, 1))

    dend_plast12 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPI22),
        weight=random.uniform(-1, 1))

    apical_plast12 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lIP22, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    basal_plast21 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lPP21, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    dend_plast21 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPI11),
        weight=random.uniform(-1, 1))

    apical_plast21 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lIP11, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    dend_plast_out22 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPP32),
        weight=random.uniform(-1, 1))

    dend_plast22 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPI22),
        weight=random.uniform(-1, 1))

    apical_plast22 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lIP22, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    basal_plast01 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lPP10, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    basal_plast02 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(
                                                                       lPP10, 0, 0, 0)),
        weight=random.uniform(-1, 1))

    # Net1 topology
    p.Projection(input1, hiinter1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=dend_plast11,
                 receptor_type='dendrite_exc')
    p.Projection(hiinter1, input1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=apical_plast11,
                 receptor_type="apical_inh")



    p.Projection(input1, out1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=dend_plast_out12,
                 receptor_type='dendrite_exc')


    p.Projection(out1, input1, p.AllToAllConnector(random_weight_matrix=True),
                 p.StaticSynapse(weight=random.uniform(-1, 1)), receptor_type="apical_exc")
    p.Projection(out1, hiinter1, p.OneToOneConnector(), p.StaticSynapse(weight=soma_weight),
                 receptor_type='soma_exc')

    # Net2 topology

    # p.Projection(input2, hidden2, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plast21,
    #              receptor_type='basal_exc')

    p.Projection(hiinter2, input2, p.AllToAllConnector(random_weight_matrix=True), synapse_type=apical_plast21,
                 receptor_type="apical_inh")
    # p.Projection(ininter2, input2, p.AllToAllConnector(random_weight_matrix=True),
    #              synapse_type=p.StaticSynapse(weight=random.uniform(-1, 1)),
    #              receptor_type="apical_inh")

    p.Projection(input2, hiinter2, p.AllToAllConnector(random_weight_matrix=True), synapse_type=dend_plast21,
                 receptor_type='dendrite_exc')
    # p.Projection(hidden2, out2, p.AllToAllConnector(random_weight_matrix=True), synapse_type=dend_plast_out22,
    #              receptor_type='dendrite_exc')
    p.Projection(input2, out2, p.AllToAllConnector(random_weight_matrix=True),
                 synapse_type=p.StaticSynapse(weight=random.uniform(-1, 1)),
                 receptor_type='dendrite_exc')


    p.Projection(out2, input2, p.AllToAllConnector(random_weight_matrix=True),
                 p.StaticSynapse(weight=random.uniform(-1, 1)), receptor_type="apical_exc")
    p.Projection(out2, hiinter2, p.OneToOneConnector(), p.StaticSynapse(weight=soma_weight),
                 receptor_type='soma_exc')

    # Networks interconnection
    p.Projection(out2, out1, p.OneToOneConnector(), p.StaticSynapse(weight=soma_weight),
                 receptor_type='soma_exc')

    # Inputs connections
    # p.Projection(input_population, input1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plast01,
    #              receptor_type='basal_exc')
    # p.Projection(input_population, input2, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plast02,
    #              receptor_type='basal_exc')
    for i in range(input_pop_size):
        p.Projection(input_population[i], input1, p.AllToAllConnector(random_weight_matrix=True),
                     synapse_type=basal_plast01, receptor_type="basal_exc")
    for i in range(input_pop_size):
        p.Projection(input_population[i], input2, p.AllToAllConnector(random_weight_matrix=True),
                     synapse_type=p.StaticSynapse(weight=random.uniform(-1, 1)),
                     receptor_type="basal_exc")

    input1.record(["gsyn_exc"])
    input2.record(["gsyn_exc", "gsyn_inh"])

    out1.record(['v', 'gsyn_exc', 'gsyn_inh'])
    out2.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u1 = out1.get_data('v').segments[0].filter(name='v')[0]

    v1 = out1.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    u2 = out2.get_data('v').segments[0].filter(name='v')[0]

    v2 = out2.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vai1 = input1.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vai2 = input2.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    sens_input = input2.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    u11 = []
    u12 = []
    u21 = []
    u22 = []

    for i in range(len(u1)):
        u11.append(float(u1[i][0]))
        u12.append(float(u1[i][1]))
        u21.append(float(u2[i][0]))
        u22.append(float(u2[i][1]))

    x = [_ for _ in range(runtime)]

    plt.plot(x, u11, color="blue", linewidth=2, label="N1")
    plt.plot(x, u12, color="green", linewidth=2, label="N2")
    plt.plot(x, u21, "--", color="aqua", linewidth=2, label="N1 teacher")
    plt.plot(x, u22, "--", color="lightgreen", linewidth=2, label="N2 teacher")

    plt.grid(True)

    plt.show()

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u1, ylabel="somatic potential",
              data_labels=[out1.label], yticks=True, xlim=(0, runtime)),
        Panel(v1, ylabel="dendritic potential",
              data_labels=[out1.label], yticks=True, xlim=(0, runtime)),
        Panel(u2, ylabel="somatic potential teacher",
              data_labels=[out2.label], yticks=True, xlim=(0, runtime)),
        Panel(v2, ylabel="dendritic potential teacher",
              data_labels=[out2.label], yticks=True, xlim=(0, runtime)),
        title="Regression output",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(vai1, ylabel="input apical potential",
              data_labels=[input1.label], yticks=True, xlim=(0, runtime)),
        Panel(vai2, ylabel="input apical potential teacher",
              data_labels=[input2.label], yticks=True, xlim=(0, runtime)),
        title="Regression output",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    with open("/localhome/g90604lp/reg/va.txt", "w") as fp:
        for i in vai1:
            fp.write(str(i))

    with open("/localhome/g90604lp/reg/vb.txt", "w") as fp:
        for i in sens_input:
            fp.write(str(i))

    with open("/localhome/g90604lp/reg/uteach.txt", "w") as fp:
        for i in u2:
            fp.write(str(i))

    with open("/localhome/g90604lp/reg/ustud.txt", "w") as fp:
        for i in u1:
            fp.write(str(i))

    p.end()

    return True


def success_desc():
    return "Regression test PASSED"


def failure_desc():
    return "Regression layer test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())

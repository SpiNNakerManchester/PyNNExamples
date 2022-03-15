import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from mnist import MNIST
import math
import random


def test(g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8):

    epochs = 7
    refresh_rate = 20

    data = MNIST('/localhome/g90604lp/datasets')

    images, labels = data.load_training()
    test_img, test_lab = data.load_testing()

    filtered_images = []
    filtered_labels = []

    filtered_test_imgs = []
    filtered_test_lab = []

    for i in range(len(labels)):
        if labels[i] == 0 or labels[i] == 1 or labels[i] == 2:
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    for i in range(len(test_lab)):
        if test_lab[i] == 0 or test_lab[i] == 1 or test_lab[i] == 2:
            filtered_test_imgs.append(test_img[i])
            filtered_test_lab.append(test_lab[i])

    sending_imgs = []
    sending_labs = []

    sending_imgs.extend(filtered_images)
    sending_labs.extend(filtered_labels)

    sending_imgs.extend(filtered_test_imgs)
    sending_labs.extend([11 for _ in range(len(filtered_test_lab))])

    runtime = (epochs * len(filtered_labels) * refresh_rate) + (len(filtered_test_lab) * 10)

    #n_images = int(math.ceil(runtime / (refresh_rate + 1))) + 2

    p.setup(timestep=1)

    # Weights
    exc_apical_weight = 1
    inh_apical_weight = 0.25
    basal_weight = 0.125
    sst_dend_weight = 0.5
    sst_soma_weight = 0

    #learning rates
    lPP32 = 0.01
    lPP21 = 0.0333
    lPP10 = 0.1111
    lIP22 = 0.02
    lIP11 = 0.0667
    # Not indicated in the paper
    lPI22 = 0.04
    lPI11 = 0.1111

    top_down = p.Population(10, p.RateLiveTeacher(10, refresh_rate, sending_labs, len(filtered_labels), partitions=1, epochs=epochs),
                            label='top_down_input')

    source = p.Population(784, p.RateSourceLive(
        784, refresh_rate, sending_imgs, len(filtered_images), partitions=7, packet_compressor=True, epochs=epochs),
                          label='input_source')

    pyramidalL1 = p.Population(500, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
        label='pyramidalL1', in_partitions=[1, 7, 1, 0], out_partitions=10, packet_compressor=[False, True, False, False],
                               atoms_per_core=4, input_pop=True)
    interneuronL1 = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
        label='interneuronL1', in_partitions=[1, 10, 0, 0], out_partitions=1, atoms_per_core=10)

    output_neurons = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom, out=True),
        label='top_down_neurons', in_partitions=[1, 10, 0, 0], out_partitions=1, atoms_per_core=10)

    basal_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(lPP10, 0, 0, 0)),
        weight=random.uniform(-1, 1)) #basal_weight

    apical_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(lPI11, 0, 0, 0)),
        weight=random.uniform(-1, 1)) #inh_apical_weight

    sst_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lIP11),
        weight=random.uniform(-1, 1)) #sst_dend_weight

    basal_plasticityOUT = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=lPP32),
        weight=random.uniform(-1, 1)) #basal_weight

    # LAYER 1
    p.Projection(source, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plasticityL1,
                 receptor_type="basal_exc")

    p.Projection(pyramidalL1, interneuronL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=sst_plasticityL1,
                 receptor_type='dendrite_exc')

    p.Projection(interneuronL1, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=apical_plasticityL1,
                 receptor_type="apical_inh")

    p.Projection(output_neurons, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), p.StaticSynapse(weight=random.uniform(-1, 1)), #exc_apical_weight
                 receptor_type="apical_exc")

    p.Projection(output_neurons, interneuronL1, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')


    # OUTPUT LAYER
    p.Projection(pyramidalL1, output_neurons, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plasticityOUT,
                 receptor_type='dendrite_exc')

    p.Projection(top_down, output_neurons, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    output_neurons.record(['gsyn_inh'])

    p.run(runtime)

    # u = output_neurons.get_data('v').segments[0].filter(name='v')[0]
    #
    # v = output_neurons.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    out_rate = output_neurons.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    # uint = interneuronL1.get_data('v').segments[0].filter(name='v')[0]

    classification = out_rate[(runtime - len(filtered_test_lab)):runtime]

    print(filtered_test_lab)

    # Plot values from SpiNNaker
    # Figure(
    #     Panel(classification, ylabel="Output_rate",
    #           data_labels=[output_neurons.label], yticks=True, xlim=((runtime - len(filtered_test_lab)), runtime)),
    #     title="MNIST output",
    #     annotations="Simulated with {}".format(p.name())
    # )

    # plt.grid(True)
    #
    # plt.show()

    p.end()

    with open("/localhome/g90604lp/out_mnist.txt", "w") as fp:
        for j in out_rate:
            fp.write(str(j))
            fp.write("\n")

    return True


def success_desc():
    return "MNIST one layer test PASSED"


def failure_desc():
    return "MNIST one layer test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
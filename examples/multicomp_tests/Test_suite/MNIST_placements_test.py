import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from mnist import MNIST
import math
import random


def test(g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8):

    runtime = 6000

    # The number of chips we want to use for the simulation
    n_chips = 48

    # The max number of neurons per core in the pyramidal population
    pyramidal_atoms_per_chip = 8

    # Parameters used to compute the chips for the pyramidal pop
    # DO NOT EDIT THIS!
    input_chips = 2
    lateral_chips = 1
    top_down_chips = 1
    teacher_chips = 1

    # The number of chips that we will be able to use for the pyramidal pop
    available_chips =\
        n_chips - input_chips - lateral_chips - top_down_chips - teacher_chips

    if available_chips <= 0:
        print("\nIncompatible number of chips, the minimum number you can use with this configuration is:")
        print(str(input_chips + lateral_chips + top_down_chips + teacher_chips + 1) + ", you gave " + str(n_chips))
        return False

    # Computes the size of the pyramidal population according to the number of chips we have
    # DO NOT EDIT THIS!
    pyramidal_pop_size = available_chips * pyramidal_atoms_per_chip

    # Needs the path of the MNIST dataset files
    data = MNIST('./datasets')

    # Loads the dataset and labels
    images, labels = data.load_training()

    # How long are we keeping an image on the inputs
    refresh_rate = 30

    # Computes the number of images to send, in case we don't use the full dataset
    # DO NOT EDIT THIS!
    n_images = int(math.ceil(runtime / (refresh_rate + 1))) + 2

    p.setup(timestep=1)

    # Weights, not used, random weight matrix used instead
    # exc_apical_weight = 1
    # inh_apical_weight = 0.25
    # basal_weight = 0.125
    # sst_dend_weight = 0.5
    # sst_soma_weight = 0

    #learning rates
    lPP32 = 0.01
    lPP10 = 0.1111
    lIP11 = 0.0667
    lPI11 = 0.1111
    #------------------------------------------------------------------------------------------------#
    # Populations
    # Teacher Population, currently uses 1 core
    top_down = p.Population(10, p.RateLiveTeacher(10, refresh_rate, labels[:n_images], partitions=1),
                            label='top_down')

    # Source Population, currently uses 2 chips
    source = p.Population(784, p.RateSourceLive(784, refresh_rate, images[:n_images], partitions=7, packet_compressor=True),
        label='input_source')


    pyramidalL1 = p.Population(pyramidal_pop_size, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
        label='pyramidalL1', in_partitions=[1, 7, 1, 0], out_partitions=7,
        packet_compressor=[False, True, False, False], atoms_per_core=pyramidal_atoms_per_chip)
    
    # Lateral interneurons, currently on 1 chip
    interneuronL1 = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
        label='interneuronL1', in_partitions=[1, 7, 0, 0], out_partitions=1, atoms_per_core=16)

    # Output populaiton, currently uses 1 chip
    output_neurons = p.Population(10, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, g_som=gsom),
        label='output_neurons', in_partitions=[1, 7, 0, 0], out_partitions=1, atoms_per_core=16)

    #------------------------------------------------------------------------------------------------#
    # Plasticity rules
    basal_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(lPP10, 0, 0, 0)),
        weight=random.uniform(-1, 1)) #basal_weight

    apical_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(lPI11, 0, 0, 0)),
        weight=random.uniform(-1, 1)) #inh_apical_weight

    sst_plasticityL1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=lIP11),
        weight=random.uniform(-1, 1)) #sst_dend_weight

    basal_plasticityOUT = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=lPP32),
        weight=random.uniform(-1, 1)) #basal_weight
    #------------------------------------------------------------------------------------------------#
    # Projections
    # LAYER 1
    p.Projection(source, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plasticityL1,
                 receptor_type="basal_exc")

    p.Projection(pyramidalL1, interneuronL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=sst_plasticityL1,
                 receptor_type='dendrite_exc')

    p.Projection(interneuronL1, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), synapse_type=apical_plasticityL1,
                 receptor_type="apical_inh")

    p.Projection(output_neurons, pyramidalL1, p.AllToAllConnector(random_weight_matrix=True), p.StaticSynapse(weight=random.uniform(-1, 1)), #exc_apical_weight
                 receptor_type="apical_exc")

    p.Projection(output_neurons, interneuronL1, p.OneToOneConnector(), p.StaticSynapse(weight=0), # sst_soma_weight
                 receptor_type='soma_exc')
    
    # OUTPUT LAYER
    p.Projection(pyramidalL1, output_neurons, p.AllToAllConnector(random_weight_matrix=True), synapse_type=basal_plasticityOUT,
                 receptor_type='dendrite_exc')

    p.Projection(top_down, output_neurons, p.OneToOneConnector(), p.StaticSynapse(weight=0), # sst_soma_weight
                 receptor_type='soma_exc')
    #------------------------------------------------------------------------------------------------#
    p.run(runtime)

    p.end()

    return False


def success_desc():
    return "MNIST one layer test PASSED"


def failure_desc():
    return "MNIST one layer test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
from mnist import MNIST
import spynnaker8 as p
import numpy as np

def test():

    data = MNIST('./datasets')

    images, labels = data.load_training()

    runtime = 10

    refresh_rate = 2
    post_neurons = 8
    partitions_involved = 7

    basal_weight = 1
    learning_rate = 0.5
    g_A = 0.8
    g_B = 1
    g_L = 0.1
    
    # Do not edit this index as it adapts the length of the dataset
    data_under_test = images[:3*refresh_rate]

    data_under_test_converted = np.array(data_under_test) / 256

    p.setup(timestep=1)

    source = p.Population(784, p.RateSourceLive(784, refresh_rate, data_under_test,
        partitions=partitions_involved, packet_compressor=True), label='input_source')

    population = p.Population(post_neurons, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
        label='population_1', in_partitions=[0, partitions_involved, 0, 0],
        out_partitions=1, packet_compressor=[False, True, False, False])


    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=basal_weight)

    p.Projection(source, population, p.AllToAllConnector(), synapse_type=plasticity,
                 receptor_type="basal_exc")

    
    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    va = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    refresh = 0
    j = 0

    VB = list()
    VB.append(0.0)

    U = list()
    U.append(0.0)

    plastic_basal_weight = [basal_weight for _ in range(784)]
    Urate = 0
    VBrate = 0
    incoming_rate = [0 for _ in range(784)]
    Iapical = 0

    for _ in range(runtime):

        Ibasal = 0
        refresh += 1

        for i in range(784):

            plastic_basal_weight[i] += (learning_rate * (Urate - VBrate) * incoming_rate[i])

            if plastic_basal_weight[i] > 10:
                plastic_basal_weight[i] = 10
            elif plastic_basal_weight[i] < -10:
                plastic_basal_weight[i] = -10

            bas = data_under_test_converted[j][i]
            incoming_rate[i] = bas
            Ibasal += (bas * plastic_basal_weight[i])

        som_voltage = float(g_B * Ibasal + g_A * Iapical)/(g_L + g_B + g_A)

        VB.append(Ibasal)
        U.append(som_voltage)

        VBrate = (Ibasal if (Ibasal > 0 and Ibasal < 2) else 0 if Ibasal <= 0 else 2)
        som_voltage = float(som_voltage * (g_L + g_B + g_A)) / g_B
        Urate = (som_voltage if (som_voltage > 0 and som_voltage < 2) else 0 if som_voltage <= 0 else 2)

        if refresh > refresh_rate:
            refresh = 0
            j += 1

    for n in range(post_neurons):
        for i in range(runtime):
            num = float(vb[i][n])
            if (float(int(num * 100)) / 100 != float(int(VB[i] * 100)) / 100) and (round(num, 2) != round(VB[i], 2)):
                print("neuron " + str(n))
                for j in range(i + 1):
                    print("Basal voltage " + str(float(vb[j][n])) + " != " + str(VB[j]) + " time " + str(j))
                return False
    
    return True

def success_desc():
    return "MNIST plastic injector test PASSED"


def failure_desc():
    return "MNIST plastic injector test FAILED"

if __name__ == "__main__":
    if test():
        print(success_desc())
    else:
        print(failure_desc())
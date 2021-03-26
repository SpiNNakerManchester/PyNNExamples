from mnist import MNIST
import spynnaker8 as p
import numpy as np
import math

def test():

    data = MNIST('./datasets')

    images, labels = data.load_training()

    runtime = 100

    refresh_rate = 2
    post_neurons = 16
    partitions_involved = 7

    n_images = int(math.ceil(runtime / (refresh_rate + 1))) + 1
    
    # Do not edit this index as it adapts the length of the dataset
    data_under_test = images[:n_images]

    data_under_test_converted = np.array(data_under_test) / 256

    p.setup(timestep=1)

    source = p.Population(784, p.RateSourceLive(784, refresh_rate, data_under_test,
        partitions=partitions_involved, packet_compressor=True), label='input_source')

    population = p.Population(post_neurons, p.extra_models.PyramidalRate(),
        label='population_1', in_partitions=[0, partitions_involved, 0, 0],
        out_partitions=1, packet_compressor=[False, True, False, False], atoms_per_core=8)

    p.Projection(source, population, p.AllToAllConnector(), p.StaticSynapse(weight=1),
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

    for _ in range(runtime):
        refresh += 1
        VB.append(sum(data_under_test_converted[j]))
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
    return "MNIST static injector test PASSED"


def failure_desc():
    return "MNIST static injector test FAILED"

if __name__ == "__main__":
    if test():
        print(success_desc())
    else:
        print(failure_desc())
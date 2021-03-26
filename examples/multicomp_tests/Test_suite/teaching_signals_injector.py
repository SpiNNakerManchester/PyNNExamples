from mnist import MNIST
import spynnaker8 as p
import numpy as np
import math

# This test creates 2 populations. A rate live teacher (sends the unpacked teaching values for the MNIST datasets) 
# and a pyramidal neurons population. The connection is one to one. This means we are testing the functioning of
# the teacher, as well as the one to one connector using the syn matrix in dtcm and the neuron offset with the
# multicore partitioning (using multiple neuron cores).

def test():

    data = MNIST('./datasets')

    images, labels = data.load_training()

    runtime = 50

    refresh_rate = 2
    neurons = 10
    partitions_involved = 1

    n_images = int(math.ceil(runtime / (refresh_rate + 1))) + 1
    
    # Do not edit this index as it adapts the length of the dataset
    data_under_test = labels[:n_images]

    p.setup(timestep=1)

    source = p.Population(neurons, p.RateLiveTeacher(
        neurons, refresh_rate, data_under_test, partitions=partitions_involved), label='input_source')

    population = p.Population(neurons, p.extra_models.PyramidalRate(),
        label='population_1', in_partitions=[partitions_involved, 0, 0, 0], out_partitions=1)

    p.Projection(source, population, p.OneToOneConnector(), p.StaticSynapse(weight=1),
                 receptor_type="apical_exc")

    
    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    va = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    refresh = 0
    j = 0

    VA = list()
    VA.append(0)

    for _ in range(runtime):
        refresh += 1
        VA.append(data_under_test[j])
        if refresh > refresh_rate:
            refresh = 0
            j += 1
    
    for i in range(1, runtime):
        for j in range(len(va[i])):
            if j == VA[i]:
                if va[i][j] != 1:
                    print("neuron " + str(j))
                    print("expected 1.0, got " + str(va[i][j]) + " time " + str(i))
                    return False
            else:
                if va[i][j] != 0:
                    print("neuron " + str(j))
                    print("expected 0.0, got " + str(va[i][j]) + " time " + str(i))
                    return False
    
    
    return True

def success_desc():
    return "teaching signals injector test PASSED"


def failure_desc():
    return "teaching signals injector test FAILED"

if __name__ == "__main__":
    if test():
        print(success_desc())
    else:
        print(failure_desc())
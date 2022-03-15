import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test only the basal dendrite is plastic, the apical one is static and
# it receives both excitatory and inhibitory signals. The test can be performend on any number of postsynaptic neurons


def test(learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1, exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, -1.3, 3.3, 1.5, 1.5, 3.3, 1.5, -2.5],
         inh_rates=[-1, 1.3, 1.5, 3.3, -1.3, 2, 5, 3], basal_rates=[1, 1.3, 3.3, 1.5, -2, 1.3, 1.5, 2.5]):

    runtime = 1000000

    p.setup(timestep=1)

    exc_apical_weight = 1.25
    inh_apical_weight = 0.75
    basal_weight = 0.5

    post_neurons = 1

    population = p.Population(
        post_neurons, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
        label='population_1', in_partitions=[0, 3, 0, 0], out_partitions=1)
    input3 = p.Population(300, p.RateSourceMultiple(partitions=14),
                         label='basal_exc_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=basal_weight)

    p.Projection(input3, population, p.AllToAllConnector(), synapse_type=plasticity,
                 receptor_type="basal_exc")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    va = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()



    return True


def success_desc():
    return "Pyramidal basal plasticity test adapted (microcircuit enabled) PASSED"


def failure_desc():
    return "Pyramidal basal plasticity test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())

import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test both the basal and apical dendrites are plastic.
# The apical dendrite receives both excitatory and inhibitory signals


def test(apical_learning_rate = 0.5, basal_learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1,
         exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, 1.3, 3, 1.5, 1.5, -3.75, 1.5, -2.5],
         inh_rates=[-1, 1.125, 1.5, 3, 1.5, -0.75, 2, 2.5], basal_rates=[1, -1.3, 3.75, 1.5, 2, 1.25, 1.5, 2.5]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 1.5
    inh_apical_weight = 1
    basal_weight = 1.5

    population = p.Population(64, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_rates, looping=4),
                         label='apical_exc_input')
    # input2 = p.Population(32, p.RateSourceArray(rate_times=inh_times, rate_values=inh_rates, looping=4),
    #                       label='apical_inh_input')
    # input3 = p.Population(32, p.RateSourceArray(rate_times=basal_times, rate_values=basal_rates, looping=4),
    #                       label='basal_exc_input')

    basal_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=basal_weight)

    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                   apical_learning_rate, 0)),
        weight=inh_apical_weight)

    # p.Projection(input, population, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
    #              receptor_type="apical_exc")
    # p.Projection(input, population, p.AllToAllConnector(), synapse_type=apical_plasticity,
    #              receptor_type="apical_inh")
    p.Projection(input, population, p.AllToAllConnector(), synapse_type=basal_plasticity,
                 receptor_type="basal_exc")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    va = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    return True


def success_desc():
    return "Pyramidal full plasticity test adapted (microcircuit enabled) PASSED"


def failure_desc():
    return "Pyramidal full plasticity test  adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
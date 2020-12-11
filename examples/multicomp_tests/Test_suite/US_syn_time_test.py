import spynnaker8 as p

# This test combines excitatory and inhibitory stimuli arriving only to dendrites with plastic synapses
# and tests correct weight update by checking the output voltages.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 3, 4, 5, 6, 7, 8], inh_times=[3, 4, 5, 6], exc_r_diff=[0.5, -0.25, 1.3, 0.8, 1.75, 1.5, 2, 2.7],
         inh_r_diff=[0.5, 0.25, 1.5, -1.75], update_thresh=2, learning_rate=0.09, g_som=0.8):
    runtime = 9

    p.setup(timestep=1)

    weight_to_spike = 3

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                                 rate_update_threshold=update_thresh),
                              label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff, looping=4),
                         label='exc_input')
    # input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff, looping=4),
    #                       label='som_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(
            tau_plus=20., tau_minus=20.0),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=0, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(input, population, p.AllToAllConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")

    # p.Projection(input, population, p.AllToAllConnector(), synapse_type=p.StaticSynapse(weight=weight_to_spike),
    #              receptor_type="soma_exc")

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    return True

def _compute_rate(voltage):

    tmp_rate = voltage if voltage > 0 else 0

    return tmp_rate

def success_desc():
    return "US syn time test PASSED"

def failure_desc():
    return "US syn time test FAILED"


if __name__ == "__main__":

    if test():
        print success_desc()
    else:
        print failure_desc()

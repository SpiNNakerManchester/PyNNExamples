import spynnaker8 as p

# This tests combine excitatory and inhibitory stimuli arriving only to dendrites. First, only excitatory stimulus is
# presented, then only inhibitory and finally both combined.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         inh_r_diff=[1, 1.3, 1.5, 3.3]):
    runtime = 9

    p.setup(timestep=1)

    weight_to_spike = 1.15

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L), label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff), label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff), label='inh_input')

    p.Projection(input, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_inh")

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    #U_exp = [0.0, 0.0, 9.52377319, 20.95211792, 11.42834473, 0.0, 16.19024658, 0.0, 0.0]
    #V_exp = [0.0, 0.0, 10.0, 21.99981689, 11.99981689, 0.0, 16.99981689, 0.0, 0.0]

    test_exc_times = [t + 1 for t in exc_times]
    test_inh_times = [t + 1 for t in inh_times]

    Idnd = 0
    j = 0
    k = 0

    for i in range(len(u)):
        if i in test_exc_times:
            Idnd += (exc_r_diff[j] * weight_to_spike)
            j += 1
        if i in test_inh_times:
            Idnd -= (inh_r_diff[k] * weight_to_spike)
            k += 1
        U = (float(g_D * Idnd) / (g_L * (g_L + g_D)))
        num =float(u[i])
        if float(int(num * 1000) / 1000) != float(int(U * 1000) / 1000):
            print "Somatic voltage " + str(float(u[i])) + " != " + str(U)
            return False

    Idnd = 0
    j = 0
    k = 0

    for i in range(len(v)):
        if i in test_exc_times:
            Idnd += (exc_r_diff[j] * weight_to_spike)
            j += 1
        if i in test_inh_times:
            Idnd -= (inh_r_diff[k] * weight_to_spike)
            k += 1
        V = float(Idnd) / g_L
        num = float(v[i])
        if float(int(num * 1000) / 1000) != float(int(V * 1000) / 1000):
            print "Dendritic voltage " + str(float(v[i])) + " != " + str(V)
            return False


    return True

def success_desc():
    return "Dendrite excitatory and inhibitory stimuli test PASSED"

def failure_desc():
    return "Dendrite excitatory and inhibitory stimuli test FAILED"

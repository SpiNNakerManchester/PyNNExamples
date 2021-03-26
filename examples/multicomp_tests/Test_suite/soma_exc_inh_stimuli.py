import spynnaker8 as p

# This tests combine excitatory and inhibitory stimuli arriving only to soma. First, only excitatory stimulus is
# presented, then only inhibitory and finally both combined.


def test(g_D=2, g_L=0.1, E_E=4.667, E_I=-0.333, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6],
         exc_r_diff=[1, 1.3, 3.3, 1.5] , inh_r_diff=[1, 1.3, 1.5, 3.3]):
    runtime = 900

    p.setup(timestep=1)

    weight_to_spike = 0.7

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, e_rev_E=E_E, e_rev_I=E_I),
                              label='population_1')
    input1 = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff, looping=4),
                          label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff, looping=4),
                          label='inh_input')

    p.Projection(input1, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_inh")

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    j = 0
    k = 0

    test_exc_times = [t + 1 for t in exc_times]
    test_inh_times = [t + 1 for t in inh_times]

    for i in range(len(u)):

        r_E = 0
        r_I = 0

        if i in test_exc_times:
            r_E += exc_r_diff[j]
            j += 1
        if i in test_inh_times:
            r_I += inh_r_diff[k]
            k += 1
        num = float(u[i])
        U_exp = float((weight_to_spike * E_E * r_E) + (weight_to_spike * E_I * r_I)) \
                    / (g_D + g_L + (r_E * weight_to_spike) + (r_I * weight_to_spike))
        if (float(int(num * 1000)) / 1000 != float(int(U_exp * 1000)) / 1000) and (round(num, 3) != round(U_exp, 3)):
            print("Somatic voltage " + str(float(u[i])) + " != " + str(U_exp))
            return False

    V_exp = 0

    for i in range(len(v)):
        num = float(v[i])
        if (float(int(num * 1000)) / 1000 != V_exp) and (round(num, 3) != round(V_exp, 3)):
            print("Dendritic voltage " + str(float(v[i])) + " != " + str(V_exp))
            return False


    return True

def success_desc():
    return "Soma excitatory and inhibitory stimuli test PASSED"

def failure_desc():
    return "Soma excitatory and inhibitory stimuli test FAILED"


if __name__ == "__main__":
    if test():
        print(success_desc())
    else:
        print(failure_desc())
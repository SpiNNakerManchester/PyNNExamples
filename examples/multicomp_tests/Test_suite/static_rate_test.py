import spynnaker8 as p

# This test uses the same network as Mixed static dendritic and somatic stimuli test, but with different values and
# also checks the correctness of the output rates.


def test(g_D=2, g_L=0.1, E_E=4.667, E_I=0.333, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_dend_times=[1, 3, 4, 7],
         inh_dend_times=[2, 3, 5, 6], exc_r_diff=[[1, 1.3, 3.3, 1.5], [1.5, 2, 1.3, 3]] , inh_r_diff=[[1, 1.3, 1.5, 3.3], [1.5, 2, 3, 1.3]],
         exc_dend_r_diff=[[4, 3.5, 12, -6], [1.5, 8, 5, 2.5]], inh_dend_r_diff=[[1.5, 8, -5, -2.5], [1, 3.5, 2, 9]],
         update_thresh=2):

    runtime = 9

    p.setup(timestep=1)

    weight_to_spike = 1

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, e_rev_E=E_E, e_rev_I=E_I,
                                                                 rate_update_threshold=update_thresh),
                              label='population_1')

    input1 = p.Population(1, p.RateSourceArray(rate_times=exc_dend_times, rate_values=exc_dend_r_diff[0]), label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_dend_times, rate_values=inh_dend_r_diff[0]), label='inh_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=exc_dend_times, rate_values=exc_dend_r_diff[1]), label='exc_input2')
    input4 = p.Population(1, p.RateSourceArray(rate_times=inh_dend_times, rate_values=inh_dend_r_diff[1]), label='inh_input2')

    soma_input1 = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff[0]), label='soma_exc_input')
    soma_input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff[0]), label='soma_inh_input')
    soma_input3 = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff[1]), label='soma_exc_input2')
    soma_input4 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff[1]), label='soma_inh_input2')

    p.Projection(input1, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_inh")
    p.Projection(input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_exc")
    p.Projection(input4, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="dendrite_inh")

    p.Projection(soma_input1, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_exc")
    p.Projection(soma_input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_inh")
    p.Projection(soma_input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_exc")
    p.Projection(soma_input4, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_inh")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    rate = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    # Compute the expected values and check with the extracted ones
    r_E = 0
    r_I = 0

    j = 0
    k = 0
    l = 0
    m = 0
    Idnd = 0

    test_exc_times = [t + 1 for t in exc_times]
    test_inh_times = [t + 1 for t in inh_times]

    test_exc_dend_times = [t + 1 for t in exc_dend_times]
    test_inh_dend_times = [t + 1 for t in inh_dend_times]

    for i in range(len(v)):

        if i in test_exc_dend_times:
            Idnd += (exc_dend_r_diff[0][l] * weight_to_spike)
            Idnd += (exc_dend_r_diff[1][l] * weight_to_spike)
            l += 1
        if i in test_inh_dend_times:
            Idnd -= (inh_dend_r_diff[0][m] * weight_to_spike)
            Idnd -= (inh_dend_r_diff[1][m] * weight_to_spike)
            m += 1

        V_exp = float(Idnd) / g_L
        num = float(v[i])
        if float(int(num * 1000)) / 1000 != float(int(V_exp * 1000)) / 1000:
            print "Dendritic voltage " + str(float(v[i])) + " != " + str(V_exp)
            return False

    j = 0
    k = 0
    l = 0
    m = 0
    Idnd = 0
    r_E = 0
    r_I = 0
    U_vals = []

    for i in range(len(u)):

        if i in test_exc_times:
            r_E += exc_r_diff[0][j]
            r_E += exc_r_diff[1][j]
            j += 1
        if i in test_inh_times:
            r_I += inh_r_diff[0][k]
            r_I += inh_r_diff[1][k]
            k += 1
        if i in test_exc_dend_times:
            Idnd += (exc_dend_r_diff[0][l] * weight_to_spike)
            Idnd += (exc_dend_r_diff[1][l] * weight_to_spike)
            l += 1
        if i in test_inh_dend_times:
            Idnd -= (inh_dend_r_diff[0][m] * weight_to_spike)
            Idnd -= (inh_dend_r_diff[1][m] * weight_to_spike)
            m += 1

        num = float(u[i])
        U_exp = float((weight_to_spike * E_E * r_E) - (weight_to_spike * E_I * r_I) + (g_D * float(Idnd) / g_L)) \
                    / (g_D + g_L + (r_E * weight_to_spike) + (r_I * weight_to_spike))
        U_vals.append(U_exp)
        if float(int(num * 1000)) / 1000 != float(int(U_exp * 1000)) / 1000:
            print "Somatic voltage " + str(float(u[i])) + " != " + str(U_exp)
            return False

    cubic_term = -24.088958
    square_term = 48.012303
    linear_term = 73.084895
    constant_term = 1.994506
    expected_rate = 0

    for i in range(len(U_vals)):
        if U_vals[i] > 2:
            tmp_rate = 150.0
        elif U_vals[i] < 0:
            tmp_rate = 0.01
        else:
            tmp_rate =\
                (cubic_term * (U_vals[i] ** 3)) + (square_term * (U_vals[i] ** 2)) + (linear_term * U_vals[i]) + constant_term
        if (tmp_rate - expected_rate > update_thresh) or (tmp_rate - expected_rate < -update_thresh):
            expected_rate = tmp_rate
        if float(int(rate[i] *1000)) / 1000 != float(int(expected_rate * 1000)) / 1000:
            print "Rate " + str(float(rate[i])) + " != " + str(expected_rate)
            return False


    return True

def success_desc():
    return "Static rate test PASSED"

def failure_desc():
    return "Static rate test FAILED"
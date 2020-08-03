import spynnaker8 as p

# This test combines excitatory and inhibitory stimuli arriving to dendrites with plastic synapses with some somatic
# teaching current provided and tests correct weight update by checking the output voltages.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[0.6, 0.2, 1.7, 1.4],
         inh_r_diff=[0.6, 0.2, 1.4, 1.7], update_thresh=2, learning_rate=0.09, E_E=4.667, E_I=0.333,
         som_exc_times=[1, 3, 4, 7], som_inh_times=[2, 3, 5, 6], som_exc_r_diff=[0.3, 0.4, 1, 0.25],
         som_inh_r_diff=[0.4, 1, 0.7, 1.6]):
    runtime = 9

    p.setup(timestep=1)

    weight_to_spike = 1.41

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=update_thresh,
                                                                 e_rev_E=E_E, e_rev_I=E_I), label='population_1')

    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff), label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff), label='inh_input')

    soma_input1 = p.Population(1, p.RateSourceArray(rate_times=som_exc_times, rate_values=som_exc_r_diff),
                               label='soma_exc_input')
    soma_input2 = p.Population(1, p.RateSourceArray(rate_times=som_inh_times, rate_values=som_inh_r_diff),
                               label='soma_inh_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(
            tau_plus=20., tau_minus=20.0),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=0, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(input, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_inh")

    p.Projection(soma_input1, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_exc")
    p.Projection(soma_input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_inh")

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    test_exc_times = [t + 1 for t in exc_times]
    test_inh_times = [t + 1 for t in inh_times]

    som_test_exc_times = [t + 1 for t in som_exc_times]
    som_test_inh_times = [t + 1 for t in som_inh_times]

    Idnd = 0
    j = 0
    k = 0
    l = 0
    m = 0
    weight_exc = weight_to_spike
    weight_inh = weight_to_spike
    delta_exc = 0
    delta_inh = 0
    irate_exc = 0
    irate_inh = 0
    r_E = 0
    r_I = 0
    prev_rate = 0
    U_rate = 0
    V_rate = 0
    V_vals = []
    U_vals = []

    for i in range(len(u)):

        if i in test_exc_times:
            weight_exc += delta_exc
            if weight_exc > 10:
                weight_exc = 10
            elif weight_exc < 0:
                weight_exc = 0
            Idnd += (exc_r_diff[j] * weight_exc)
            irate_exc += exc_r_diff[j]
            j += 1
        if i in test_inh_times:
            weight_inh += delta_inh
            if weight_inh > 10:
                weight_inh = 10
            elif weight_inh < 0:
                weight_inh = 0
            Idnd -= (inh_r_diff[k] * weight_inh)
            irate_inh += inh_r_diff[k]
            k += 1

        if i in som_test_exc_times:
            r_E += som_exc_r_diff[l]
            l += 1
        if i in som_test_inh_times:
            r_I += som_inh_r_diff[m]
            m += 1

        Vdnd = float(Idnd) / g_L
        V_vals.append(Vdnd)
        num = float(v[i])
        if (float(int(num * 100)) / 100 != float(int(Vdnd * 100)) / 100) and (round(num, 2) != round(Vdnd, 2)):
            for z in range(len(V_vals)):
                print "Dendritic voltage " + str(float(v[z])) + " expected " + str(V_vals[z]) + " index " + str(z)
            return False

        U = float((weight_to_spike * E_E * r_E) - (weight_to_spike * E_I * r_I) + (g_D * Vdnd)) \
                / (g_D + g_L + (r_E * weight_to_spike) + (r_I * weight_to_spike))
        U_vals.append(U)
        num =float(u[i])
        if (float(int(num * 100)) / 100 != float(int(U * 100)) / 100) and (round(num, 2) != round(U, 2)):
            for z in range(len(U_vals)):
                print "Somatic voltage " + str(float(u[z])) + " expected " + str(U_vals[z]) + " index " + str(z)
            return False

        out_rate = _compute_rate(U)

        if abs(out_rate - prev_rate) > update_thresh:
            prev_rate = out_rate

            V_rate = _compute_rate(Vdnd)
            U_rate = _compute_rate((float(g_L + g_D) * U) / g_D)

        delta_exc = learning_rate * (U_rate - V_rate) * irate_exc
        delta_inh = learning_rate * (U_rate - V_rate) * irate_inh



    return True

def _compute_rate(voltage):

    cubic_term = -24.088958
    square_term = 48.012303
    linear_term = 73.084895
    constant_term = 1.994506

    if voltage > 2:
        tmp_rate = 150.0
    elif voltage < 0:
        tmp_rate = 0.01
    else:
        tmp_rate = \
            (cubic_term * (voltage ** 3)) + (square_term * (voltage ** 2)) + (linear_term * voltage) + constant_term

    return tmp_rate

def success_desc():
    return "Dendritc plasticity with somatic teaching current test PASSED"

def failure_desc():
    return "Dendritic plasticity with somatic teaching current test FAILED"

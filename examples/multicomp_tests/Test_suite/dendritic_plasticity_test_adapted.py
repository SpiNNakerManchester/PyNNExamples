import spynnaker8 as p

# This test combines excitatory and inhibitory stimuli arriving only to dendrites with plastic synapses
# and tests correct weight update by checking the output voltages.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[0.5, -0.25, 1.75, 1.5],
         inh_r_diff=[0.5, 0.25, 1.5, -1.75], update_thresh=2, learning_rate=0.09, g_som=0.8):
    runtime = 9

    p.setup(timestep=1)

    weight_to_spike = 3

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                                 rate_update_threshold=update_thresh),
                              label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff, looping=4),
                         label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff, looping=4),
                          label='inh_input')

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

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    test_exc_times = [t + 1 for t in exc_times]
    test_inh_times = [t + 1 for t in inh_times]

    ext_times = [_ for _ in range(runtime)]

    j = 0
    k = 0
    dend_exc = []
    dend_inh = []

    for i in ext_times:
        if i in test_exc_times:
            dend_exc.append(exc_r_diff[j])
            j += 1
        else:
            dend_exc.append(0)

        if i in test_inh_times:
            dend_inh.append(inh_r_diff[k])
            k += 1
        else:
            dend_inh.append(0)

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
    prev_rate = 0
    U_rate = 0
    V_rate = 0
    V_vals = []
    U_vals = []

    for i in range(len(u)):

        Idnd = 0

        if i in ext_times:
            weight_exc += delta_exc
            if weight_exc > 10:
                weight_exc = 10
            elif weight_exc < -10:
                weight_exc = -10
            Idnd += ((dend_exc[j] * weight_exc) if dend_exc[j] > 0 else 0)
            irate_exc = dend_exc[j]
            j += 1
        if i in ext_times:
            weight_inh += delta_inh
            if weight_inh > 10:
                weight_inh = 10
            elif weight_inh < -10:
                weight_inh = -10
            Idnd -= ((dend_inh[k] * weight_inh) if dend_inh[k] > 0 else 0)
            irate_inh = dend_inh[k]
            k += 1

        Vdnd = float(Idnd)
        V_vals.append(Vdnd)
        num = float(v[i])
        if (float(int(num * 100)) / 100 != float(int(Vdnd * 100)) / 100) and (round(num, 2) != round(Vdnd, 2)):
            for z in range(len(V_vals)):
                print "Dendritic voltage " + str(float(v[z])) + " expected " + str(V_vals[z]) + " index " + str(z)
            return False

        U = float((g_D * Vdnd)) \
            / (g_D + g_L + g_som)
        U_vals.append(U)
        num = float(u[i])
        if (float(int(num * 100)) / 100 != float(int(U * 100)) / 100) and (round(num, 2) != round(U, 2)):
            for z in range(len(U_vals)):
                print "Somatic voltage " + str(float(u[z])) + " expected " + str(U_vals[z]) + " index " + str(z)
            return False

        out_rate = _compute_rate(U)

        V_rate = _compute_rate(Vdnd)
        U_rate = _compute_rate((float(g_L + g_D) * U) / g_D)

        delta_exc = learning_rate * (U_rate - V_rate) * irate_exc
        delta_inh = learning_rate * (U_rate - V_rate) * irate_inh


    return True

def _compute_rate(voltage):

    tmp_rate = voltage if voltage > 0 else 0

    return tmp_rate

def success_desc():
    return "Dendritc plasticity test adapted (microcircuit enabled) PASSED"

def failure_desc():
    return "Dendritic plasticity test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print success_desc()
    else:
        print failure_desc()

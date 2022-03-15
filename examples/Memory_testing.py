import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def Memory_testing(n_cores, syn_cores, times, min_times, avg_times, max_syn_t, min_syn_t, avg_syn_t):

    runtime = 100
    nNeurons = 64*n_cores
    n_targets = n_cores
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    cell_params_lif = {'cm': 0.25,
                       'i_offset': 0.0,
                       'tau_m': 20.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50.0,
                       }

    population = p.Population(nNeurons, p.IF_curr_exp(**cell_params_lif), label='population_1', in_partitions=[syn_cores-1, 1], out_partitions=1, n_targets=n_targets)

    population.record(['gsyn_exc'])

    p.run(runtime)

    ge = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
    #syn_times = population.get_data('synapse')


    # tmpmax = 0
    # tmpmin = 1000000
    # tmpavg = []
    # for k in syn_times.keys():
    #     if max(syn_times[k]) > tmpmax:
    #         tmpmax = max(syn_times[k])
    #     if min(syn_times[k]) < tmpmin:
    #         tmpmin = min(syn_times[k])
    #     tmpavg.append(sum(syn_times[k]) / len(syn_times[k]))
    #
    # average = sum(tmpavg) / len(tmpavg)
    #
    # tmpmax = tmpmax * 0.005
    # tmpmin = tmpmin * 0.005
    # average = average * 0.005
    #
    # max_syn_t[(n_cores, syn_cores)] = tmpmax
    # min_syn_t[(n_cores, syn_cores)] = tmpmin
    # avg_syn_t[(n_cores, syn_cores)] = average

    rd_time = 0
    rd_time_min = 1000000
    rd_avg = []

    for i in range(1, len(ge)):
        if max(ge[i]) > rd_time:
            rd_time = max(ge[i])
        if min(ge[i]) < rd_time:
            rd_time_min = min(ge[i])

        s = 0
        for j in ge[i]:
            s += float(j)
        rd_avg.append(s / len(ge[i]))

    rd_time = rd_time * 0.005
    rd_time_min = rd_time_min * 0.005
    rd_avg_time = (sum(rd_avg) / len(rd_avg)) * 0.005

    times[(n_cores, syn_cores)] = rd_time
    min_times[(n_cores, syn_cores)] = rd_time_min
    avg_times[(n_cores, syn_cores)] = rd_avg_time

    p.end()


if __name__ == "__main__":

    times = dict()
    min_times = dict()
    avg_times = dict()
    max_syn_t = dict()
    min_syn_t = dict()
    avg_syn_t = dict()

    with open("/localhome/g90604lp/memory_times_SDRAM_out.txt", "w") as fp:
        fp.write("Read times\n")

    # with open("/localhome/g90604lp/syn_times_SysRAM_fin.txt", "a") as fp:
    #   pf.write("Write times\n")

    for n_cores in range(1, 10):
        for syn_cores in range(2, 15):
            if n_cores + syn_cores <= 15:
                print("Simulating " + str(n_cores) + " neuron cores and " + str(syn_cores) + " synapse cores")
                Memory_testing(n_cores, syn_cores, times, min_times, avg_times, max_syn_t, min_syn_t, avg_syn_t)

                with open("/localhome/g90604lp/memory_times_SDRAM_out.txt", "a") as fp:
                    fp.write(str((n_cores, syn_cores)) + " " + str(times[(n_cores, syn_cores)]) + " " + str(min_times[(n_cores, syn_cores)]) + " " + str(avg_times[(n_cores, syn_cores)]) + "\n")

                # with open("/localhome/g90604lp/syn_times_SysRAM_fin.txt", "a") as fp:
                #   fp.write(str((n_cores, syn_cores)) + " " + str(max_syn_t[(n_cores, syn_cores)]) + " " + str(min_syn_t[(n_cores, syn_cores)]) + " " + str(avg_syn_t[(n_cores, syn_cores)]) + "\n")


    print(times)

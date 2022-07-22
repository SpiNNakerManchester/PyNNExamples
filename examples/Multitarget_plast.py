import spynnaker8 as p
import math
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import random


def multipartition_test(part, step, timestep, neurons, p_conn, n_targets, out_partitions, post_neurons, wc_time):

    runtime = 3 * timestep
    p.setup(timestep=timestep)

    cell_params_lif_input = {'cm': 0.25,
                             'i_offset': 0.0,
                             'tau_m': 20.0,
                             'tau_refrac': 2.0,
                             'tau_syn_E': 5.0,
                             'tau_syn_I': 5.0,
                             'v_reset': -70.0,
                             'v_rest': -65.0,
                             'v_thresh': -50.0,
                            }

    cell_params_lif_dest = {'cm': 0.25,
                            'i_offset': 0.0,
                            'tau_m': 20.0,
                            'tau_refrac': 2.0,
                            'tau_syn_E': 5.0,
                            'tau_syn_I': 5.0,
                            'v_reset': -70.0,
                            'v_rest': -65.0,
                            'v_thresh': -50.0
                            }

    weight_to_spike = 0.1
    delay = 1 * timestep

    input_population = p.Population(neurons, p.IF_curr_exp(**cell_params_lif_input),
                                    label='input', in_partitions=[1, 1], out_partitions=part[0], n_targets=1, wc_time=20)
    # input_population = p.Population(int(neurons/step), p.IF_curr_exp(**cell_params_lif_input),
    #                                label='input', in_partitions=[1, 1], out_partitions=part[0], n_targets=1, wc_time=20)
    sel = [int(i * step) for i in range(int(neurons/step))]
    # sel = [_ for _ in range(int(neurons/step))]
    input_population.set_initial_value("v", -40, sel)

    destination_population = p.Population(post_neurons, p.IF_curr_exp(**cell_params_lif_dest),
                                          label='dest', in_partitions=part, out_partitions=out_partitions,
                                          n_targets=n_targets, wc_time=wc_time)

    # connections = conn_generator(neurons, post_neurons, p_conn)
    # connections = conn_generator(int(neurons/step), post_neurons, p_conn)

    stdp_model = p.STDPMechanism(
        timing_dependence=p.SpikePairRule(
            tau_plus=1.7, tau_minus=3.7, A_plus=0.005, A_minus=0.005),
        weight_dependence=p.AdditiveWeightDependence(
            w_min=0.0, w_max=1), weight=weight_to_spike)

    p.Projection(input_population, destination_population, p.FixedProbabilityConnector(p_connect=p_conn),
                 synapse_type=stdp_model)

    destination_population.record(['synapse'])

    p.run(runtime)

    synaptic_events = destination_population.get_data('synapse')

    p.end()

    syn_events = dict()
    for key in synaptic_events:
        syn_events[key] = int(synaptic_events[key][0])

    ret_val = {"syn_events": syn_events}

    events_sum = 0
    tgt = 0
    for key in syn_events:
        #if tgt < part[0]:
        events_sum += syn_events[key]
            #tgt += 1
    syn_avg = float(events_sum) / post_neurons

    ret_val_avg = {"dma_read": 0,
                   "state_update": 0,
                   "loop_time": 0,
                   "syn_events": events_sum,
                   "syn_events_avg": syn_avg,
                   "state_update_input": 0,
                   "loop_time_input": 0}

    return ret_val_avg


def conn_generator(pre_size, post_size, probability):

    connections = []

    random.seed(1)

    for i in range(pre_size):
        for j in range(post_size):
            random_num = random.uniform(0.0000, 1.0000)
            if random_num <= probability:
                connections.append((i, j, 0.1, 1))

    return connections


if __name__ == "__main__":

    values = {"dma_read": [],
              "state_update": [],
              "loop_time": [],
              "syn_events": [],
              "syn_events_avg": [],
              "state_update_input": [],
              "loop_time_input": []}


    # combs = [[2, 2], [3, 3], [4, 2], [4, 4], [5, 5], [6, 3], [6, 6], [7, 7], [8, 2], [8, 4]]
    # combs_sing = [[1, 2], [1, 3], [2, 2], [1, 4], [1, 5], [2, 3], [1, 6], [1, 7], [4, 2], [2, 4]]

    combs = [[2, 2], [3, 3], [4, 2], [6, 3], [8, 2], [10, 2], [12, 2]]
    combs_sing = [[1, 2], [1, 3], [2, 2], [2, 3], [4, 2], [5, 2], [6, 2]]

    #steps = [21, 12, 10, 10, 7, 6, 6, 5, 5, 5]
    #steps_sing = [42, 42, 21, 42, 42, 21, 42, 42, 10, 21, 7, 6]
    steps = [21, 14, 13, 11, 7, 6, 5]

    #steps_micro = [21, 16, 13, 13, 10, 9, 9, 8, 7, 7]
    # steps_sing = [45, 45, 25, 45, 45, 25, 45, 45, 13, 25]

    wc_time = [35, 29, 29, 29, 29, 29, 20, 20, 20, 21, 23, 25]

    # targets.append([_ for _ in range(1, 12)])
    # targets.append([_ for _ in range(1, 10)])

    timestep = 1

    p_conn = [0.001, 0.01, 0.05, 0.09]
    #p_conn = [0.01, 0.1]

    n = int(10000 * timestep)

    if timestep == 1:
        ts = 0
    else:
        ts = 1

    with open("/localhome/g90604lp/more_neu/plasticity/" + str(int(n)) + "_neurons_" + str(timestep) + "_f.txt",
              "w") as fp:
        fp.write("Synaptic events simulations\n")

    for j in range(len(p_conn)):
        for i in range(len(combs_sing)):
            with open(
                    "/localhome/g90604lp/more_neu/plasticity/" + str(int(n)) + "_neurons_" + str(timestep) + "_f.txt",
                    "a") as fp:
                fp.write("[" + str(combs_sing[i][0]) + ", 1] " + str(n/steps[i]) + " " + str(combs_sing[i][1]) + " " + str(p_conn[j] * 100) + "\n")
                print("\n\n\n\n\n\n----------------RUNNING WITH " + str(combs_sing[i][0]) + " PARTITIONS, " + str(n/steps[i]) + " SPIKING NEURONS " + str(combs_sing[i][1]) + " TARGETS " + str(p_conn[j] * 100) + " PROB----------------\n\n\n\n\n\n")
                results = multipartition_test([combs_sing[i][0], 1], steps[i], timestep, n, p_conn[j], 1, 1, 128 * combs_sing[i][1], 100)
                fp.write(#str(results["dma_read"] * 0.005) + " " +
                         #str(results["state_update"] * 0.005) + " " +
                         #str(results["loop_time"] * 0.005) + " " +
                         str(results["syn_events"]) + " " +
                         str(results["syn_events_avg"]) + " " +
                         "\n")

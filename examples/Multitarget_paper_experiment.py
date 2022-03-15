import spynnaker8 as p
import math
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def multipartition_test(part, step, timestep, neurons, p_conn, n_targets, out_partitions, post_neurons):

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

    # n_target = -1 is a tmp flag
    input_population = p.Population(neurons, p.IF_curr_exp(**cell_params_lif_input),
                                    label='input', in_partitions=[1, 1], out_partitions=part[0], n_targets=n_targets)
    sel = [i * step for i in range(int(neurons/step))]
    input_population.set_initial_value("v", -40, sel)
    # 960 = 64 * 15
    destination_population = p.Population(post_neurons, p.IF_curr_exp(**cell_params_lif_dest),
                                          label='dest', in_partitions=part, out_partitions=out_partitions,
                                          n_targets=n_targets)

    p.Projection(input_population, destination_population, p.FixedProbabilityConnector(p_connect=p_conn),
                 p.StaticSynapse(weight=weight_to_spike, delay=delay))

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
        if tgt < part[0]:
            events_sum += syn_events[key]
            tgt += 1
    syn_avg = float(events_sum) / post_neurons

    ret_val_avg = {"dma_read": 0,
                   "state_update": 0,
                   "loop_time": 0,
                   "syn_events": events_sum,
                   "syn_events_avg": syn_avg,
                   "state_update_input": 0,
                   "loop_time_input": 0}

    return ret_val_avg


if __name__ == "__main__":

    values = {"dma_read": [],
              "state_update": [],
              "loop_time": [],
              "syn_events": [],
              "syn_events_avg": [],
              "state_update_input": [],
              "loop_time_input": []}

    partitions = []
    steps = []
    targets = []

    partitions.append([[i, 1] for i in range(2, 15)])
    partitions.append([[i, 1] for i in range(2, 9)])

    targets.append([_ for _ in range(1, 12)])
    targets.append([_ for _ in range(1, 10)])

    timestep = 0.1

    p_conn = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 0.99]

    steps.append([21, 12, 10, 7, 6, 5, 5, 5, 4, 4, 4, 3])
    steps.append([25, 16, 13, 10, 9, 8, 7, 5, 4, 4, 4, 3])

    n = 10000 * timestep

    if timestep == 1:
        ts = 0
    else:
        ts = 1

    with open("/localhome/g90604lp/multitarget_experiments/" + str(int(n)) + "_neurons_" + str(timestep) + ".txt",
              "w") as fp:
        fp.write("Synaptic events simulations\n")

    for i in range(len(partitions[ts])):
        for j in range(len(targets[ts])):
            if partitions[ts][i][0] + 1 + targets[ts][j] <= 15:
                with open(
                        "/localhome/g90604lp/multitarget_experiments/" + str(int(n)) + "_neurons_" + str(timestep) + ".txt",
                        "a") as fp:
                    fp.write(str(partitions[ts][i]) + " " + str(n/steps[ts][i]) +  " " + str(targets[ts][j]) + "\n")
                    print("\n\n\n\n\n\n----------------RUNNING WITH " + str(partitions[ts][i]) + " PARTITIONS, " + str(n/steps[ts][i+5]) + " SPIKING NEURONS " + str(targets[ts][j]) + " TARGETS----------------\n\n\n\n\n\n")
                    results = multipartition_test(partitions[ts][i], steps[ts][i], timestep, n, p_conn[1], targets[ts][j], 1, 64 * targets[ts][j])

                    fp.write(#str(results["dma_read"] * 0.005) + " " +
                             #str(results["state_update"] * 0.005) + " " +
                             #str(results["loop_time"] * 0.005) + " " +
                             str(results["syn_events"]) + " " +
                             str(results["syn_events_avg"]) + " " +
                             "\n")

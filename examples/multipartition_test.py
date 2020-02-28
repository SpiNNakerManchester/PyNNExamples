import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def multipartition_test(part, step, timestep):

    runtime = 3
    neurons = 10000
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

    weight_to_spike = 2
    delay = 0.2

    input_population = p.Population(neurons, p.IF_curr_exp(**cell_params_lif_input),
                                    label='input', in_partitions=part, out_partitions=part)
    sel = [i * step for i in range(neurons/step)]
    input_population.set_initial_value("v", -40, sel)
    destination_population = p.Population(64, p.IF_curr_exp(**cell_params_lif_dest),
                                          label='dest', in_partitions=part, out_partitions=1)

    p.Projection(input_population, destination_population, p.FixedProbabilityConnector(p_connect=0.1),
                 p.StaticSynapse(weight=weight_to_spike, delay=delay))

    destination_population.record(['gsyn_exc', 'gsyn_inh', 'synapse'])
    input_population.record(['gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    g_syn = destination_population.get_data('gsyn_exc')
    timings = destination_population.get_data('gsyn_inh')
    synaptic_events = destination_population.get_data('synapse')

    inputg_e = input_population.get_data('gsyn_exc')
    inputg_i = input_population.get_data('gsyn_inh')

    p.end()

    dma_read = list()
    state_update = list()
    loop_time = list()

    state_update_input = list()
    loop_time_input = list()

    for i in range(runtime):
        dma_read.append(int(timings.segments[0].filter(name='gsyn_inh')[0][i][0]))
        state_update.append(int(timings.segments[0].filter(name='gsyn_inh')[0][i][1]))
        state_update_input.append(int(inputg_i.segments[0].filter(name='gsyn_inh')[0][i][1]))
        avg_sum = 0
        avg_sum_i = 0
        for j in range(64):
            avg_sum += float(g_syn.segments[0].filter(name='gsyn_exc')[0][i][j])
            avg_sum_i += float(inputg_e.segments[0].filter(name='gsyn_exc')[0][i][j])
        loop_time.append(int(avg_sum / 64))
        loop_time_input.append(int(avg_sum_i / 64))

    syn_events = dict()
    for key in synaptic_events:
        syn_events[key] = int(synaptic_events[key][0])

    ret_val = {"dma_read": dma_read,
               "state_update": state_update,
               "loop_time": loop_time,
               "syn_events": syn_events}

    dma_avg = int(float(sum(dma_read)) / len(dma_read))
    state_avg = int(float(sum(state_update)) / len(state_update))
    loop_avg = int(float(sum(loop_time)) / len(loop_time))

    state_avg_input = int(float(sum(state_update_input)) / len(state_update_input))
    loop_avg_input = int(float(sum(loop_time_input)) / len(loop_time_input))

    events_sum = 0
    for key in syn_events:
        events_sum += syn_events[key]
    syn_avg = float(events_sum) / len(syn_events.keys())

    ret_val_avg = {"dma_read": dma_avg,
                   "state_update": state_avg,
                   "loop_time": loop_avg,
                   "syn_events": events_sum,
                   "syn_events_avg": syn_avg,
                   "state_update_input": state_avg_input,
                   "loop_time_input": loop_avg_input}

    return ret_val_avg


if __name__ == "__main__":

    values = {"dma_read": [],
              "state_update": [],
              "loop_time": [],
              "syn_events": [],
              "syn_events_avg": [],
              "state_update_input": [],
              "loop_time_input": []}

    partitions = [i for i in range(2, 14)]

    timestep = 1

    steps = [5]

    n = 10000

    for s in steps:

        with open("/localhome/g90604lp/ICPP_res/" + str(n/s) + "_neurons_" + str(timestep) + "_res_sopt.txt", "w") as fp:

            for part in partitions:

                fp.write(str(part) + " " + str(n/s) + "\n")
                print("\n\n\n\n\n\n----------------RUNNING WITH " + str(part) + " PARTITIONS, " + str(n/s) + " SPIKING NEURONS----------------\n\n\n\n\n\n")
                results = multipartition_test(part, s, timestep)

                fp.write(str(results["dma_read"] * 0.005) + " " +
                         str(results["state_update"] * 0.005) + " " +
                         str(results["loop_time"] * 0.005) + " " +
                         str(results["syn_events"]) + " " +
                         str(results["state_update_input"] * 0.005) + " " +
                         str(results["loop_time_input"] * 0.005) + "\n")

                values["dma_read"].append(results["dma_read"] * 0.005)
                values["state_update"].append(results["state_update"] * 0.005)
                values["loop_time"].append(results["loop_time"] * 0.005)
                values["syn_events"].append(results["syn_events"])
                values["syn_events_avg"].append(results["syn_events_avg"])
                values["state_update_input"].append(results["state_update_input"] * 0.005)
                values["loop_time_input"].append(results["loop_time_input"] * 0.005)

        plt.subplot(3, 2, 1)
        plt.plot(partitions, values["dma_read"], "o-")
        plt.title("DMA read timings")

        plt.subplot(3, 2, 2)
        plt.plot(partitions, values["syn_events"], "o-")
        plt.title("Synaptic events")

        plt.subplot(3, 2, 3)
        plt.plot(partitions, values["state_update"], "o-")
        plt.title("Neuron state update timings")

        plt.subplot(3, 2, 4)
        plt.plot(partitions, values["state_update_input"], "o-")
        plt.title("Input Neuron state update timings")

        plt.subplot(3, 2, 5)
        plt.plot(partitions, values["loop_time"], "o-")
        plt.title("Synaptic contribution sum timing")

        plt.subplot(3, 2, 6)
        plt.plot(partitions, values["loop_time_input"], "o-")
        plt.title("Input Synaptic contribution sum timing")
        plt.xlabel('number of synapse cores')

        plt.show()





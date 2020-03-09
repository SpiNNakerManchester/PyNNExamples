import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def multipartition_test(part, step, timestep, neurons):

    runtime = 3*timestep
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
    delay = 0.1

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

    #for i in range(runtime):
        #dma_read.append(int(timings.segments[0].filter(name='gsyn_inh')[0][i][0]))
        #state_update.append(int(timings.segments[0].filter(name='gsyn_inh')[0][i][1]))
        #state_update_input.append(int(inputg_i.segments[0].filter(name='gsyn_inh')[0][i][1]))
        #avg_sum = 0
        #avg_sum_i = 0
        #for j in range(64):
        #    avg_sum += float(g_syn.segments[0].filter(name='gsyn_exc')[0][i][j])
        #    avg_sum_i += float(inputg_e.segments[0].filter(name='gsyn_exc')[0][i][j])
        #loop_time.append(int(avg_sum / 64))
        #loop_time_input.append(int(avg_sum_i / 64))

    syn_events = dict()
    for key in synaptic_events:
        syn_events[key] = int(synaptic_events[key][0])

    ret_val = {"dma_read": dma_read,
               "state_update": state_update,
               "loop_time": loop_time,
               "syn_events": syn_events}

    #dma_avg = int(float(sum(dma_read)) / len(dma_read))
    #state_avg = int(float(sum(state_update)) / len(state_update))
    #loop_avg = int(float(sum(loop_time)) / len(loop_time))

    #state_avg_input = int(float(sum(state_update_input)) / len(state_update_input))
    #loop_avg_input = int(float(sum(loop_time_input)) / len(loop_time_input))

    events_sum = 0
    for key in syn_events:
        events_sum += syn_events[key]
    syn_avg = float(events_sum) / len(syn_events.keys())

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

    partitions = [i for i in range(13, 14)]

    timestep = 1

    steps = [4]

    n = 10000

    with open("/localhome/g90604lp/ICPP_res/" + str(n) + "_neurons_" + str(timestep) + "_res_sopt(4).txt", "w") as fp:

        for i in range(len(partitions)):

            fp.write(str(partitions[i]) + " " + str(n/steps[i]) + "\n")
            print("\n\n\n\n\n\n----------------RUNNING WITH " + str(partitions[i]) + " PARTITIONS, " + str(n/steps[i]) + " SPIKING NEURONS----------------\n\n\n\n\n\n")
            results = multipartition_test(partitions[i], steps[i], timestep, n)

            fp.write(#str(results["dma_read"] * 0.005) + " " +
                     #str(results["state_update"] * 0.005) + " " +
                     #str(results["loop_time"] * 0.005) + " " +
                     str(results["syn_events"]) + " " +
                     #str(results["state_update_input"] * 0.005) + " " +
                     "\n")




import spynnaker8 as pynn
# from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager
from eprop_testing.shd_testing.plot_shd_graph import draw_graph_from_list, plot_learning_curve
from eprop_testing.shd_testing.incremental_shd_config import *
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

def load_connections(npy_label, pop_size, connections=[], rec=True):
    if npy_label:
        in_conn = [list(ele) for ele in np.load(npy_label+' in.npy').tolist()]
        if rec:
            rec_conn = [list(ele) for ele in np.load(npy_label+' rec.npy').tolist()]
        out_conn = [list(ele) for ele in np.load(npy_label+' out.npy').tolist()]
    else:
        [in_conn, rec_conn, out_conn] = connections
    new_in_conn = []
    for ndx in range(len(in_conn)):
        if in_conn[ndx][3] == 16 and in_conn[ndx][0] == 0:
            in_conn[ndx][3] = 0
        new_in_conn.append([in_conn[ndx][0], in_conn[ndx][1], in_conn[ndx][2], in_conn[ndx][3]])
    new_rec_conn = []
    if rec:
        for ndx in range(len(rec_conn)):
            if rec_conn[ndx][3] == 16 and rec_conn[ndx][0] == 0:
                rec_conn[ndx][3] = 0
            new_rec_conn.append([rec_conn[ndx][0], rec_conn[ndx][1], rec_conn[ndx][2], rec_conn[ndx][3]])
    new_out_conn = []
    for ndx in range(len(out_conn)):
        if out_conn[ndx][3] == 16 and out_conn[ndx][0] == 0:
            out_conn[ndx][3] = 0
        new_out_conn.append([out_conn[ndx][0], out_conn[ndx][1], out_conn[ndx][2], out_conn[ndx][3]])
    checking_delays = [[] for i in range(pop_size)]
    list_to_check = new_in_conn
    if rec:
        list_to_check = new_in_conn+new_rec_conn
    for [pre, post, weight, delay] in list_to_check:
        if delay not in checking_delays[post]:
            checking_delays.append(delay)
        else:
            print("delays are overlapped")
            Exception
    return new_in_conn, new_rec_conn, new_out_conn

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size) #+ 0.5
    # base_weight = 0
    return base_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0, base_weight=0.0, rec_conn=False):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob and not (rec_conn and i == j):
                neuron_syn_count += 1
                conn = [i, j, weight_distribution(pre_pop_size)+base_weight, delay_count]
                delay_count += 1
                connections.append(conn)
        if neuron_syn_count > max_syn_per_neuron:
            max_syn_per_neuron = neuron_syn_count
    return connections, max_syn_per_neuron

def range_connector(pre_min, pre_max, post_min, post_max, weight=1.5, delay_offset=0):
    connections = []
    for j in range(int(post_min), int(post_max)):
        # delay = delay_offset
        for i in range(int(pre_min), int(pre_max)):
            nd_weight = weight_distribution(pre_max-pre_min)
            connections.append([i, j, weight+nd_weight, i+delay_offset])
            # delay += 1
    return connections

def collect_tests_and_labels(selected_classes):
    test_idx = [i for i in range(min(max_tests, 399*len(selected_classes)))]
    new_spikes = [[] for i in range(700)]
    new_labels = [-1 for i in range(len(test_idx))]
    np.random.shuffle(test_idx)
    count = -1
    for spike, label in zip(spike_times, labels):
        if label in selected_classes:
            count += 1
            new_labels[test_idx[count]] = label
            for idx, neuron in enumerate(spike):
                new_spikes[idx] = new_spikes[idx] + list(map(lambda x: x + (1000*test_idx[count]), neuron))
        if count+1 >= max_tests or count+1 >= len(test_idx):
            break
    return new_labels, new_spikes

def get_all_weights(in_proj, from_list_in, out_proj, from_list_out,
                    recurrent_proj, from_list_rec, layer_proj):
    new_connections_in = []  # in_proj.get('weight', 'delay').connections[0]#[]
    for partition in in_proj.get('weight', 'delay').connections:
        for conn in partition:
            new_connections_in.append(conn)
    new_connections_in.sort(key=lambda x: x[0])
    new_connections_in.sort(key=lambda x: x[1])
    from_list_in.sort(key=lambda x: x[0])
    from_list_in.sort(key=lambda x: x[1])
    connection_diff_in = []
    for i in range(len(from_list_in)):
        connection_diff_in.append(new_connections_in[i][2] - from_list_in[i][2])
    print("Input connections\noriginal\n", np.array(from_list_in))
    print("new\n", np.array(new_connections_in))
    print("diff\n", np.array(connection_diff_in))

    new_connections_out = []  # out_proj.get('weight', 'delay').connections[0]#[]
    for partition in out_proj.get('weight', 'delay').connections:
        for conn in partition:
            new_connections_out.append(conn)
    new_connections_out.sort(key=lambda x: x[0])
    new_connections_out.sort(key=lambda x: x[1])
    from_list_out.sort(key=lambda x: x[0])
    from_list_out.sort(key=lambda x: x[1])
    connection_diff_out = []
    for i in range(len(from_list_out)):
        connection_diff_out.append(new_connections_out[i][2] - from_list_out[i][2])
    print("Output connections\noriginal\n", np.array(from_list_out))
    print("new\n", np.array(new_connections_out))
    print("diff\n", np.array(connection_diff_out))

    new_connections_rec = []
    if recurrent_connections:
        for partition in recurrent_proj.get('weight', 'delay').connections:
            for conn in partition:
                new_connections_rec.append(conn)
        new_connections_rec.sort(key=lambda x: x[0])
        new_connections_rec.sort(key=lambda x: x[1])
        from_list_rec.sort(key=lambda x: x[0])
        from_list_rec.sort(key=lambda x: x[1])
        connection_diff_rec = []
        for i in range(len(from_list_rec)):
            connection_diff_rec.append(new_connections_rec[i][2] - from_list_rec[i][2])
        print("Recurrent connections\noriginal\n", np.array(from_list_rec))
        print("new\n", np.array(new_connections_rec))
        print("diff\n", np.array(connection_diff_rec))

    new_connections_layer = [[] for i in range(layers)]
    for lay_ndx, layer in enumerate(layer_proj):
        for partition in layer.get('weight', 'delay').connections:
            for conn in partition:
                new_connections_layer[lay_ndx].append(conn)
    return new_connections_in, new_connections_out, new_connections_rec, new_connections_layer

def print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                 confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeat=0):
    # print(cycle_error)
    print(experiment_label)
    print(experiment_label)
    # print("cycle_error =", cycle_error)
    print("correct or not = ", correct_or_not)
    print("\\", "|\t", end="")
    for i in range(output_size):
        print("{:5}\t|\t".format(i), end="")
    print("")
    class_count = 0
    for i, test_label in enumerate(confusion_matrix):
        print(class_count, "|\t", end="")
        for j, choice in enumerate(test_label):
            if i == j:
                prCyan("{:4}\t|\t".format(round(choice, 3)), end="")
            else:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
        print("")
        class_count += 1
    print("")
    print("\\", "|\t", end="")
    for i in range(output_size):
        print("{:5}\t|\t".format(i), end="")
    print("")
    class_count = 0
    for i, test_label in enumerate(final_confusion_matrix):
        print(class_count, "|\t", end="")
        for j, choice in enumerate(test_label):
            if i == j:
                prCyan("{:4}\t|\t".format(round(choice, 3)), end="")
            else:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
        print("")
        class_count += 1
    for i in range(int(len(correct_or_not)/window_cycles)):
        if i*window_cycles in cue_break:
            print("increased classes")
        print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
              np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]),
              '({})'.format(np.average(correct_or_not[max(((i + 1) * window_cycles)-100, 0):(i + 1) * window_cycles])))
    print("average classification = ", np.average(correct_or_not))
    print("weighted average classification = ", np.average(correct_or_not,
                                                           weights=[i for i in range(len(correct_or_not))]))
    print("classification = ", test_classification)
    print(experiment_label)
    print(cue_break)
    print("current classes = ", len(cue_break) + no_class_start)
    print("average error = ", np.average(cycle_error))
    print("weighted average", np.average(cycle_error, weights=[i for i in range(len(cycle_error))]))
    print("minimum error = ", np.min(cycle_error))
    print("minimum iteration = ", cycle_error.index(np.min(cycle_error)), "- with time stamp =",
          cycle_error.index(np.min(cycle_error)) * 1024)
    print("iteration: ", current_window*window_cycles, "/", len(new_labels), " - repeat #", repeat)

def first_create_pops():
    new_labels, new_spikes = collect_tests_and_labels(class_order[:no_class_start])

    pynn.setup(timestep=1)
    input_pop = pynn.Population(input_size,
                                pynn.SpikeSourceArray,
                                {'spike_times': new_spikes},
                                label='input_pop')
    readout_neuron_params["target_data"] = new_labels
    # Output population
    readout_pop = pynn.Population(output_size,  # HARDCODED 1
                                  pynn.extra_models.SHDReadout(
                                      **readout_neuron_params
                                  ),
                                  label="readout_pop"
                                  )

    eprop_learning_output = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=0.0))
    neuron = []
    for i in range(layers):
        neuron.append(pynn.Population(neuron_pop_size,
                                      pynn.extra_models.EPropAdaptive(**neuron_params),
                                      label='eprop_pop{}'.format(i)))
    eprop_learning_neuron = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    from_list_in, max_syn_per_input = probability_connector(input_size, neuron_pop_size, p_connect_in,
                                                            base_weight=base_weight_in)
    in_proj = pynn.Projection(input_pop,
                              neuron[0],
                              pynn.FromListConnector(from_list_in),
                              synapse_type=eprop_learning_neuron,
                              label='input_connections',
                              receptor_type='input_connections')

    # from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
    from_list_out, max_syn_per_output = probability_connector(neuron_pop_size, output_size, p_connect_out,
                                                              base_weight=base_weight_out)
    if max_syn_per_output > 100:
        Exception
    else:
        print("max number of synapses per readout:", max_syn_per_output)
    out_proj = pynn.Projection(neuron[-1],
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')
    learning_proj = []
    layer_proj = []
    for i in range(layers):
        if i > 0:
            from_list_l, max_syn_per_output = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec,
                                                                    base_weight=base_weight_in*(700/neuron_pop_size),
                                                                    rec_conn=True)
            layer_proj.append(pynn.Projection(neuron[i - 1],
                                              neuron[i],
                                              # pynn.OneToOneConnector(),
                                              pynn.FromListConnector(from_list_l),
                                              synapse_type=eprop_learning_output,
                                              label='input_connections',
                                              receptor_type='input_connections'))
        if record_data:
            neuron[i].record('all')
        learning_proj.append(pynn.Projection(readout_pop,
                                             neuron[i],
                                             # pynn.OneToOneConnector(),
                                             # pynn.StaticSynapse(weight=[0.5], delay=[0]),
                                             pynn.AllToAllConnector(),
                                             pynn.StaticSynapse(weight=0.5, delay=0),
                                             receptor_type='learning_signal'))

    if recurrent_connections:
        eprop_learning_recurrent = pynn.STDPMechanism(
            timing_dependence=pynn.extra_models.TimingDependenceEprop(),
            weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
                w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

        from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size,
                                                               neuron_pop_size,
                                                               p_connect_rec,
                                                               offset=input_size,
                                                               base_weight=base_weight_rec)
        recurrent_proj = pynn.Projection(neuron[0],
                                         neuron[0],
                                         pynn.FromListConnector(from_list_rec),
                                         synapse_type=eprop_learning_recurrent,
                                         label='recurrent_connections',
                                         receptor_type='recurrent_connections')
    else:
        from_list_rec = []
        recurrent_proj = None
    if record_data:
        input_pop.record('spikes')
    # neuron.record('spikes')
    # neuron.record(['gsyn_exc', 'v', 'gsyn_inh'],
    #               indexes=[i for i in range(int((neuron_pop_size / 2) - 5), int((neuron_pop_size / 2) + 5))])
    readout_pop.record(['gsyn_exc', 'v', 'gsyn_inh'])

    runtime = cycle_time * num_repeats

    experiment_label = "{}eta({}) h{}o{} - b{}-{} - tr{}-{} - vmem{} - fb{} - in{} out{} rec{}{} ({}x{})".format(
        free_label, batch_size, neuron_params["eta"], readout_neuron_params["eta"], threshold_beta, ratio_of_LIF,
        neuron_params["target_rate"], neuron_params["firing_lr"], neuron_params["v_mem_lr"], w_fb_type,
        base_weight_in, base_weight_out, base_weight_rec, recurrent_connections, layers, neuron_pop_size)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj, \
           input_pop, neuron, readout_pop, from_list_in, from_list_rec, from_list_out, new_labels


def next_create_pops(from_list_in, from_list_rec, from_list_lay, from_list_out, no_classes):
    new_labels, new_spikes = collect_tests_and_labels(class_order[:no_classes])

    from_list_in, from_list_rec, from_list_out = \
        load_connections(None, neuron_pop_size,
                         connections=[from_list_in, from_list_rec, from_list_out],
                         rec=recurrent_connections)

    pynn.setup(timestep=1)
    input_pop = pynn.Population(input_size,
                                pynn.SpikeSourceArray,
                                {'spike_times': new_spikes},
                                label='input_pop')
    readout_neuron_params["target_data"] = new_labels
    # Output population
    readout_pop = pynn.Population(output_size,  # HARDCODED 1
                                  pynn.extra_models.SHDReadout(
                                      **readout_neuron_params
                                  ),
                                  label="readout_pop"
                                  )

    eprop_learning_output = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=0.0))
    neuron = []
    for i in range(layers):
        neuron.append(pynn.Population(neuron_pop_size,
                                      pynn.extra_models.EPropAdaptive(**neuron_params),
                                      label='eprop_pop{}'.format(i)))
    eprop_learning_neuron = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    in_proj = pynn.Projection(input_pop,
                              neuron[0],
                              pynn.FromListConnector(from_list_in),
                              synapse_type=eprop_learning_neuron,
                              label='input_connections',
                              receptor_type='input_connections')

    out_proj = pynn.Projection(neuron[-1],
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')
    learning_proj = []
    layer_proj = []
    for i in range(layers):
        if i > 0:
            layer_proj.append(pynn.Projection(neuron[i - 1],
                                              neuron[i],
                                              # pynn.OneToOneConnector(),
                                              pynn.FromListConnector(from_list_lay[i-1]),
                                              synapse_type=eprop_learning_output,
                                              label='input_connections',
                                              receptor_type='input_connections'))
        if record_data:
            neuron[i].record('all')
        learning_proj.append(pynn.Projection(readout_pop,
                                             neuron[i],
                                             # pynn.OneToOneConnector(),
                                             # pynn.StaticSynapse(weight=[0.5], delay=[0]),
                                             pynn.AllToAllConnector(),
                                             pynn.StaticSynapse(weight=0.5, delay=0),
                                             receptor_type='learning_signal'))

    if recurrent_connections:
        eprop_learning_recurrent = pynn.STDPMechanism(
            timing_dependence=pynn.extra_models.TimingDependenceEprop(),
            weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
                w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))
        recurrent_proj = pynn.Projection(neuron[0],
                                         neuron[0],
                                         pynn.FromListConnector(from_list_rec),
                                         synapse_type=eprop_learning_recurrent,
                                         label='recurrent_connections',
                                         receptor_type='recurrent_connections')
    else:
        from_list_rec = []
        recurrent_proj = None
    if record_data:
        input_pop.record('spikes')
    # neuron.record('spikes')
    # neuron.record(['gsyn_exc', 'v', 'gsyn_inh'],
    #               indexes=[i for i in range(int((neuron_pop_size / 2) - 5), int((neuron_pop_size / 2) + 5))])
    readout_pop.record(['gsyn_exc', 'v', 'gsyn_inh'])

    runtime = cycle_time * num_repeats

    experiment_label = "{}eta({}) h{}o{} - b{}-{} - tr{}-{} - vmem{} - fb{} - in{} out{} rec{}{} ({}x{})".format(
        free_label, batch_size, neuron_params["eta"], readout_neuron_params["eta"], threshold_beta, ratio_of_LIF,
        neuron_params["target_rate"], neuron_params["firing_lr"], neuron_params["v_mem_lr"], w_fb_type,
        base_weight_in, base_weight_out, base_weight_rec, recurrent_connections, layers, neuron_pop_size)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj, \
           input_pop, neuron, readout_pop, from_list_in, from_list_rec, from_list_out, new_labels

def run_until(experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj,
              input_pop, neuron, readout_pop,
              from_list_in, from_list_rec, from_list_out,
              correct_or_not, cycle_error, confusion_matrix, new_labels,
              threshold=0.95, cue_break=[], final_class=False, repeating=0):
    good_performance = False
    current_window = 0
    current_iter = current_window * window_cycles
    # window_size = neuron_params["window_size"] * window_cycles
    runtime = cycle_time * len(readout_neuron_params["target_data"])

    final_confusion_matrix = [[0. for i in range(output_size)] for j in range(output_size)]
    while (current_window+1) * cycle_time * window_cycles < runtime:
        print(experiment_label)
        pynn.run(cycle_time*window_cycles)
        readout_res = readout_pop.get_data(['gsyn_exc', 'v', 'gsyn_inh'])#, clear=True)

        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                confusion_matrix[i][j] *= long_confusion_decay
                final_confusion_matrix[i][j] *= short_confusion_decay
        test_classification = []
        for cycle in range(window_cycles):
            cycle_error.append(0.0)
            correct_or_not.append([])
            cycle_classification = [-1 for i in range(cycle_time)]
            ce = [0.0 for i in range(output_size)]
            for time_index in range(cycle_time):
                instantaneous_error = np.abs(float(
                    readout_res.segments[0].filter(name='gsyn_inh')[0][time_index + ((cycle+current_iter) * cycle_time)][0]))
                cycle_error[-1] += instantaneous_error
                softmaxes = [0.0 for i in range(output_size)]
                for n_out in range(output_size):
                    ce[n_out] += float(readout_res.segments[0].filter(name='v')[0][time_index + ((cycle+current_iter) * cycle_time)][n_out])
                    # ce[n_out] = np.exp(max(min(8.75, v_mem), -8.75))
                # ce_sum = sum(ce)
                # for i in range(len(ce)):
                #     softmaxes[i] += ce[i] / ce_sum
            test_classification.append(
                [new_labels[cycle+(current_window*window_cycles)], ce.index(max(ce))])  # mode
            # print("current label = ", cycle+(current_window*window_cycles))
            confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
            final_confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
            correct_or_not[-1] = int(test_classification[-1][0] == test_classification[-1][1])

        current_window += 1
        current_iter = current_window * window_cycles

        print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                     confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeating)

        if current_iter >= 40:
            print(cue_break)
        if current_iter >= 100:
            print(cue_break)
        if current_iter >= 200:
            print(cue_break)
        if current_iter >= 400:
            print(cue_break)
        if current_iter >= 40 and repeating == 1:
            print(cue_break)
        if current_iter >= 40 and repeating == 2:
            print(cue_break)
        if current_iter >= 40 and repeating == 3:
            print(cue_break)
        if current_iter >= 40 and repeating == 5:
            print(cue_break)
        if current_iter >= 40 and current_iter <= 100:
            print(cue_break)
        if current_iter >= 100 and repeating:
            print(cue_break)
        if current_iter >= 200 and repeating:
            print(cue_break)
        if current_iter >= 400 and repeating:
            print(cue_break)
        if current_iter >= 2360:
            print(cue_break)

        if record_data and current_window == 1:
            in_spikes = input_pop.get_data('spikes')  # , clear=True)
            neuron_res = neuron[0].get_data(['spikes', 'gsyn_inh', 'gsyn_exc', 'v'])  # , clear=True)
            plot_time = current_iter * cycle_time
            start_time = max(plot_time - cycle_time * 3, 0)
            end_time = plot_time
            # plt.figure()
            Figure(

                Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].spiketrains, ylabel='early neuron_spikes', xlabel='early neuron_spikes',
                      yticks=True,
                      xticks=True, xlim=(0, plot_time - start_time)),

                Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes',
                      yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                title="neuron data for {}".format(experiment_label)
            )
            # plt.show()
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(16, 9)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig('./../shd_graphs/neuron data ' + experiment_label + " {} {}.png".format(cue_break, repeating))
                        # , bbox_inches='tight')

        if plot_membranes:
            # graph_directory = './../shd_graphs/'
            graph_directory = '/data/mbaxrap7/shd_graphs/'
            plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                                output_size,
                                graph_directory, 'lc ' + experiment_label + ' {}'.format(int(len(correct_or_not) / window_cycles)),
                                save_flag=True,
                                cue_break=cue_break,
                                plot_flag=False,
                                learning_threshold=learning_threshold,
                                no_classes=no_class_start + (len(cue_break) * class_progress))

        if np.sum(correct_or_not[-min(100, current_iter+(repeating*len(new_labels))):]) > threshold * 100 \
                and not final_class:
        # if current_iter > 0 and \
        #         np.average(correct_or_not[-10:]) > threshold:
            new_connections_in, new_connections_out, new_connections_rec, new_connections_layer = \
                get_all_weights(in_proj, from_list_in, out_proj, from_list_out,
                                recurrent_proj, from_list_rec, layer_proj)
            print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                         confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeating)
            graph_directory = './../shd_graphs/'
            # graph_directory = '/data/mbaxrap7/shd_graphs/'
            plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                                output_size,
                                graph_directory, 'pass ' + experiment_label + ' {}'.format(current_window),
                                save_flag=True,
                                cue_break=cue_break,
                                plot_flag=False,
                                learning_threshold=learning_threshold,
                                no_classes=no_class_start + (len(cue_break) * class_progress))

            pynn.end()
            print(cycle_error)
            print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                         confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeating)
            print("Simulation has achieved threshold performance at window:", current_window)
            print("this corresponds to itertation:", current_iter)
            print(experiment_label)
            good_performance = len(correct_or_not)
            return new_connections_in, new_connections_rec, new_connections_layer, new_connections_out, \
                   correct_or_not, cycle_error, good_performance

    new_connections_in, new_connections_out, new_connections_rec, new_connections_layer = \
        get_all_weights(in_proj, from_list_in, out_proj, from_list_out,
                        recurrent_proj, from_list_rec, layer_proj)
    print_status(current_window, experiment_label, cycle_error, test_classification, correct_or_not,
                 confusion_matrix, final_confusion_matrix, new_labels, cue_break, repeating)
    graph_directory = './../shd_graphs/'
    plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                        output_size,
                        graph_directory, 'fail ' + experiment_label + ' {}'.format(current_window),
                        save_flag=True,
                        cue_break=cue_break,
                        plot_flag=False,
                        learning_threshold=learning_threshold,
                        no_classes=no_class_start + (len(cue_break) * class_progress))
    pynn.end()
    print("Learning has failed to achieve threshold performance in runtime")
    return new_connections_in, new_connections_rec, new_connections_layer, new_connections_out, \
           correct_or_not, cycle_error, good_performance


def prCyan(skk, end='/n'):
    print("\033[96m {}\033[00m" .format(skk), end=end)
import spynnaker8 as pynn
# from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager
from eprop_testing.shd_testing.plot_shd_graph import draw_graph_from_list, plot_learning_curve
from eprop_testing.shd_testing.incremental_shd_config import *
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

def load_connections(npy_label, pop_size, rec=True):
    in_conn = [list(ele) for ele in np.load(npy_label+' in.npy').tolist()]
    if rec:
        rec_conn = [list(ele) for ele in np.load(npy_label+' rec.npy').tolist()]
    out_conn = [list(ele) for ele in np.load(npy_label+' out.npy').tolist()]
    for ndx in range(len(in_conn)):
        if in_conn[ndx][3] == 16 and in_conn[ndx][0] == 0:
            in_conn[ndx][3] = 0
    if rec:
        for ndx in range(len(rec_conn)):
            if rec_conn[ndx][3] == 16 and rec_conn[ndx][0] == 0:
                rec_conn[ndx][3] = 0
    for ndx in range(len(out_conn)):
        if out_conn[ndx][3] == 16 and out_conn[ndx][0] == 0:
            out_conn[ndx][3] = 0
    checking_delays = [[] for i in range(pop_size)]
    list_to_check = in_conn
    if rec:
        list_to_check = in_conn+rec_conn
    for [pre, post, weight, delay] in list_to_check:
        if delay not in checking_delays[post]:
            checking_delays.append(delay)
        else:
            print("delays are overlapped")
            Exception
    if not rec:
        rec_conn = []
    return in_conn, rec_conn, out_conn

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size) #+ 0.5
    # base_weight = 0
    return base_weight

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0, base_weight=0.0):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
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
    new_labels = []
    new_spikes = [[] for i in range(700)]
    count = -1
    for spike, label in zip(spike_times, labels):
        if label in selected_classes:
            count += 1
            new_labels.append(label)
            for idx, neuron in enumerate(spike):
                new_spikes[idx] = new_spikes[idx] + list(map(lambda x: x + (1000*count), neuron))
    return new_labels, new_spikes

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
                                                                    base_weight=base_weight_in)
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
                                                               offset=0,
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

    experiment_label = "base_w in{} out{} rec{}{} ({}x{}) eta h{}o{} - b{}-{} - w_fb{}".format(
        base_weight_in, base_weight_out, base_weight_rec, recurrent_connections,
        layers, neuron_pop_size, neuron_params["eta"], readout_neuron_params["eta"],
        threshold_beta, ratio_of_LIF, forced_w_fb)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj, \
           input_pop, neuron, readout_pop, from_list_in, from_list_rec, from_list_out, new_labels


def next_create_pops(from_list_in, from_list_rec, from_list_lay, from_list_out, no_classes):
    new_labels, new_spikes = collect_tests_and_labels(class_order[:no_classes])

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
    readout_pop.record('gsyn_exc', 'v', 'gsyn_inh')

    runtime = cycle_time * num_repeats

    experiment_label = "base_w in{} out{} rec{}{} ({}x{}) eta h{}o{} - b{}-{} - w_fb{}".format(
        base_weight_in, base_weight_out, base_weight_rec, recurrent_connections,
        layers, neuron_pop_size, neuron_params["eta"], readout_neuron_params["eta"],
        threshold_beta, ratio_of_LIF, forced_w_fb)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj, \
           input_pop, neuron, readout_pop, from_list_in, from_list_rec, from_list_out, new_labels

def run_until(experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj,
              input_pop, neuron, readout_pop,
              from_list_in, from_list_rec, from_list_out,
              correct_or_not, cycle_error, confusion_matrix, new_labels,
              threshold=0.95, cue_break=[]):
    good_performance = False
    current_window = 0
    current_iter = current_window * window_cycles
    window_size = neuron_params["window_size"] * window_cycles
    runtime = cycle_time * len(readout_neuron_params["target_data"])

    while (current_window+1) * window_size < runtime:
        print(experiment_label)
        pynn.run(window_size)
        if record_data:
            in_spikes = input_pop.get_data('spikes', clear=True)
            neuron_res = neuron[0].get_data('all', clear=True)
        readout_res = readout_pop.get_data('all', clear=True)

        final_confusion_matrix = [[0. for i in range(output_size)] for i in range(output_size)]
        test_classification = []
        for cycle in range(window_cycles):
            cycle_error.append(0.0)
            correct_or_not.append([])
            cycle_classification = [-1 for i in range(cycle_time)]
            for time_index in range(cycle_time):
                instantaneous_error = np.abs(float(
                    readout_res.segments[0].filter(name='gsyn_inh')[0][time_index + ((cycle+current_iter) * cycle_time)][0]))
                cycle_error[-1] += instantaneous_error
                voltages = [0.0 for i in range(output_size)]
                for n_out in range(output_size):
                    v_mem = np.abs(
                        float(readout_res.segments[0].filter(name='v')[0][time_index + ((cycle+current_iter) * cycle_time)][n_out]))
                    voltages[n_out] = v_mem
                cycle_classification[time_index] = voltages.index(max(voltages))
            test_classification.append(
                [new_labels[cycle+(current_window*window_cycles)], max(set(cycle_classification), key=cycle_classification.count)])  # mode
            confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
            final_confusion_matrix[test_classification[-1][0]][test_classification[-1][1]] += 1
            correct_or_not[-1] = int(test_classification[-1][0] == test_classification[-1][1])

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
        for layer in layer_proj:
            for partition in layer.get('weight', 'delay').connections:
                for conn in partition:
                    new_connections_layer[layer].append(conn)

        current_window += 1
        current_iter = current_window * window_cycles

        print(cycle_error)
        for i in range(current_window):
            print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
                  np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]))
        print(experiment_label)

        print(experiment_label)
        print("cycle_error =", cycle_error)
        print("classification = ", test_classification)
        print("correct or not = ", correct_or_not)
        print("\\", "|\t", end="")
        for i in range(output_size):
            print("{:5}\t|\t".format(i), end="")
        print("")
        class_count = 0
        for test_label in confusion_matrix:
            print(class_count, "|\t", end="")
            for choice in test_label:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
            print("")
            class_count += 1
        print("")
        print("\\", "|\t", end="")
        for i in range(output_size):
            print("{:5}\t|\t".format(i), end="")
        print("")
        class_count = 0
        for test_label in final_confusion_matrix:
            print(class_count, "|\t", end="")
            for choice in test_label:
                print("{:5}\t|\t".format(round(choice, 3)), end="")
            print("")
            class_count += 1
        print("average classification = ", np.average(correct_or_not))
        print("weighted average classification = ", np.average(correct_or_not,
                                                               weights=[i for i in range(len(correct_or_not))]))
        print(experiment_label)
        print("average error = ", np.average(cycle_error))
        print("weighted average", np.average(cycle_error, weights=[i for i in range(len(cycle_error))]))
        print("minimum error = ", np.min(cycle_error))
        print("minimum iteration = ", cycle_error.index(np.min(cycle_error)), "- with time stamp =",
              cycle_error.index(np.min(cycle_error)) * 1024)
        print("iteration: ", len(correct_or_not), "/", len(new_labels))

        if record_data and plot_membranes:
            plot_time = current_iter * cycle_time
            start_time = plot_time - cycle_time * 10
            end_time = plot_time
            plt.figure()
            Figure(
                Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True,
                      xlim=(start_time, end_time)),

                Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True,
                      xticks=True, xlim=(start_time, end_time)),

                # Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes',
                #       yticks=True,
                #       xticks=True, xlim=(0, plot_time - start_time)),

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
            plt.show()

        graph_directory = './../shd_graphs/'

        plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                            output_size,
                            graph_directory, experiment_label + ' {}'.format(current_window),
                            save_flag=False,
                            cue_break=cue_break,
                            plot_flag=False)

        if current_iter > 64 and \
                np.average(correct_or_not[-64:]) > threshold:
            pynn.end()
            print(cycle_error)
            for i in range(current_window):
                print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
                      np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]))
            print("Simulation has achieved threshold performance at window:", current_window)
            print("this corresponds to itertation:", current_iter)
            print(experiment_label)
            good_performance = len(correct_or_not)
            plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                                output_size,
                                graph_directory, experiment_label + ' {}'.format(current_window),
                                save_flag=True,
                                cue_break=cue_break,
                                plot_flag=True)
            return new_connections_in, new_connections_rec, new_connections_layer, new_connections_out, \
                   correct_or_not, cycle_error, good_performance

    pynn.end()
    print("Learning has failed to achieve threshold performance in runtime")
    return new_connections_in, new_connections_rec, new_connections_layer, new_connections_out, \
           correct_or_not, cycle_error, good_performance

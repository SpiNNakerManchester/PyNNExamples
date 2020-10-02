import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from PyNN8Examples.eprop_testing.frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager
from PyNN8Examples.eprop_testing.plot_graph import draw_graph_from_list, plot_learning_curve
from PyNN8Examples.eprop_testing.incremental_config import *

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

def probability_connector(pre_pop_size, post_pop_size, prob, offset=0):
    connections = []
    max_syn_per_neuron = 0
    for j in range(post_pop_size):
        neuron_syn_count = 0
        delay_count = offset
        for i in range(pre_pop_size):
            if np.random.random() < prob:
                neuron_syn_count += 1
                conn = [i, j, weight_distribution(pre_pop_size), delay_count]
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


def first_create_pops():
    pynn.setup(timestep=1)
    input_pop = pynn.Population(input_size,
                                pynn.SpikeSourcePoisson(rate=rates),
                                # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                                # {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, input_split, input_split, input_size)},
                                label='input_pop')
    neuron = pynn.Population(neuron_pop_size,
                             pynn.extra_models.EPropAdaptive(**neuron_params),
                             label='eprop_pop')

    # Output population
    readout_pop = pynn.Population(3,  # HARDCODED 1
                                  pynn.extra_models.LeftRightReadout(
                                      **readout_neuron_params
                                  ),
                                  label="readout_pop"
                                  )

    SpynnakerExternalDevicePluginManager.add_edge(readout_pop._get_vertex, input_pop._get_vertex, "CONTROL")

    eprop_learning_neuron = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    # from_list_in, max_syn_per_input = probability_connector(input_size, neuron_pop_size, p_connect_in)
    ps = int(readout_neuron_params["poisson_pop_size"])
    # from_list_in = range_connector(0, ps, 0, neuron_pop_size/2, weight=in_weight)  # connect 1/2st 2 left
    # from_list_in += range_connector(ps, ps*2, neuron_pop_size/2, neuron_pop_size, weight=in_weight)  # connect 2/2nd 2 right
    # from_list_in = range_connector(0, ps*2, 0, neuron_pop_size, weight=in_weight)  # connect all cues to pop
    # from_list_in += range_connector(ps*2, ps*3, 0, neuron_pop_size, delay_offset=0, weight=prompt_weight)  # connect all 2 prompt
    from_list_in = range_connector(0, ps * 2, 0, neuron_pop_size, delay_offset=0,
                                   weight=in_weight)  # connect all 2 prompt
    from_list_in += range_connector(ps * 2, ps * 4, 0, neuron_pop_size, delay_offset=0,
                                    weight=prompt_weight)  # connect all 2 prompt
    in_proj = pynn.Projection(input_pop,
                              neuron,
                              pynn.FromListConnector(from_list_in),
                              # pynn.AllToAllConnector(),
                              synapse_type=eprop_learning_neuron,
                              label='input_connections',
                              receptor_type='input_connections')

    eprop_learning_output = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=0.0))

    # from_list_out = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
    # from_list_out, max_syn_per_output = probability_connector(neuron_pop_size, 2, p_connect_out)
    # from_list_out = range_connector(0, neuron_pop_size/2, 1, 2, weight=out_weight)  # connect 1/2st 2 right output
    # from_list_out += range_connector(neuron_pop_size/2, neuron_pop_size, 0, 1, weight=out_weight)  # connect 2/2nd 2 left output
    # from_list_out += range_connector(0, neuron_pop_size/2, 0, 1, weight=-out_weight)  # connect 1/2st -2 left output
    # from_list_out += range_connector(neuron_pop_size/2, neuron_pop_size, 1, 2, weight=-out_weight)  # connect 2/2nd -2 right output
    from_list_out = range_connector(0, neuron_pop_size, 0, 2, weight=out_weight)  # connect all
    out_proj = pynn.Projection(neuron,
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')

    learning_proj = pynn.Projection(readout_pop,
                                    neuron,
                                    pynn.AllToAllConnector(),
                                    pynn.StaticSynapse(weight=0.5, delay=0),
                                    receptor_type='learning_signal')

    if recurrent_connections:
        eprop_learning_recurrent = pynn.STDPMechanism(
            timing_dependence=pynn.extra_models.TimingDependenceEprop(),
            weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
                w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

        # from_list_rec, max_syn_per_rec = probability_connector(neuron_pop_size, neuron_pop_size, p_connect_rec, offset=0)
        # from_list_rec = range_connector(0, neuron_pop_size/2, neuron_pop_size/2, neuron_pop_size, weight=rec_weight, delay_offset=100)  # inhibitory connections between 1/2s
        # from_list_rec += range_connector(neuron_pop_size/2, neuron_pop_size, 0, neuron_pop_size/2, weight=rec_weight, delay_offset=100)  # inhibitory connections between 1/2s
        from_list_rec = range_connector(0, neuron_pop_size, 0, neuron_pop_size, weight=rec_weight,
                                        delay_offset=100)  # recurrent connections
        recurrent_proj = pynn.Projection(neuron,
                                         neuron,
                                         pynn.FromListConnector(from_list_rec),
                                         synapse_type=eprop_learning_recurrent,
                                         label='recurrent_connections',
                                         receptor_type='recurrent_connections')
    else:
        from_list_rec = []
        recurrent_proj = None

    # input_pop.record('spikes')
    # neuron.record('spikes')
    # neuron.record(['gsyn_exc', 'v', 'gsyn_inh'],
    #               indexes=[i for i in range(int((neuron_pop_size / 2) - 5), int((neuron_pop_size / 2) + 5))])
    readout_pop.record('all')

    runtime = cycle_time * num_repeats

    experiment_label = "c-{} eta-{}_{} - size-{}_{} - weights-{} - p_conn-{}_{}_{} - rec-{} - cycle-{}_{}_{} b-{}".format(
        readout_neuron_params["number_of_cues"], readout_neuron_params["eta"], neuron_params["eta"], input_size, neuron_pop_size, weight_string, p_connect_in,
        p_connect_rec, p_connect_out, recurrent_connections, cycle_time, window_size, runtime, threshold_beta)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
           from_list_in, from_list_rec, from_list_out


def next_create_pops(from_list_in, from_list_rec, from_list_out):
    pynn.setup(timestep=1)
    input_pop = pynn.Population(input_size,
                                pynn.SpikeSourcePoisson(rate=rates),
                                # {'spike_times': build_input_spike_train(num_repeats, cycle_time, input_size)},
                                # {'spike_times': frozen_poisson_variable_hz(num_repeats, cycle_time, input_split, input_split, input_size)},
                                label='input_pop')

    neuron = pynn.Population(neuron_pop_size,
                             pynn.extra_models.EPropAdaptive(**neuron_params),
                             label='eprop_pop')

    # Output population
    readout_pop = pynn.Population(3,  # HARDCODED 1
                                  pynn.extra_models.LeftRightReadout(
                                      **readout_neuron_params
                                  ),
                                  label="readout_pop"
                                  )

    SpynnakerExternalDevicePluginManager.add_edge(readout_pop._get_vertex, input_pop._get_vertex, "CONTROL")

    eprop_learning_neuron = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

    from_list_in_list = []
    for conn in from_list_in:
        a = conn[0]
        b = conn[1]
        w = conn[2]
        d = conn[3]
        if a % input_size == 0:
            d = 0.0
        from_list_in_list.append([a, b, w, d])

    in_proj = pynn.Projection(input_pop,
                              neuron,
                              pynn.FromListConnector(from_list_in_list),
                              # pynn.AllToAllConnector(),
                              synapse_type=eprop_learning_neuron,
                              label='input_connections',
                              receptor_type='input_connections')

    eprop_learning_output = pynn.STDPMechanism(
        timing_dependence=pynn.extra_models.TimingDependenceEprop(),
        weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
            w_min=-max_weight, w_max=max_weight, reg_rate=0.0))

    from_list_out_list = []
    for conn in from_list_out:
        a = conn[0]
        b = conn[1]
        w = conn[2]
        d = conn[3]
        if a % neuron_pop_size == 0:
            d = 0.0
        from_list_out_list.append([a, b, w, d])

    out_proj = pynn.Projection(neuron,
                               readout_pop,
                               # pynn.OneToOneConnector(),
                               pynn.FromListConnector(from_list_out_list),
                               synapse_type=eprop_learning_output,
                               label='input_connections',
                               receptor_type='input_connections')

    learning_proj = pynn.Projection(readout_pop,
                                    neuron,
                                    pynn.AllToAllConnector(),
                                    pynn.StaticSynapse(weight=0.5, delay=0),
                                    receptor_type='learning_signal')
    recurrent_proj = None
    if recurrent_connections:
        eprop_learning_recurrent = pynn.STDPMechanism(
            timing_dependence=pynn.extra_models.TimingDependenceEprop(),
            weight_dependence=pynn.extra_models.WeightDependenceEpropReg(
                w_min=-max_weight, w_max=max_weight, reg_rate=reg_rate))

        from_list_rec_list = []
        for conn in from_list_rec:
            a = conn[0]
            b = conn[1]
            w = conn[2]
            d = conn[3]
            # might need some check for delay
            from_list_rec_list.append([a, b, w, d])

        recurrent_proj = pynn.Projection(neuron,
                                         neuron,
                                         pynn.FromListConnector(from_list_rec_list),
                                         synapse_type=eprop_learning_recurrent,
                                         label='recurrent_connections',
                                         receptor_type='recurrent_connections')

    # input_pop.record('spikes')
    # neuron.record('spikes')
    # neuron.record(['gsyn_exc', 'v', 'gsyn_inh'],
    #               indexes=[i for i in range(int((neuron_pop_size / 2) - 5), int((neuron_pop_size / 2) + 5))])
    readout_pop.record('all')

    window_size = neuron_params["window_size"]
    cycle_time = int(window_size / window_cycles)
    runtime = cycle_time * num_repeats

    experiment_label = "c-{} eta-{}_{} - size-{}_{} - weights-{} - p_conn-{}_{}_{} - rec-{} - cycle-{}_{}_{} b-{}".format(
        readout_neuron_params["number_of_cues"], readout_neuron_params["eta"], neuron_params["eta"], input_size, neuron_pop_size, weight_string, p_connect_in,
        p_connect_rec, p_connect_out, recurrent_connections, cycle_time, window_size, runtime, threshold_beta)
    print("\n", experiment_label, "\n")

    return experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
           from_list_in, from_list_rec, from_list_out

def run_until(experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop,
              from_list_in, from_list_rec, from_list_out, threshold=0.95):
    good_performance = False
    current_window = 0
    current_iter = current_window * window_cycles
    window_size = neuron_params["window_size"]
    cycle_time = int(window_size / window_cycles)
    runtime = cycle_time * num_repeats
    cycle_error = [0.0 for i in range(int(runtime/window_size))]
    correct_or_not = [0 for i in range(int(runtime/window_size))]

    while current_window * window_size < runtime:
        pynn.run(window_size)
        # in_spikes = input_pop.get_data('spikes', clear=True)
        # neuron_res = neuron.get_data('all', clear=True)
        readout_res = readout_pop.get_data('all', clear=True)
        # plot_start = (window_size * current_window)
        # plot_end = (window_size * current_window)

        total_error = 0.0
        for cycle in range(window_cycles):
            for time_index in range(cycle_time):
                instantaneous_error = np.abs(float(
                    readout_res.segments[0].filter(name='gsyn_inh')[0][time_index + ((cycle+current_iter) * cycle_time)][0]))
                cycle_error[(current_iter)+cycle] += instantaneous_error
                total_error += instantaneous_error
            if cycle_error[(current_iter)+cycle] < 75:
                correct_or_not[(current_iter)+cycle] = 1

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

        current_window += 1
        current_iter = current_window * window_cycles

        print(cycle_error[:current_iter])
        for i in range(current_window):
            print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
                  np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]))

        graph_directory = './graphs/'
        draw_graph_from_list(new_connections_in, new_connections_rec, new_connections_out, graph_directory,
                             experiment_label + ' {}'.format(current_window), rec_flag=recurrent_connections,
                             save_flag=True)
        plot_learning_curve(correct_or_not[:current_iter], cycle_error[:current_iter], graph_directory,
                            experiment_label + ' {}'.format(current_window), save_flag=True)

        if current_iter > 64 and \
                np.average(correct_or_not[(current_iter)-64: current_iter]) > threshold:
            pynn.end()
            print(cycle_error[:current_iter])
            for i in range(current_window):
                print(correct_or_not[i * window_cycles:(i + 1) * window_cycles],
                      np.average(correct_or_not[i * window_cycles:(i + 1) * window_cycles]))
            print("Simulation has achieved threshold performance at window:", current_window)
            print("this corresponds to itertation:", current_iter)
            good_performance = current_iter
            return new_connections_in, new_connections_rec, new_connections_out, \
                   correct_or_not[:current_iter], cycle_error[:current_iter], good_performance
    pynn.end()
    print("Learning has failed to achieve threshold performance in runtime")
    return new_connections_in, new_connections_rec, new_connections_out, \
           correct_or_not[:current_iter], cycle_error[:current_iter], good_performance

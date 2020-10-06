from PyNN8Examples.eprop_testing.plot_graph import draw_graph_from_list, plot_learning_curve
from PyNN8Examples.eprop_testing.create_pops_for_incremental_learning import first_create_pops, next_create_pops, run_until
from PyNN8Examples.eprop_testing.incremental_config import *

np.random.seed(272727)

# create populations for the first time
experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
from_list_in, from_list_rec, from_list_out = first_create_pops()

correct_or_not_full_list = []
cycle_error_full_list = []
transiterations = []

for interative_cue in range(1, 9, 2):
    # run until learning threshold criteria met
    new_connections_in, new_connections_rec, new_connections_out, \
    correct_or_not_full_list, cycle_error_full_list, final_iteration = \
        run_until(experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj,
                  input_pop, neuron, readout_pop,
                  from_list_in, from_list_rec, from_list_out,
                  correct_or_not_full_list, cycle_error_full_list,
                  threshold=learning_threshold,
                  cue_break=transiterations)

    if not final_iteration:
        print("Ending simulation")
        break
    transiterations.append(final_iteration)

    # update cues in params
    cycle_time = ((interative_cue+2) * 150) + 1000 + 150
    window_size = cycle_time * window_cycles
    readout_neuron_params["number_of_cues"] += 2
    neuron_params["number_of_cues"] += 2
    neuron_params["window_size"] = window_size

    # create pops for next run
    experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
    from_list_in, from_list_rec, from_list_out = \
        next_create_pops(new_connections_in, new_connections_rec, new_connections_out)

plot_learning_curve(correct_or_not_full_list, cycle_error_full_list, './big_with_labels/',
                    'full', save_flag=True)

print("job done")
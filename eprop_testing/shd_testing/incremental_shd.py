import numpy as np
from eprop_testing.shd_testing.plot_shd_graph import plot_learning_curve
from eprop_testing.shd_testing.create_pops_shd_incremental import \
    first_create_pops, next_create_pops, run_until
from eprop_testing.shd_testing.incremental_shd_config import *


replay_counter = 0

# create populations for the first time
experiment_label, runtime, pynn, in_proj, recurrent_proj, layer_proj, out_proj, input_pop, neuron, readout_pop, \
from_list_in, from_list_rec, from_list_out, new_labels = first_create_pops()

correct_or_not_full_list = []
cycle_error_full_list = []
transiterations = []
confusion_matrix = [[0. for i in range(output_size)] for i in range(output_size)]

number_of_classes = no_class_start
while number_of_classes <= output_size:#in range(no_class_start, output_size+class_progress, class_progress):
    # run until learning threshold criteria met
    new_connections_in, new_connections_rec, new_connections_layer, new_connections_out, \
    correct_or_not_full_list, cycle_error_full_list, final_iteration = \
        run_until(experiment_label, runtime, pynn,
                  in_proj, recurrent_proj, layer_proj, out_proj,
                  input_pop, neuron, readout_pop,
                  from_list_in, from_list_rec, from_list_out,
                  correct_or_not_full_list, cycle_error_full_list, confusion_matrix, new_labels,
                  threshold=learning_threshold,
                  cue_break=transiterations,
                  final_class=(number_of_classes == 10),  # increase if increase classes
                  repeating=replay_counter)

    if not final_iteration and not repeat_on_fail:
        print("Ending simulation")
        break
    if final_iteration:
        if number_of_classes >= 10:
            print("Termination: Training has finished")
            break
        else:
            if number_of_classes == 3:
                print("Did 3 classes")
            if number_of_classes == 4:
                print("Did 4 classes")
            if number_of_classes == 5:
                print("Did 5 classes")
            if number_of_classes == 6:
                print("Did 6 classes")
            if number_of_classes == 7:
                print("Did 7 classes")
            if number_of_classes == 8:
                print("Did 8 classes")
            number_of_classes += class_progress
            number_of_classes = min(number_of_classes, 10)
            replay_counter = 0
        transiterations.append(final_iteration)
    else:
        replay_counter += 1
        print("Simulation is repeating previous number of classes: repeat #", replay_counter)

    # create pops for next run
    experiment_label, runtime, pynn, \
    in_proj, recurrent_proj, layer_proj, out_proj, \
    input_pop, neuron, readout_pop, \
    from_list_in, from_list_rec, from_list_out, new_labels = \
        next_create_pops(new_connections_in, new_connections_rec,
                         new_connections_layer, new_connections_out,
                         number_of_classes)  # increase if increase classes

print(experiment_label)
print("cycle_error =", cycle_error_full_list)
print(experiment_label)
# print("total error =", total_error)
# print("classification = ", test_classification)
print("correct or not = ", correct_or_not_full_list)
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
print("")
print("average classification = ", np.average(correct_or_not_full_list))
print("weighted average classification = ", np.average(correct_or_not_full_list,
                                                       weights=[i for i in range(len(correct_or_not_full_list))]))
print(experiment_label)
print("average error = ", np.average(cycle_error_full_list))
print("weighted average", np.average(cycle_error_full_list, weights=[i for i in range(len(cycle_error_full_list))]))
print("minimum error = ", np.min(cycle_error_full_list))
print("minimum iteration = ", cycle_error_full_list.index(np.min(cycle_error_full_list)),
      "- with time stamp =", cycle_error_full_list.index(np.min(cycle_error_full_list)) * 1024)

plot_learning_curve(correct_or_not_full_list, cycle_error_full_list,
                    confusion_matrix, confusion_matrix, './../shd_graphs',
                    'full', save_flag=True, plot_flag=True, cue_break=transiterations,
                    no_classes=number_of_classes)

print("job done")
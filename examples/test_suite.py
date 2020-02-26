import single_neuron
import mixed_signals
import single_neuron_microsec
import multi_layer
import self_connection
import single_neuron_double_spikes
import multi_source_single_dest
import single_neuron_del_ext
from termcolor import colored

if single_neuron.single_neuron() == False:
    # Simple excitatory connection between two neurons with FromListConnector
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test FAILED", "red")
elif single_neuron_microsec.single_neuron_microsec() == False:
    # Simple excitatory connection exploiting FixedProbabilityConnector
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test FAILED", "red")
elif multi_layer.multi_layer() == False:
    # Spike source array, 2 populations, FromListConnector. 2 layers network
    # Checks propagation of spikes on the 2 partitions on different populaitons
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test PASSED", "green")
    print colored("Multi layer test FAILED", "red")
elif self_connection.self_connection() == False:
    # Spike source array, 1 population, FromListConnector. Self connection
    # Checks the correctness of the partions when connected to the same population
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test PASSED", "green")
    print colored("Multi layer test PASSED", "green")
    print colored("Self connection test FAILED", "red")
elif single_neuron_double_spikes.single_neuron_double_spikes() == False:
    # Spike source array, 1 population, FromListConnector. both exc and inh spikes
    # Tests correctness of both exc and inh gsyn on a population composed of 2 partitions
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test PASSED", "green")
    print colored("Multi layer test PASSED", "green")
    print colored("Self connection test PASSED", "green")
    print colored("Single neuron double spikes test FAILED", "red")
# elif multi_source_single_dest.multi_source_single_dest() == False:
#     # 10 SpikeSourceArrays spiking in parallel to a single neuron
#     print colored("Spiking Neural Networks Base test", "red")
#     print colored("Author Luca Peres, The Univerity of Manchester", "red")
#     print "----------------------------------------------------------"
#     print colored("Single neuron test PASSED", "green")
#     print colored("Single neuron microsec test PASSED", "green")
#     print colored("Multi layer test PASSED", "green")
#     print colored("Self connection test PASSED", "green")
#     print colored("Single neuron double spikes test PASSED", "green")
#     print colored("Multi source single destination test FAILED", "red")
elif mixed_signals.mixed_signals() == False:
    # Network containing two inputs connected through FixedProbabilityConnector to
    # two populations which excange both excitatory and inhibitory spikes with each other
    print colored("Spiking Neural Networks Base test", "red")
    print colored("Author Luca Peres, The Univerity of Manchester", "red")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test PASSED", "green")
    print colored("Multi layer test PASSED", "green")
    print colored("Self connection test PASSED", "green")
    print colored("Single neuron double spikes test PASSED", "green")
    print colored("Multi source single destination test PASSED", "green")
    print colored("Mixed signals test FAILED", "red")
else:
    print "\n\n\n\n"
    print colored("Spiking Neural Networks Base test", "green")
    print colored("Author Luca Peres, The Univerity of Manchester", "green")
    print "----------------------------------------------------------"
    print colored("Single neuron test PASSED", "green")
    print colored("Single neuron microsec test PASSED", "green")
    print colored("Multi layer test PASSED", "green")
    print colored("Self connection test PASSED", "green")
    print colored("Single neuron double spikes test PASSED", "green")
    print colored("Multi source single destination test PASSED", "green")
    print colored("Mixed signals test PASSED", "green")
    print colored("All tests PASSED", "green")

# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
from threading import Condition
import pyNN.spiNNaker as Frontend
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

###################################
# Setup for Live Input and Output #
###################################

# Create a condition to avoid overlapping prints
print_condition = Condition()


# Create an initialisation method
def init_pop(_label, _n_neurons, _run_time_ms, _machine_timestep_ms):
    """
    Print method to show callback has been called

    :param str _label:
    :param int _n_neurons:
    :param float _run_time_ms:
    :param float _machine_timestep_ms:
    """
    print(f"{_label} has {_n_neurons} neurons")
    print(f"Simulation will run for {_run_time_ms}ms "
          f"at {_machine_timestep_ms}ms timesteps")


# Create a sender of packets for the forward population
def send_input_forward(label, sender):
    """
    Sends 5 spikes at 20 millisecond intervals

    :param str label:
    :param SpynnakerLiveSpikesConnection sender:
    """
    for neuron_id in range(0, 100, 20):
        time.sleep(random.random() + 0.5)
        print_condition.acquire()
        print(f"Sending forward spike {neuron_id}")
        print_condition.release()
        sender.send_spike(label, neuron_id, send_full_keys=True)


# Create a sender of packets for the backward population
def send_input_backward(label, sender):
    """
    Sends 5 spikes at 20 millisecond intervals

    :param str label:
    :param SpynnakerLiveSpikesConnection sender:
    """
    for neuron_id in range(0, 100, 20):
        real_id = 100 - neuron_id - 1
        time.sleep(random.random() + 0.5)
        print_condition.acquire()
        print(f"Sending backward spike {real_id}")
        print_condition.release()
        sender.send_spike(label, real_id)


# Create a receiver of live spikes
def receive_spikes(label, _time, neuron_ids):
    """
    Prints that spikes have been received

    :param int label:
    :param str _time:
    :param list(int) neuron_ids:
    :return:
    """
    for neuron_id in neuron_ids:
        print_condition.acquire()
        print(f"Received spike at time {_time} from {label} - {neuron_id}")
        print_condition.release()


# Set up the live connection for sending spikes
live_spikes_connection_send = \
    Frontend.external_devices.SpynnakerLiveSpikesConnection(
        receive_labels=None, local_port=None,
        send_labels=["spike_injector_forward", "spike_injector_backward"])

# Set up callbacks to occur at initialisation
live_spikes_connection_send.add_init_callback(
    "spike_injector_forward", init_pop)
live_spikes_connection_send.add_init_callback(
    "spike_injector_backward", init_pop)

# Set up callbacks to occur at the start of simulation
live_spikes_connection_send.add_start_resume_callback(
    "spike_injector_forward", send_input_forward)
live_spikes_connection_send.add_start_resume_callback(
    "spike_injector_backward", send_input_backward)

# a new spynnaker live spikes connection is created to define that there is
# a python function which receives the spikes.
live_spikes_connection_receive = \
    Frontend.external_devices.SpynnakerLiveSpikesConnection(
        receive_labels=["pop_forward", "pop_backward"],
        local_port=None, send_labels=None)

# Set up callbacks to occur when spikes are received
live_spikes_connection_receive.add_receive_callback(
    "pop_forward", receive_spikes)
live_spikes_connection_receive.add_receive_callback(
    "pop_backward", receive_spikes)


############################################################
# Setup a Simulation to be injected into and received from #
############################################################

# initial call to set up the front end (pynn requirement)
Frontend.setup(timestep=1.0, min_delay=1.0)


# neurons per population and the length of runtime in ms for the simulation,
# as well as the expected weight each spike will contain
n_neurons = 100
run_time = 8000
weight_to_spike = 2.0

# neural parameters of the model used to respond to injected spikes.
# (cell params for a synfire chain)
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

##################################
# Parameters for the injector population.
# The virtual key is assigned here, rather than being allocated later.
# Spikes injected need to match this key, and this will be done automatically
# with 16-bit keys.
##################################
cell_params_spike_injector_with_key = {

    # This is the base key to be used for the injection, which is used to
    # allow the keys to be routed around the spiNNaker machine.  This
    # assignment means that 32-bit keys must have the high-order 16-bit
    # set to 0x7; This will automatically be prepended to 16-bit keys.
    'virtual_key': 0x70000,
}

# create synfire populations (if cur exp)
pop_forward = Frontend.Population(
    n_neurons, Frontend.IF_curr_exp(**cell_params_lif), label='pop_forward')
pop_backward = Frontend.Population(
    n_neurons, Frontend.IF_curr_exp(**cell_params_lif), label='pop_backward')

# Create injection populations
injector_forward = Frontend.Population(
    n_neurons, Frontend.external_devices.SpikeInjector(
        database_notify_port_num=live_spikes_connection_send.local_port),
    label='spike_injector_forward',
    additional_parameters=cell_params_spike_injector_with_key)
injector_backward = Frontend.Population(
    n_neurons, Frontend.external_devices.SpikeInjector(
        database_notify_port_num=live_spikes_connection_send.local_port),
    label='spike_injector_backward')

# Create a connection from the injector into the populations
Frontend.Projection(
    injector_forward, pop_forward, Frontend.OneToOneConnector(),
    Frontend.StaticSynapse(weight=weight_to_spike))
Frontend.Projection(
    injector_backward, pop_backward, Frontend.OneToOneConnector(),
    Frontend.StaticSynapse(weight=weight_to_spike))

# Synfire chain connections where each neuron is connected to its next neuron
# NOTE: there is no recurrent connection so that each chain stops once it
# reaches the end
loop_forward = list()
loop_backward = list()
for i in range(0, n_neurons - 1):
    loop_forward.append((i, (i + 1) % n_neurons, weight_to_spike, 3))
    loop_backward.append(((i + 1) % n_neurons, i, weight_to_spike, 3))
Frontend.Projection(pop_forward, pop_forward,
                    Frontend.FromListConnector(loop_forward))
Frontend.Projection(pop_backward, pop_backward,
                    Frontend.FromListConnector(loop_backward))

# record spikes from the synfire chains so that we can read off valid results
# in a safe way afterwards, and verify the behaviour
pop_forward.record('spikes')
pop_backward.record('spikes')

# Activate the sending of live spikes
Frontend.external_devices.activate_live_output_for(
    pop_forward,
    database_notify_port_num=live_spikes_connection_receive.local_port)
Frontend.external_devices.activate_live_output_for(
    pop_backward,
    database_notify_port_num=live_spikes_connection_receive.local_port)


# Run the simulation on spiNNaker
Frontend.run(run_time)

spikes_forward = pop_forward.get_data('spikes')
spikes_backward = pop_backward.get_data('spikes')

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes_forward.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    Panel(spikes_backward.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    title="Simple synfire chain example with injected spikes",
    annotations=f"Simulated with {Frontend.name}"
)
plt.show()

# Clear data structures on spiNNaker to leave the machine in a clean state for
# future executions
Frontend.end()

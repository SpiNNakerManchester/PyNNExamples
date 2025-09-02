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

import multiprocessing
#import tkinter as tk
import pyNN.spiNNaker as Frontend
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


class PyNNScript(object):
    """
    the class which contains the pynn script
    """

    def __init__(self):

        # initial call to set up the front end (pynn requirement)
        Frontend.setup(timestep=1.0, min_delay=1.0)

        use_c_visualiser = False
        visualiser_port = 19996
        use_spike_injector = True

        # neurons per population and the length of runtime in ms for the
        # simulation, as well as the expected weight each spike will contain
        self.n_neurons = 100

        # set up GUI
        p = None
        sender_port = None
        if use_spike_injector:
            port = multiprocessing.Value('i', 0)
            event = multiprocessing.Event()
            p = multiprocessing.Process(
                target=GUI, args=[self.n_neurons, event, port])
            p.start()
            event.wait()
            sender_port = port.value

        if not use_c_visualiser:
            # if not using the c visualiser, then a new spynnaker live spikes
            # connection is created to define that there are python code which
            # receives the outputted spikes.
            live_spikes_connection_receive = \
                Frontend.external_devices.SpynnakerLiveSpikesConnection(
                    receive_labels=["pop_forward", "pop_backward"],
                    local_port=None, send_labels=None)
            visualiser_port = live_spikes_connection_receive.local_port

            # Set up callbacks to occur when spikes are received
            live_spikes_connection_receive.add_receive_callback(
                "pop_forward", receive_spikes)
            live_spikes_connection_receive.add_receive_callback(
                "pop_backward", receive_spikes)

        # different run times for demonstration purposes
        run_time = None
        if not use_c_visualiser and not use_spike_injector:
            run_time = 1000
        elif use_c_visualiser and not use_spike_injector:
            run_time = 10000
        elif use_c_visualiser and use_spike_injector:
            run_time = 100000
        elif not use_c_visualiser and use_spike_injector:
            run_time = 10000

        weight_to_spike = 2.0

        # neural parameters of the IF_curr model used to respond to injected
        # spikes.
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
        # Parameters for the injector population.  The virtual key is assigned
        # here, rather than being allocated later.  Spikes injected need to
        # match this key, and this will be done automatically with 16-bit keys.
        ##################################
        cell_params_spike_injector_with_key = {

            # This is the base key to be used for the injection, which is used
            # to allow the keys to be routed around the spiNNaker machine.
            # This assignment means that 32-bit keys must have the high-order
            # 16-bit set to 0x7; This will automatically be prepended to
            # 16-bit keys.
            'virtual_key': 0x70000
        }

        # create synfire populations (if cur exp)
        pop_forward = Frontend.Population(
            self.n_neurons, Frontend.IF_curr_exp(**cell_params_lif),
            label='pop_forward')
        pop_backward = Frontend.Population(
            self.n_neurons, Frontend.IF_curr_exp(**cell_params_lif),
            label='pop_backward')

        # Create injection populations
        injector_forward = None
        injector_backward = None
        if use_spike_injector:
            injector_forward = Frontend.Population(
                self.n_neurons,
                Frontend.external_devices.SpikeInjector(
                    database_notify_port_num=sender_port),
                label='spike_injector_forward',
                additional_parameters=cell_params_spike_injector_with_key)
            injector_backward = Frontend.Population(
                self.n_neurons,
                Frontend.external_devices.SpikeInjector(
                    database_notify_port_num=sender_port),
                label='spike_injector_backward')
        else:
            spike_times = []
            for _ in range(0, self.n_neurons):
                spike_times.append([])
            spike_times[0] = [0]
            spike_times[20] = [(run_time / 100) * 20]
            spike_times[40] = [(run_time / 100) * 40]
            spike_times[60] = [(run_time / 100) * 60]
            spike_times[80] = [(run_time / 100) * 80]
            cell_params_forward = {'spike_times': spike_times}
            spike_times_backwards = []
            for _ in range(0, self.n_neurons):
                spike_times_backwards.append([])
            spike_times_backwards[0] = [(run_time / 100) * 80]
            spike_times_backwards[20] = [(run_time / 100) * 60]
            spike_times_backwards[40] = [(run_time / 100) * 40]
            spike_times_backwards[60] = [(run_time / 100) * 20]
            spike_times_backwards[80] = [0]
            cell_params_backward = {'spike_times': spike_times_backwards}
            injector_forward = Frontend.Population(
                self.n_neurons, Frontend.SpikeSourceArray(
                    **cell_params_forward),
                label='spike_injector_forward')
            injector_backward = Frontend.Population(
                self.n_neurons, Frontend.SpikeSourceArray(
                    **cell_params_backward),
                label='spike_injector_backward')

        # Create a connection from the injector into the populations
        Frontend.Projection(
            injector_forward, pop_forward,
            Frontend.OneToOneConnector(),
            Frontend.StaticSynapse(weight=weight_to_spike))
        Frontend.Projection(
            injector_backward, pop_backward,
            Frontend.OneToOneConnector(),
            Frontend.StaticSynapse(weight=weight_to_spike))

        # Synfire chain connections where each neuron is connected to its next
        # neuron
        # NOTE: there is no recurrent connection so that each chain stops once
        # it reaches the end
        loop_forward = list()
        loop_backward = list()
        for i in range(0, self.n_neurons - 1):
            loop_forward.append((i, (i + 1) %
                                 self.n_neurons, weight_to_spike, 3))
            loop_backward.append(((i + 1) %
                                  self.n_neurons, i, weight_to_spike, 3))
        Frontend.Projection(pop_forward, pop_forward,
                            Frontend.FromListConnector(loop_forward))
        Frontend.Projection(pop_backward, pop_backward,
                            Frontend.FromListConnector(loop_backward))

        # record spikes from the synfire chains so that we can read off valid
        # results in a safe way afterwards, and verify the behaviour
        pop_forward.record('spikes')
        pop_backward.record('spikes')

        # Activate the sending of live spikes
        Frontend.external_devices.activate_live_output_for(
            pop_forward, database_notify_host="localhost",
            database_notify_port_num=visualiser_port)
        Frontend.external_devices.activate_live_output_for(
            pop_backward, database_notify_host="localhost",
            database_notify_port_num=visualiser_port)

        # Run the simulation on spiNNaker
        Frontend.run(run_time)

        # Retrieve spikes from the synfire chain population
        spikes_forward = pop_forward.get_data('spikes')
        spikes_backward = pop_backward.get_data('spikes')

        # Clear data structures on spiNNaker to leave the machine in a clean
        # state for future executions
        Frontend.end()

        if use_spike_injector:
            p.join()

        Figure(
            # raster plot of the presynaptic neuron spike times
            Panel(spikes_forward.segments[0].spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, run_time)),
            Panel(spikes_backward.segments[0].spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, run_time)),
            title="Simple synfire chain example with injected spikes",
            annotations=f"Simulated with {Frontend.name()}"
        )
        plt.show()


# Create a receiver of live spikes
def receive_spikes(label, time, neuron_ids):
    """
    Print that spikes have been received

    :param str label:
    :param int time:
    :param list(int) neuron_ids:
    :return:
    """
    for neuron_id in neuron_ids:
        print(f"Received spike at time {time} from {label} - {neuron_id}")


class GUI(object):
    """ Simple GUI to demonstrate live injection of the spike io script.
    """

    def __init__(self, n_neurons, ready, port):
        """
        :param n_neurons: Number of neurons to show
        :param ready: multiprocessing.Event
        :param port: multiprocessing.Value
        """
        self._n_neurons = n_neurons

        # Set up the live connection for sending and receiving spikes
        self._live_spikes_connection = \
            Frontend.external_devices.SpynnakerLiveSpikesConnection(
                receive_labels=None, local_port=None,
                send_labels=["spike_injector_forward",
                             "spike_injector_backward"])
        port.value = self._live_spikes_connection.local_port

        # Set up callbacks to occur at the start of simulation
        self._live_spikes_connection.add_start_resume_callback(
            "spike_injector_forward", self.start)

        self._root = tk.Tk()
        self._root.title("Injecting Spikes GUI")
        tk.Label(self._root, fg="dark green").pack()

        self._neuron_id = tk.IntVar()
        tk.Spinbox(
            self._root, from_=0, to=self._n_neurons - 1,
            textvariable=self._neuron_id).pack()

        self._pop_label = tk.StringVar()
        tk.Spinbox(
            self._root, textvariable=self._pop_label,
            values=("spike_injector_forward", "spike_injector_backward")
            ).pack()

        self._button = tk.Button(
            self._root, text='Inject', width=25, command=self.inject_spike,
            state="disabled")
        self._button.pack()

        ready.set()

        self._root.mainloop()

    def start(self, pop_label, connection):
        """
        Set the start button to state to normal

        :param pop_label: IGNORED
        :param connection:  IGNORED
        """
        # pylint: disable=unused-argument
        self._button["state"] = "normal"

    def inject_spike(self):
        """
        Inject a spike into system
        """
        neuron_id = self._neuron_id.get()
        label = self._pop_label.get()
        print(f"injecting with neuron_id {neuron_id} to pop {label}")
        self._live_spikes_connection.send_spike(label, neuron_id)


# set up the initial script
if __name__ == '__main__':
    multiprocessing.freeze_support()
    script = PyNNScript()

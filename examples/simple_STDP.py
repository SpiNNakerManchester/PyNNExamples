# Copyright (c) 2020 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
A simple example of using STDP.

A single post-synaptic neuron fires at a constant rate. We connect several
pre-synaptic neurons to it, each of which fires spikes with a fixed time
lag or time advance with respect to the post-synaptic neuron.
The weights of these connections are small, so they will not
significantly affect the firing times of the post-synaptic neuron.
We plot the amount of potentiation or depression of each synapse as a
function of the time difference.

Adapted from http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html
to run on SpiNNaker, but with alpha-synapses rather than conductance neurons;
the weights involved are too low to be resolved using fixed-point arithmetic,
so some alteration of parameters is necessary to get a similar effect.
"""

import numpy
import spynnaker8 as sim
# from quantities import ms
from pyNN.utility.plotting import Figure, Panel, DataTable
import matplotlib.pyplot as plt

# === Parameters ============================================================

firing_period = 100.0    # (ms) interval between spikes
cell_parameters = {
    "tau_m": 10.0,       # (ms)
    "v_thresh": -50.0,   # (mV)
    "v_reset": -60.0,    # (mV)
    "v_rest": -60.0,     # (mV)
    "cm": 1.0,           # (nF)
    "tau_refrac": firing_period / 2,  # (ms) long period to prevent bursting
}
n = 60                   # number of synapses / number of presynaptic neurons
delta_t = 1.0            # (ms) time between firing of neighbouring neurons
t_stop = 10 * firing_period + n * delta_t
delay = 3.0              # (ms) synaptic time delay

# === Set up the simulator ==================================================

sim.setup(timestep=0.1, min_delay=delay, max_delay=delay)

# === Build the network =====================================================


def build_spike_sequences(period, duration, n, delta_t):
    """
    Return a spike time generator for `n` neurons (spike sources), where
    all neurons fire with the same period, but neighbouring neurons have a
    relative firing time difference of `delta_t`.
    """
    def spike_time_gen(i):
        """Spike time generator. `i` should be an array of indices."""
        return [numpy.arange(
            period + j * delta_t, duration, period) for j in (i - n // 2)]
    return spike_time_gen


spike_sequence_generator = build_spike_sequences(
    firing_period, t_stop, n, delta_t)

spike_sequence = spike_sequence_generator(numpy.arange(n))

# presynaptic population
p1 = sim.Population(n, sim.SpikeSourceArray(spike_times=spike_sequence),
                    label="presynaptic")
# single postsynaptic neuron
p2 = sim.Population(1, sim.IF_curr_alpha(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]},
                    label="postsynaptic")
# drive to the postsynaptic neuron, ensuring it fires at exact multiples of
# the firing period
p3 = sim.Population(
    1, sim.SpikeSourceArray(spike_times=numpy.arange(
        firing_period - delay, t_stop, firing_period)),
    label="driver")

# we set the initial weights to be small, to avoid perturbing the firing
# times of the postsynaptic neurons
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=20.0, tau_minus=20.0, A_plus=0.05, A_minus=0.06),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=1.0),
    weight=0.5, delay=delay)
connections = sim.Projection(p1, p2, sim.AllToAllConnector(), stdp_model)

# the connection weight from the driver neuron is very strong, to ensure the
# postsynaptic neuron fires at the correct times
driver_connection = sim.Projection(p3, p2, sim.OneToOneConnector(),
                                   sim.StaticSynapse(weight=10.0, delay=delay))

# === Instrument the network =================================================

p1.record('spikes')
p2.record(['spikes', 'v'])

# === Run the simulation =====================================================

sim.run(t_stop)

# === Save the results, optionally plot a figure =============================

presynaptic_spikes = p1.get_data('spikes').segments[0]
postsynaptic_spikes = p2.get_data('spikes').segments[0]
postsynaptic_v = p2.get_data('v').segments[0]
print("Post-synaptic spike times: %s" % postsynaptic_spikes.spiketrains[0])

weights = connections.get(["weight"], "list")
final_weights = numpy.array([w[-1] for w in weights])
deltas = delta_t * numpy.arange(n // 2, -n // 2, -1)
print("Final weights: %s" % final_weights)
plasticity_data = DataTable(deltas, final_weights)

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(presynaptic_spikes.spiketrains,
          yticks=True, markersize=0.2, xlim=(0, t_stop)),
    # membrane potential of the postsynaptic neuron
    Panel(postsynaptic_v.filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[p2.label], xticks=True, yticks=True, xlim=(0, t_stop)),
    # evolution of the synaptic weights with time
    # Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)",
    #       legend=False, xlim=(0, t_stop)),
    # scatterplot of the final weight of each synapse against the relative
    # timing of pre- and postsynaptic spikes for that synapse
    Panel(plasticity_data,
          xticks=True, yticks=True, xlim=(-n / 2 * delta_t, n / 2 * delta_t),
          ylim=(0.9 * final_weights.min(), 1.1 * final_weights.max()),
          xlabel="t_post - t_pre (ms)", ylabel="Final weight (nA)"),
    title="Simple STDP example",
    annotations="Simulated with {}".format(sim.name())
)

# figure_filename = "simple_STDP.png"
# plt.savefig(figure_filename)
plt.show()

# === Clean up and quit =======================================================

sim.end()

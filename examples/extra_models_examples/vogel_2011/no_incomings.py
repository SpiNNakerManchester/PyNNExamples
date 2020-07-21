# Copyright (c) 2017-2019 The University of Manchester
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
from pyNN.random import NumpyRNG
import spynnaker8 as sim

# how much slowdown to put into the network to allow it to run without any
# runtime errors
SLOWDOWN = 200


class Vogels2011NoIncoming(object):
    """
    This example uses the sPyNNaker implementation of the inhibitory
    Plasticity rule developed by Vogels, Sprekeler, Zenke et al (2011)
    https://www.ncbi.nlm.nih.gov/pubmed/22075724
    http://www.opensourcebrain.org/projects/vogelsetal2011/wiki
    To reproduce the experiment from their paper
    """

    # Population parameters
    MODEL = sim.IF_curr_exp
    CELL_PARAMETERS = {
        'cm': 0.2,  # nF
        'i_offset': 0.2,
        'tau_m': 20.0,
        'tau_refrac': 5.0,
        'tau_syn_E': 5.0,
        'tau_syn_I': 10.0,
        'v_reset': -60.0,
        'v_rest': -60.0,
        'v_thresh': -50.0}

    # How large should the population of excitatory neurons be?
    # (Number of inhibitory neurons is proportional to this)
    NUM_EXCITATORY = 2000

    # How large should the population of inhibitory neurons be?
    # (Number of inhibitory neurons is proportional to excitatory)
    NUM_INHIBITORY = int(NUM_EXCITATORY / 4)

    # the number of neurons per core
    N_NEURONS_PER_CORE = 10

    # seed for random number generator
    RANDOM_NUMBER_GENERATOR_SEED = 0xDEADBEEF

    # first run runtime
    FIRST_RUN_RUNTIME = 1000

    def _build_network(self, slow_down):
        """ builds the network with either stdp or not, and with a given
        slowdown
        :param slow_down: the time scale factor to adjust the simulation by.
        :return: the excitatory population and the inhibitory->excitatory
        # projection
        """

        # SpiNNaker setup
        sim.setup(
            timestep=1.0, min_delay=1.0, max_delay=10.0,
            time_scale_factor=slow_down)

        # Reduce number of neurons to simulate on each core
        sim.set_number_of_neurons_per_core(
            self.MODEL, self.N_NEURONS_PER_CORE)

        # Create excitatory and inhibitory populations of neurons
        ex_pop = sim.Population(
            self.NUM_EXCITATORY, self.MODEL(**self.CELL_PARAMETERS))
        in_pop = sim.Population(
            self.NUM_INHIBITORY, self.MODEL(**self.CELL_PARAMETERS))
        ex_sink = sim.Population(1, sim.IF_cond_exp)
        in_sink = sim.Population(1, sim.IF_cond_exp)

        # Record excitatory spikes
        ex_pop.record(['spikes'])
        in_pop.record(['spikes'])

        # Make excitatory->inhibitory projections
        sim.Projection(
            ex_pop, ex_sink,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='excitatory',
            synapse_type=sim.StaticSynapse(weight=0.03))
        sim.Projection(
            ex_pop, ex_sink,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='excitatory',
            synapse_type=sim.StaticSynapse(weight=0.03))

        # Make inhibitory->inhibitory projections
        sim.Projection(
            in_pop, in_sink,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='inhibitory',
            synapse_type=sim.StaticSynapse(weight=-0.3))

    def run(self, slow_down):
        """ builds and runs a network

        :param slow_down: the slowdown for the network.
        :return: plastic weights, the static and plastic spikes.
        """
        # Build static network
        self._build_network(slow_down)

        sim.run(self.FIRST_RUN_RUNTIME)

        # end static simulation
        sim.end()


if __name__ == "__main__":
    x = Vogels2011NoIncoming()
    x.run(SLOWDOWN)

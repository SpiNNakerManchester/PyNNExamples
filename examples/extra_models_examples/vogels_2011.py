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
import numpy
import matplotlib.pyplot as pylab
from spynnaker8.utilities import neo_convertor

# how much slowdown to put into the network to allow it to run without any
# runtime errors
SLOWDOWN = 12

# bool hard code for extracting the weights or not
EXTRACT_WEIGHTS = False


class Vogals2011(object):
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

    # second run runtime
    SECOND_RUN_RUNTIME = 10000

    # bool for saving plastic spikes
    SAVE_PLASTIC_SPIKES = False

    def _build_network(self, uses_stdp, slow_down):
        """ builds the network with either stdp or not, and with a given
        slowdown

        :param uses_stdp: the bool for having plastic projections or not.
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
            sim.IF_curr_exp, self.N_NEURONS_PER_CORE)

        # Create excitatory and inhibitory populations of neurons
        ex_pop = sim.Population(
            self.NUM_EXCITATORY, self.MODEL(**self.CELL_PARAMETERS))
        in_pop = sim.Population(
            self.NUM_INHIBITORY, self.MODEL(**self.CELL_PARAMETERS))

        # Record excitatory spikes
        ex_pop.record(['spikes'])

        # create seeder
        rng_seeder = NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)

        # Make excitatory->inhibitory projections
        sim.Projection(ex_pop, in_pop,
                       sim.FixedProbabilityConnector(0.02, rng=rng_seeder),
                       receptor_type='excitatory',
                       synapse_type=sim.StaticSynapse(weight=0.03))
        sim.Projection(ex_pop, ex_pop,
                       sim.FixedProbabilityConnector(0.02, rng=rng_seeder),
                       receptor_type='excitatory',
                       synapse_type=sim.StaticSynapse(weight=0.03))

        # Make inhibitory->inhibitory projections
        sim.Projection(in_pop, in_pop,
                       sim.FixedProbabilityConnector(0.02, rng=rng_seeder),
                       receptor_type='inhibitory',
                       synapse_type=sim.StaticSynapse(weight=-0.3))

        # Build inhibitory plasticity model
        stdp_model = None
        if uses_stdp:
            stdp_model = sim.STDPMechanism(
                timing_dependence=sim.extra_models.Vogels2011Rule(
                    alpha=0.12, tau=20.0, A_plus=0.05),
                weight_dependence=sim.AdditiveWeightDependence(
                    w_min=0.0, w_max=1.0))

        # Make inhibitory->excitatory projection
        ie_projection = sim.Projection(
            in_pop, ex_pop,
            sim.FixedProbabilityConnector(0.02, rng=rng_seeder),
            receptor_type='inhibitory', synapse_type=stdp_model)

        # return the excitatory population and the inhibitory->excitatory
        # projection
        return ex_pop, ie_projection

    def run(self, slow_down, extract_weights):
        """ builds and runs a network

        :param slow_down: the slowdown for the network.
        :param extract_weights: bool for if we should run without weight
        extraction
        :return: plastic weights, the static and plastic spikes.
        """

        # Build static network
        static_ex_pop, _ = self._build_network(False, slow_down)

        # Run for 1s
        sim.run(self.FIRST_RUN_RUNTIME)

        # Get static spikes
        static_spikes = static_ex_pop.get_data('spikes')
        static_spikes_numpy = neo_convertor.convert_spikes(static_spikes)

        # Build plastic network
        sim.end()
        plastic_ex_pop, plastic_ie_projection = self._build_network(
            True, slow_down)

        # Run simulation
        sim.run(self.SECOND_RUN_RUNTIME)

        # Get plastic spikes and save to disk
        plastic_spikes = plastic_ex_pop.get_data('spikes')
        plastic_spikes_numpy = neo_convertor.convert_spikes(plastic_spikes)

        if self.SAVE_PLASTIC_SPIKES:
            numpy.save("plastic_spikes.npy", plastic_spikes_numpy)

        plastic_weights = None
        if extract_weights:
            plastic_weights = plastic_ie_projection.get('weight', 'list')

        # End simulation on SpiNNaker
        sim.end()

        # return things for plotting
        return plastic_weights, static_spikes_numpy, plastic_spikes_numpy

    def plot(
            self, plastic_weights, static_spikes_numpy, plastic_spikes_numpy):
        """ generates plots for a paper

        :param plastic_weights: the plastic weights.
        :param static_spikes_numpy: the static spikes.
        :param plastic_spikes_numpy: the plastic spikes.
        :rtype: None
        """

        # Weights(format="array")
        # print mean weight, if we bothered to extract them.
        if plastic_weights is not None:
            mean_weight = numpy.average(plastic_weights)
            print("Mean learnt ie weight:%f" % mean_weight)

        # Create plot
        fig, axes = pylab.subplots(3)

        # Plot last 200ms of static spikes (to match Brian script)
        axes[0].set_title("Excitatory raster without inhibitory plasticity")
        axes[0].scatter(static_spikes_numpy[:, 1],
                        static_spikes_numpy[:, 0], s=2, color="blue")
        axes[0].set_xlim(800, 1000)
        axes[0].set_ylim(0, self.NUM_EXCITATORY)

        # Plot last 200ms of plastic spikes (to match Brian script)
        axes[1].set_title("Excitatory raster with inhibitory plasticity")
        axes[1].scatter(plastic_spikes_numpy[:, 1],
                        plastic_spikes_numpy[:, 0], s=2, color="blue")
        axes[1].set_xlim(9800, 10000)
        axes[1].set_ylim(0, self.NUM_EXCITATORY)

        # Plot rates
        binsize = 10
        bins = numpy.arange(0, 10000 + 1, binsize)
        plastic_hist, _ = (
            numpy.histogram(plastic_spikes_numpy[:, 1], bins=bins))
        plastic_rate = (
            plastic_hist * (1000.0 / binsize) * (1.0 / self.NUM_EXCITATORY))
        axes[2].set_title("Excitatory rates with inhibitory plasticity")
        axes[2].plot(bins[0:-1], plastic_rate, color="red")
        axes[2].set_xlim(9800, 10000)
        axes[2].set_ylim(0, 20)

        # Show figures
        pylab.show()


if __name__ == "__main__":
    x = Vogals2011()
    result_weights, static, plastic = x.run(
        SLOWDOWN, EXTRACT_WEIGHTS)
    x.plot(result_weights, static, plastic)

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
import os
import matplotlib.pyplot as pylab
from spynnaker8.utilities import neo_convertor

# how much slowdown to put into the network to allow it to run without any
# runtime errors

# cheating with 6 boards and pair compressor
SLOWDOWN_STATIC = 4  # confirmed
# SLOWDOWN_STATIC = 2
SLOWDOWN_PLASTIC = 7

# cheating with 6 boards
# SLOWDOWN_STATIC = 4 # confirmed
# SLOWDOWN_PLASTIC = 136

# slow down bitfields and placer
# SLOWDOWN_STATIC = 6 # confirmed
# SLOWDOWN_PLASTIC = 136

# slowdown bitfields merged
# SLOWDOWN_STATIC = 6 # confirmed
# SLOWDOWN_PLASTIC = 136

# slowdown bitfields on core
# SLOWDOWN_STATIC = 10  # confirmed
# SLOWDOWN_PLASTIC = 136 # confirmed

# slowdown old master
# SLOWDOWN_STATIC = 13 # confirmed
# SLOWDOWN_PLASTIC = 158 #  confirmed

# bool hard code for extracting the weights or not
EXTRACT_WEIGHTS = False
GENERATE_PLOT = False

# how many boards to use for this test
N_BOARDS = 1


class Vogels2011(object):
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
    SECOND_RUN_RUNTIME = 1000

    # bool for saving spikes
    SAVE_SPIKES = True
    EXTRACT_SPIKES = True

    # bool saying to run static version or not
    RUN_STATIC_VERSION = True

    # bool saying to run the plastic version or not
    RUN_PLASTIC_VERSION = False

    # bool for reading and saving all connectivity
    SAVE_ALL_CONNECTIVITY_IF_INSANE = False

    # file name for ex static spikes
    STATIC_EX_SPIKES_FILE_NAME = "staticExcitSpikes"

    # file name for in static spikes
    STATIC_IN_SPIKES_FILE_NAME = "staticInhibSpikes"

    # file name for plastic ex spikes
    PLASTIC_EX_SPIKES_FILE_NAME = "plasticExcitSpikes"

    # file name for plastic in spikes
    PLASTIC_IN_SPIKES_FILE_NAME = "plasticInhibSpikes"

    def _build_network(self, uses_stdp, slow_down):
        """ builds the network with either stdp or not, and with a given
        slowdown

        :param uses_stdp: the bool for having plastic projections or not.
        :param slow_down: the time scale factor to adjust the simulation by.
        :return: the excitatory population and the inhibitory->excitatory
        """

        # SpiNNaker setup
        sim.setup(
            timestep=1.0, min_delay=1.0, max_delay=10.0,
            time_scale_factor=slow_down, n_boards_required=N_BOARDS)

        # Reduce number of neurons to simulate on each core
        sim.set_number_of_neurons_per_core(
            sim.IF_curr_exp, self.N_NEURONS_PER_CORE)

        # Create excitatory and inhibitory populations of neurons
        ex_pop = sim.Population(
            self.NUM_EXCITATORY, self.MODEL(**self.CELL_PARAMETERS),
            label="excit_pop")
        in_pop = sim.Population(
            self.NUM_INHIBITORY, self.MODEL(**self.CELL_PARAMETERS),
            label="inhib_pop")

        # Record excitatory spikes
        ex_pop.record(['spikes'])
        in_pop.record(['spikes'])

        # create seeder
        # rng_seeder = NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)

        # Make excitatory->inhibitory projections
        proj1 = sim.Projection(
            ex_pop, in_pop,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='excitatory',
            synapse_type=sim.StaticSynapse(weight=0.029))
        proj2 = sim.Projection(
            ex_pop, ex_pop,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='excitatory',
            synapse_type=sim.StaticSynapse(weight=0.029))

        # Make inhibitory->inhibitory projections
        proj3 = sim.Projection(
            in_pop, in_pop,
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='inhibitory',
            synapse_type=sim.StaticSynapse(weight=-0.29))

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
            sim.FixedProbabilityConnector(
                0.02, rng=NumpyRNG(seed=self.RANDOM_NUMBER_GENERATOR_SEED)),
            receptor_type='inhibitory', synapse_type=stdp_model)

        # return the excitatory population and the inhibitory->excitatory
        # projection
        return ex_pop, in_pop, ie_projection, proj1, proj2, proj3

    @staticmethod
    def save_name(spike_name):
        index = 0
        file_name = spike_name + "{}".format(index)
        while os.path.exists(file_name):
            index += 1
            file_name = spike_name + "{}".format(index)
        return file_name

    def run(self, slow_down_static, slow_down_plastic, extract_weights):
        """ builds and runs a network

        :param slow_down_static: the slowdown for the network during \
        static run.
        :param slow_down_plastic: the slowdown for the network during \
        plastic run
        :param extract_weights: bool for if we should run without weight
        extraction
        :return: plastic weights, the static and plastic spikes.
        """

        static_ex_spikes_numpy = None
        plastic_weights = None
        plastic_spikes_numpy = None

        if self.RUN_STATIC_VERSION:
            print("Generating Static network")
            # Build static network
            (static_ex_pop, static_in_pop, static_ie_projection, proj1, proj2,
             proj3) = self._build_network(False, slow_down_static)

            # Run for 1s
            sim.run(self.FIRST_RUN_RUNTIME)

            # get all connectivity
            projs = [proj2, proj3, static_ie_projection, proj1]
            index = 0
            if self.SAVE_ALL_CONNECTIVITY_IF_INSANE:
                for proj in projs:
                    proj.save("all", "projection{}_data".format(index))
                    index += 1

            # Get static spikes
            static_ex_spikes_numpy = None
            static_in_spikes_numpy = None
            if self.EXTRACT_SPIKES:
                static_ex_spikes = static_ex_pop.get_data('spikes')
                static_ex_spikes_numpy = neo_convertor.convert_spikes(
                    static_ex_spikes)
                static_in_spikes = static_in_pop.get_data('spikes')
                static_in_spikes_numpy = neo_convertor.convert_spikes(
                    static_in_spikes)

            if self.SAVE_SPIKES:
                ex_name = self.save_name(self.STATIC_EX_SPIKES_FILE_NAME)
                in_name = self.save_name(self.STATIC_IN_SPIKES_FILE_NAME)
                numpy.savetxt(ex_name, static_ex_spikes_numpy)
                numpy.savetxt(in_name, static_in_spikes_numpy)

            # end static simulation
            sim.end()

        if self.RUN_PLASTIC_VERSION:
            print("Generating plastic network")
            # Build plastic network
            (plastic_ex_pop, static_in_pop, plastic_ie_projection, proj1,
             proj2, proj3) = self._build_network(True, slow_down_plastic)

            index = 0
            if self.SAVE_ALL_CONNECTIVITY_IF_INSANE:
                plastic_ie_projection.save(
                    "all", "projection{}_before_data_plastic".format(index))

            # Run simulation
            sim.run(self.SECOND_RUN_RUNTIME)

            if self.SAVE_ALL_CONNECTIVITY_IF_INSANE:
                projs = [plastic_ie_projection]
                index = 0
                for proj in projs:
                    proj.save("all", "projection{}_data_plastic".format(index))
                    index += 1

            # Get plastic spikes and save to disk
            static_in_spikes_numpy = None
            plastic_spikes_numpy = None
            if self.EXTRACT_SPIKES:
                plastic_spikes = plastic_ex_pop.get_data('spikes')
                plastic_spikes_numpy = (
                    neo_convertor.convert_spikes(plastic_spikes))
                static_in_spikes = static_in_pop.get_data('spikes')
                static_in_spikes_numpy = neo_convertor.convert_spikes(
                    static_in_spikes)

            if self.SAVE_SPIKES:
                ex_name = self.save_name(self.PLASTIC_EX_SPIKES_FILE_NAME)
                in_name = self.save_name(self.PLASTIC_IN_SPIKES_FILE_NAME)
                numpy.savetxt(ex_name, plastic_spikes_numpy)
                numpy.savetxt(in_name, static_in_spikes_numpy)

            if extract_weights:
                plastic_weights = plastic_ie_projection.get('weight', 'list')

            # End simulation on SpiNNaker
            sim.end()

        # return things for plotting
        return plastic_weights, static_ex_spikes_numpy, plastic_spikes_numpy

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
        if static_spikes_numpy is not None:
            axes[0].set_title(
                "Excitatory raster without inhibitory plasticity")
            axes[0].scatter(static_spikes_numpy[:, 1],
                            static_spikes_numpy[:, 0], s=2, color="blue")
            axes[0].set_xlim(800, 1000)
            axes[0].set_ylim(0, self.NUM_EXCITATORY)

        # Plot last 200ms of plastic spikes (to match Brian script)
        if plastic_spikes_numpy is not None:
            axes[1].set_title("Excitatory raster with inhibitory plasticity")
            axes[1].scatter(plastic_spikes_numpy[:, 1],
                            plastic_spikes_numpy[:, 0], s=2, color="blue")

            # only plot last 50th
            axes[1].set_xlim(
                self.SECOND_RUN_RUNTIME - (self.SECOND_RUN_RUNTIME / 50),
                self.SECOND_RUN_RUNTIME)
            axes[1].set_ylim(0, self.NUM_EXCITATORY)

            # Plot rates
            binsize = 10
            bins = numpy.arange(0, self.SECOND_RUN_RUNTIME + 1, binsize)
            plastic_hist, _ = (
                numpy.histogram(plastic_spikes_numpy[:, 1], bins=bins))
            plastic_rate = (
                plastic_hist * (1000.0 / binsize) *
                (1.0 / self.NUM_EXCITATORY))
            axes[2].set_title("Excitatory rates with inhibitory plasticity")
            axes[2].plot(bins[0:-1], plastic_rate, color="red")
            # only plot last 50th
            axes[2].set_xlim(
                self.SECOND_RUN_RUNTIME - (self.SECOND_RUN_RUNTIME / 50),
                self.SECOND_RUN_RUNTIME)
            axes[2].set_ylim(0, 20)

        # Show figures
        pylab.show()


if __name__ == "__main__":
    x = Vogels2011()
    result_weights, static, plastic = x.run(
        SLOWDOWN_STATIC, SLOWDOWN_PLASTIC, EXTRACT_WEIGHTS)
    if GENERATE_PLOT:
        x.plot(result_weights, static, plastic)

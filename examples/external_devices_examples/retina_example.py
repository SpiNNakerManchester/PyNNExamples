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

"""
retina example that just feeds data from a retina to live output via an
intermediate population
"""
# try:
# import pyNN.spiNNaker as p
# except Exception:
import spynnaker as p

connected_chip_details = {
    "spinnaker_link_id": 0,
}


def get_updated_params(params):
    params.update(connected_chip_details)
    return params


# Setup
p.setup(timestep=1.0)

# FPGA Retina - Down Polarity
retina_pop = p.Population(
    None, p.external_devices.ExternalFPGARetinaDevice, get_updated_params({
        'retina_key': 0x5,
        'mode': p.external_devices.ExternalFPGARetinaDevice.MODE_128,
        'polarity': (
            p.external_devices.ExternalFPGARetinaDevice.DOWN_POLARITY)}),
    label='External retina')

population = p.Population(256, p.IF_curr_exp(), label='pop_1')
p.Projection(
    retina_pop, population, p.FixedProbabilityConnector(0.1),
    synapse_type=p.StaticSynapse(weight=0.1))

# q.activate_live_output_for(population)
p.run(1000)
p.end()

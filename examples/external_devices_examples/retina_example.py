# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
retina example that just feeds data from a retina to live output via an
intermediate population
"""
import pyNN.spiNNaker as p

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

# Copyright (c) 2017-2023 The University of Manchester
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
motor example that just feeds data to the motor pop which starts the motor
going forward
"""

import pyNN.spiNNaker as p

# set up the tools
p.setup(timestep=1.0, min_delay=1.0)

# set up the virtual chip coordinates for the motor
connected_chip_coords = {'x': 0, 'y': 0}
link = 4

populations = list()
projections = list()


input_population = p.Population(6, p.SpikeSourcePoisson(rate=10))
control_population = p.Population(6, p.IF_curr_exp())
motor_device = p.Population(
    6, p.external_devices.MunichMotorDevice(spinnaker_link_id=0))

p.Projection(
    input_population, control_population, p.OneToOneConnector(),
    synapse_type=p.StaticSynapse(weight=5.0))

p.external_devices.activate_live_output_to(control_population, motor_device)

p.run(1000)
p.end()

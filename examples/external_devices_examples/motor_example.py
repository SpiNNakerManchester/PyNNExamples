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
motor example that just feeds data to the motor pop which starts the motor
going forward
"""

try:
    import pyNN.spiNNaker as p
except ImportError:
    import spynnaker8 as p

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

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

###########################################
#  Import libraries
############################################
import numpy
import pyNN.spiNNaker as p


###########################################
#  Connection and simulation specifications
###########################################

# The UART port on IO Board the pushbot connected to
UART_ID = 0

# SpiNNaker link ID (IO Board connected to)
spinnaker_link = 0

# Retina resolution
retina_resolution = \
    p.external_devices.PushBotRetinaResolution.NATIVE_128_X_128

# Number of machine vertices.  We divide the retina into eight and then also
# into polarity.
n_machine_vertices = retina_resolution.value.pixels * 16

# Name to call the retina
retina_label = "Retina"

retina_viewer = p.external_devices.PushBotRetinaViewer(
    retina_resolution, retina_label)

# Simulate with 1 ms time step
p.setup(1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 1)


###########################################
#  Create external devices
###########################################
# Create the pushbot protocols to pass to the devices of Pushbot
# (i.e. motors, laser, etc)
pushbot_protocol = p.external_devices.MunichIoSpiNNakerLinkProtocol(
    mode=p.external_devices.protocols.MUNICH_MODES.PUSH_BOT,
    uart_id=UART_ID)

# Create motor devices on SpiNNaker
motor_0 = p.external_devices.PushBotSpiNNakerLinkMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_0_PERMANENT, pushbot_protocol,
    spinnaker_link)

motor_1 = p.external_devices.PushBotSpiNNakerLinkMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_1_PERMANENT, pushbot_protocol,
    spinnaker_link)

# Create retina device on SpiNNaker
retina_device = p.external_devices.PushBotSpiNNakerLinkRetinaDevice(
        spinnaker_link_id=spinnaker_link,
        protocol=pushbot_protocol,
        resolution=retina_resolution,
        n_machine_vertices=n_machine_vertices)

# Motor devices for connection
devices = [motor_0, motor_1]

###########################################
#  Create populations
###########################################

# Retina population ( number of neurons:n_pixel*n_pixel*2 )
pushbot_retina = p.Population(
    retina_resolution.value.n_neurons, retina_device, label=retina_label)
p.external_devices.activate_live_output_for(
    pushbot_retina, database_notify_port_num=retina_viewer.port)

# A conceptual neuron population to drive motor neurons
# (0: left, 1:right)
driver_pop = p.Population(2, p.IF_curr_exp(), label='driver')

###########################################
#  Connections lists
###########################################

# Connection from exc_pop to driver_pop
# Number of columns to separate for left, right and forward
n_conn = 3 * (retina_resolution.value.pixels / 8)

# The events coming from the specified columns of exc_pop population
# will be connected to the relevant neuron in driver_pop
# Starting column of the left-side column group
start_of_left = 0

# Last column of the left-side column group
end_of_left = n_conn

# Starting column of the right-side column group
start_of_right = retina_resolution.value.pixels - n_conn

# Last column of the right-side column group
end_of_right = retina_resolution.value.pixels

# First row to consider in any group of pixels 
# (i.e. ignore bright top lights)
start_row = 30

# Last row to consider in any group of pixels
# (i.e. ignore bright bottom lights)
end_row = retina_resolution.value.pixels - 40

# Connection weights for this connection list
w_conn = 0.2

# Connection delays for this connection list
d_conn = 1

# Array containing id of each neuron
arr = numpy.arange(retina_resolution.value.n_neurons / 2)

# Determines which neuron IDs are on the left group
id_to_left = (arr % retina_resolution.value.pixels) < end_of_left
id_to_left = (
    (arr // retina_resolution.value.pixels >= start_row)
    & (arr // retina_resolution.value.pixels < end_row)) & id_to_left

# Determines which neuron IDs are on the right group
id_to_right = (arr % retina_resolution.value.pixels) >= start_of_right
id_to_right = (
    (arr // retina_resolution.value.pixels >= start_row)
    & (arr // retina_resolution.value.pixels < end_row)) & id_to_right

# Extracts the neuron IDs to be connected to the left neuron of driver_pop
id_to_left = numpy.extract(id_to_left, arr)
print("left =", id_to_left)

# Extracts the neuron IDs to be connected to the right neuron of driver_pop
id_to_right = numpy.extract(id_to_right, arr)
print("right =", id_to_right)

# Connection list: (source neuron, target neuron, weight, delay)
# Creates connection list to connect left neuron
conn_list_left = [(i, 0, w_conn, d_conn) for i in id_to_left]

# Creates connection list to connect right neuron
conn_list_right = [(i, 1, w_conn, d_conn) for i in id_to_right]

# Winner-takes-all connections from driver_pop to motor neurons
w_motor = 1
conn_motor_exc = [(0, 1, w_motor, 1), (1, 0, w_motor, 1)]
conn_motor_inh = [(0, 0, w_motor, 1), (1, 1, w_motor, 1)]


###########################################
#  Projections
###########################################

# w_inh=90.#70

# Pushbot motor neuron population
# Each member of population is a LIF neuron without a threshold
pushbot = p.Population(
    len(devices), p.external_devices.PushBotLifSpinnakerLink(
        protocol=pushbot_protocol,
        devices=devices,
        tau_syn_E=5.0, tau_syn_I=5.0),
    label="PushBot"
)

p.Projection(
    pushbot_retina, driver_pop,
    p.FromListConnector(conn_list_left + conn_list_right))
p.Projection(
    driver_pop, pushbot, p.FromListConnector(conn_motor_exc),
    receptor_type='excitatory')
p.Projection(
    driver_pop, pushbot, p.FromListConnector(conn_motor_inh),
    receptor_type='inhibitory')


###########################################
#  Simulation
###########################################

# Record spikes and membrane potentials
# driver_pop.record(['spikes','v'])

retina_viewer.run_until_closed()

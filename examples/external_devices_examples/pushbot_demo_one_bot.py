"""

@author: emec
"""
###########################################
#  Import libraries
###########################################
import spynnaker8 as p
import numpy as np


###########################################
#  Connection and simulation specifications
###########################################

# The UART port on IO Board the pushbot connected to
UART_ID = 0

# SpiNNaker link ID (IO Board connected to)
spinnaker_link = 1

# IP Address of the SpiNNaker board connected to (or None if only one)
board_address = None  # "10.162.242.14" #"10.162.242.13" "192.168.240.1"

# Retina resolution
retina_resolution = \
    p.external_devices.PushBotRetinaResolution.DOWNSAMPLE_16_X_16

# Simulation time [ms]
simtime = 240000

# Simulate with 1 ms time step
p.setup(1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 150)


###########################################
#  Create external devices
###########################################
# Create the pushbot protocols to pass to the devices of Pushbot
# (i.e. motors, laser, etc)
pushbot_protocol = p.external_devices.MunichIoSpiNNakerLinkProtocol(
    mode=p.external_devices.MunichIoSpiNNakerLinkProtocol.MODES.PUSH_BOT,
    uart_id=UART_ID)

# Create motor devices on SpiNNaker
motor_0 = p.external_devices.PushBotSpiNNakerLinkMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_0_PERMANENT, pushbot_protocol,
    spinnaker_link, board_address=board_address)

motor_1 = p.external_devices.PushBotSpiNNakerLinkMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_1_PERMANENT, pushbot_protocol,
    spinnaker_link, board_address=board_address)

# Create retina device on SpiNNaker
retina_device = p.external_devices.PushBotSpiNNakerLinkRetinaDevice(
        spinnaker_link_id=spinnaker_link,
        board_address=board_address,
        protocol=pushbot_protocol,
        resolution=retina_resolution)

# Motor devices for connection
devices = [motor_0, motor_1]

###########################################
#  Create populations
###########################################

# Pushbot motor neuron population
# Each member of population is a LIF neuron without a threshold
pushbot = p.Population(
    len(devices), p.external_devices.PushBotLifSpinnakerLink(
        protocol=pushbot_protocol,
        devices=devices,
        tau_syn_E=100.0),
    label="PushBot"
)

# Retina population ( num of neurons:n_pixel*n_pixel*2 )
pushbot_retina = p.Population(
    retina_resolution.value.n_neurons / 2, retina_device)


# Network implementation
# An excitatory population is inhibited by an inhibitory population and
# only the neurons receiving enough number of retina events to break the
# inhibition will activate the relevant neurons to drive the robot
exc_pop = p.Population(
    retina_resolution.value.n_neurons / 2,
    p.IF_curr_exp(cm=0.75, tau_m=1.0),
    label='exc_pop')
inh_pop = p.Population(
    retina_resolution.value.n_neurons / 2,
    p.IF_curr_exp(cm=0.75, tau_m=1.0, i_offset=13, tau_refrac=0.01),
    label='inh_pop')

# A conceptual neuron population to drive motor neurons
# (0: left, 1:right, 2: forward)
driver_pop = p.Population(3, p.IF_curr_exp(cm=0.75, tau_m=1.0), label='driver')

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

# Connection weights for this connection list
w_conn = 20

# Connection delays for this connection list
d_conn = 1

# Array containing id of each neuron
arr = np.arange(retina_resolution.value.n_neurons / 2)

# Determines which neuron IDs are on the left group
id_to_left = (arr % retina_resolution.value.pixels) < end_of_left

# Determines which neuron IDs are on the right group
id_to_right = (arr % retina_resolution.value.pixels) > start_of_right

# Determines which neuron IDs are in the middle
id_to_middle_up_1 = (arr % retina_resolution.value.pixels) <= start_of_right

# Determines which neuron IDs are in the middle
id_to_middle_up_2 = (arr % retina_resolution.value.pixels) >= end_of_left

# Determines which neuron IDs are on the upper part
id_to_middle_up_3 = (
    (arr / retina_resolution.value.pixels) <
    (retina_resolution.value.pixels / 2))

# The variable to determine which neuron IDs are on the middle-up
id_to_middle_up = id_to_middle_up_1 & id_to_middle_up_2  # & id_to_middle_up_3

# Extracts the neuron IDs to be connected to the left neuron of driver_pop
id_to_left = np.extract(id_to_left, arr)
print "left =", id_to_left

# Extracts the neuron IDs to be connected to the right neuron of driver_pop
id_to_right = np.extract(id_to_right, arr)
print "right =", id_to_right

# Extracts the neuron IDs to be connected to the forward neuron of driver_pop
id_to_middle_up = np.extract(id_to_middle_up, arr)
print "middle =", id_to_middle_up

# Conn list: (source neuron, target neuron, weight, delay)
# Creates connection list to connect left neuron
conn_list_left = [(i, 0, w_conn, d_conn) for i in id_to_left]

# Creates connection list to connect right neuron
conn_list_right = [(i, 1, w_conn, d_conn) for i in id_to_right]

# Creates connection list to connect forward neuron
conn_list_middle_up = [(i, 2, w_conn, d_conn) for i in id_to_middle_up]

# Concatenates the lists into one list
conn_list = conn_list_left + conn_list_right + conn_list_middle_up

# Winner-takes-all connections from driver_pop to motor neurons
w_motor = 0.1
conn_motor_exc = [
    (0, 1, w_motor * 2, 1), (1, 0, w_motor * 2, 1)]
conn_motor_inh = [(0, 0, w_motor * 2, 1), (1, 1, w_motor * 2, 1)] + [
    (2, 0, w_motor, 1), (2, 1, w_motor, 1)]

# Creates connection list from retina population to exc_pop
# Each neuron in retina population excites the neuron with the same ID in
# exc_pop and its 8 neighbours including the diagonals
w_conn, w_inh = (3.7, 100)  # (3.7,95)#(3.8, 90.)# (3.7,65,15)

conn_list_local = []
n_row = retina_resolution.value.pixels
conn_inc = np.array(
    [-n_row-1, -n_row, -n_row + 1, -1, 0, 1, n_row - 1, n_row, n_row + 1])
cont_inc = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
for i in range(retina_resolution.value.n_neurons):
    sources = i + conn_inc
    cont_1 = i / n_row
    cont_2 = sources / n_row + cont_inc
    for j, k in zip(sources, cont_2):
        if j >= 0 and j < retina_resolution.value.n_neurons and cont_1 == k:
            conn_list_local.append((i, j, w_conn, 1))


###########################################
#  Projections
###########################################

# w_inh=90.#70

p.Projection(pushbot_retina, exc_pop, p.FromListConnector(conn_list_local))
p.Projection(
    inh_pop, exc_pop, p.OneToOneConnector(),
    synapse_type=p.StaticSynapse(weight=w_inh, delay=1.0),
    receptor_type='inhibitory')
p.Projection(exc_pop, driver_pop, p.FromListConnector(conn_list))
p.Projection(driver_pop, pushbot, p.FromListConnector(conn_motor_exc))
p.Projection(
    driver_pop, pushbot, p.FromListConnector(conn_motor_inh),
    receptor_type='inhibitory')


###########################################
#  Simulation
###########################################

# Record spikes
# exc_pop.record(['spikes'])
# inh_pop.record(['spikes'])
# driver_pop.record(['spikes'])

# Record spikes and membrane potentials
# driver_pop.record(['spikes','v'])

# Add a retina viewer to see what the pushbot sees
viewer = p.external_devices.PushBotRetinaViewer(
    retina_resolution.value, port=17895)
p.external_devices.activate_live_output_for(
    pushbot_retina, port=viewer.local_port, notify=False)
viewer.start()

# Start simulation
p.run_forever()
# p.run(simtime)

viewer.join()

p.end()

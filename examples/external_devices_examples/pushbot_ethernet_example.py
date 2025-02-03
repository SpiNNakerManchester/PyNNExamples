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

import pyNN.spiNNaker as p

p.setup(1.0)

# Set up the PushBot devices
pushbot_protocol = p.external_devices.MunichIoSpiNNakerLinkProtocol(
    mode=p.external_devices.protocols.MUNICH_MODES.PUSH_BOT, uart_id=0)
motor_0 = p.external_devices.PushBotEthernetMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_0_PERMANENT, pushbot_protocol)
motor_1 = p.external_devices.PushBotEthernetMotorDevice(
    p.external_devices.PushBotMotor.MOTOR_1_PERMANENT, pushbot_protocol)
speaker = p.external_devices.PushBotEthernetSpeakerDevice(
    p.external_devices.PushBotSpeaker.SPEAKER_TONE, pushbot_protocol)
laser = p.external_devices.PushBotEthernetLaserDevice(
    p.external_devices.PushBotLaser.LASER_ACTIVE_TIME, pushbot_protocol,
    start_total_period=1000)
led_front = p.external_devices.PushBotEthernetLEDDevice(
    p.external_devices.PushBotLED.LED_FRONT_ACTIVE_TIME, pushbot_protocol,
    start_total_period=1000)
led_back = p.external_devices.PushBotEthernetLEDDevice(
    p.external_devices.PushBotLED.LED_BACK_ACTIVE_TIME, pushbot_protocol,
    start_total_period=1000)

weights = {
    motor_0: 10.0,
    motor_1: 10.0,
    speaker: 100.0,
    laser: 100.0,
    led_front: 100.0,
    led_back: 100.0,
}

devices = [motor_0, motor_1, speaker, laser, led_front, led_back]

# Set up the PushBot control
pushbot = p.external_devices.EthernetControlPopulation(
    len(devices), p.external_devices.PushBotLifEthernet(
        protocol=pushbot_protocol,
        devices=devices,
        pushbot_ip_address="10.10.10.1",
        pushbot_port=3000,
        # "pushbot_ip_address": "127.0.0.1",
        tau_syn_E=500.0),
    label="PushBot"
)

# Send in some spikes
stimulation = p.Population(
    len(devices), p.SpikeSourceArray(
        spike_times=[[i * 1000] for i in range(len(devices))]),
    label="input"
)

connections = [
    (i, i, weights[device], 1) for i, device in enumerate(devices)
]
p.Projection(stimulation, pushbot, p.FromListConnector(connections))

retina_resolution = \
    p.external_devices.PushBotRetinaResolution.DOWNSAMPLE_64_X_64
pushbot_retina = p.external_devices.EthernetSensorPopulation(
    p.external_devices.PushBotEthernetRetinaDevice(
        protocol=pushbot_protocol,
        resolution=retina_resolution,
        pushbot_ip_address="10.10.10.1",
        pushbot_port=3000,
        retina_injector_label="Retina"
    ))

retina_viewer = p.external_devices.PushBotRetinaViewer(
    retina_resolution, pushbot_retina.label)
p.external_devices.activate_live_output_for(
    pushbot_retina, database_notify_port_num=retina_viewer.port)

retina_viewer.run(len(devices) * 1000)

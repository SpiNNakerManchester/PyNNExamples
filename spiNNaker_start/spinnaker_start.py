# Copyright (c) 2018 The University of Manchester
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


from __future__ import print_function
from six import iteritems
from threading import Thread, RLock, Condition
from matplotlib import pyplot, animation
from numpy.random.mtrand import shuffle
import functools
import numpy
import matplotlib
import time
import operator
import struct
import six
import sys
import logging
import pickle
import traceback

from spinnman.constants import ROUTER_REGISTER_P2P_ADDRESS,\
    SYSTEM_VARIABLE_BASE_ADDRESS, address_length_dtype
from spinnman.processes import AbstractMultiConnectionProcess
from spinnman.messages.scp.impl import ReadMemory, GetChipInfo
from spinnman.model import P2PTable
from spinnman.processes.get_version_process import GetVersionProcess
from spinnman.messages.sdp.sdp_flag import SDPFlag
from spinnman.messages.scp.impl.read_memory import _SCPReadMemoryResponse
from spinnman.messages.spinnaker_boot import SystemVariableDefinition
from spinnman.connections.udp_packet_connections \
    import SCAMPConnection, BootConnection
from spinnman.processes import RoundRobinConnectionSelector
from spinnman.messages.scp import SCPRequestHeader
from spinnman.messages.scp.abstract_messages import AbstractSCPRequest
from spinnman.messages.scp.enums import SCPCommand
from spinnman.messages.sdp import SDPHeader
from spinnman.messages.scp.impl.get_version_response import GetVersionResponse
from spinnman.messages.spinnaker_boot import SpinnakerBootMessages

from spinn_utilities.overrides import overrides

from spalloc.job import Job
from spalloc.states import JobState

from spinn_machine.machine import Machine

print_lock = RLock()

file_open_lock = Condition()
n_open_files = 0


def warn(*args):
    with print_lock:
        print(*args)


def open_file(filename, mode):
    global file_open_lock, n_open_files
    with file_open_lock:
        while n_open_files >= 200:
            file_open_lock.wait()
        n_open_files += 1
    return open(filename, mode)


def close_file(open_file):
    global file_open_lock, n_open_files
    with file_open_lock:
        open_file.close()
        n_open_files -= 1
        file_open_lock.notify_all()


def write_data_to_file(write_file, data_list):
    write_file.write(struct.pack("<I", len(data_list)))
    for timestamp, data in data_list:
        write_file.write(struct.pack("<f", timestamp))
        pickle.dump(data, write_file)


def read_data_from_file(read_file):
    data_list = list()
    n_items = struct.unpack("<I", read_file.read(4))[0]
    for _ in range(n_items):
        timestamp = struct.unpack("<f", read_file.read(4))[0]
        data = pickle.load(read_file)
        data_list.append((timestamp, data))
    return data_list


class GetVersion(AbstractSCPRequest):
    __slots__ = []

    def __init__(self, x, y):
        super(GetVersion, self).__init__(
            SDPHeader(
                flags=SDPFlag.REPLY_EXPECTED_NO_P2P, destination_port=0,
                destination_cpu=0, destination_chip_x=x,
                destination_chip_y=y),
            SCPRequestHeader(command=SCPCommand.CMD_VER))

    @overrides(AbstractSCPRequest.get_scp_response)
    def get_scp_response(self):
        return GetVersionResponse()


class ReadSV(AbstractSCPRequest):
    __slots__ = []

    def __init__(self, x, y, variable, size=None):
        base_address = (SYSTEM_VARIABLE_BASE_ADDRESS + variable.offset)
        if size is None:
            size = variable.data_type.value
        super(ReadSV, self).__init__(
            SDPHeader(
                flags=SDPFlag.REPLY_EXPECTED_NO_P2P, destination_port=0,
                destination_cpu=0, destination_chip_x=x,
                destination_chip_y=y),
            SCPRequestHeader(command=SCPCommand.CMD_READ),
            argument_1=base_address, argument_2=size,
            argument_3=address_length_dtype[
                (base_address % 4, size % 4)].value)

    @overrides(AbstractSCPRequest.get_scp_response)
    def get_scp_response(self):
        return _SCPReadMemoryResponse()


class ReadNetinitPhaseProcess(AbstractMultiConnectionProcess):

    def __init__(self, x, y, connection_selector, core_counter, save, load):
        super(ReadNetinitPhaseProcess, self).__init__(
            connection_selector, timeout=0.5, n_retries=0)
        self._x = x
        self._y = y
        self._core_counter = core_counter
        self._save = save
        self._load = load
        self._last_x = None
        self._last_y = None
        self._last_max_width = None
        self._last_max_height = None
        self._last_phase = None
        self._last_p2p_guess_x = None
        self._last_p2p_guess_y = None
        self._last_count = None
        self._first_response = True
        self._done = False
        self._index = 0
        self._thread = Thread(target=self._run)
        self._p2p_active_data = list()
        self._netinit_phase_data = list()
        self._start_time = time.time()
        if load:
            read_file = open_file(
                "record/netinit_{}_{}.dat".format(x, y), "rb")
            self._p2p_active_data = read_data_from_file(read_file)
            self._netinit_phase_data = read_data_from_file(read_file)
            close_file(read_file)

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def close(self):
        self._done = True

    def running(self):
        return not self._done

    def _process_p2p_active_response(self, response):
        count = struct.unpack_from("<H", response.data, response.offset)[0]
        self._process_p2p_active_count(count)

    def _process_p2p_active_count(self, count):
        if count != self._last_count:
            warn("{:03d} {:03d} {:05d} has {} active P2P entries".format(
                self._x, self._y, self._index, count))
            self._last_count = count
            self._core_counter.set_ethernet_n_routes(self._x, self._y, count)
            self._index += 1
            if self._save:
                self._p2p_active_data.append(
                    (time.time() - self._start_time, count))

    def _process_netinit_phase_response(self, response):
        value = struct.unpack_from("<B", response.data, response.offset)[0]
        self._process_netinit_phase_value(value)

    def _process_netinit_phase_value(self, value):
        if self._first_response:
            self._core_counter.set_ethernet_responding(self._x, self._y)
            self._first_response = False

        if self._last_phase != value:
            self._core_counter.set_ethernet_netinit_phase(
                self._x, self._y, value)
            warn("{:03d} {:03d} {:05d} moved to phase {:03d}".format(
                self._x, self._y, self._index, value))
            self._index += 1
            self._last_phase = value
            if self._save:
                self._netinit_phase_data.append(
                    (time.time() - self._start_time, value))

        if value == 0xFF:
            self._done = True
        elif value == 3 and not self._load:
            self._send_request(
                ReadSV(
                    self._x, self._y,
                    SystemVariableDefinition.n_active_peer_to_peer_addresses),
                callback=self._process_p2p_active_response)

    def _run(self):
        if self._load:
            netinit_phase_index = 0
            p2p_active_index = 0
            while (netinit_phase_index < len(self._netinit_phase_data) or
                   p2p_active_index < len(self._p2p_active_data)):
                callback = None
                next_time = None
                response = None
                netinit_time = sys.maxint
                netinit_data = None
                p2p_time = sys.maxint
                p2p_data = None
                if netinit_phase_index < len(self._netinit_phase_data):
                    netinit_time, netinit_data = self._netinit_phase_data[
                        netinit_phase_index]
                if p2p_active_index < len(self._p2p_active_data):
                    p2p_time, p2p_data = self._p2p_active_data[
                        p2p_active_index]
                if netinit_time < p2p_time:
                    next_time = netinit_time
                    callback = self._process_netinit_phase_value
                    response = netinit_data
                    netinit_phase_index += 1
                else:
                    next_time = p2p_time
                    callback = self._process_p2p_active_count
                    response = p2p_data
                    p2p_active_index += 1
                time_to_sleep = next_time - (
                    time.time() - self._start_time)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                callback(response)
        else:
            while not self._done:
                self._send_request(
                    ReadSV(self._x, self._y,
                           SystemVariableDefinition.netinit_phase),
                    callback=self._process_netinit_phase_response)
                self._finish()
                time.sleep(0.1)
            if self._save:
                write_file = open_file(
                    "record/netinit_{}_{}.dat".format(self._x, self._y), "wb")
                write_data_to_file(write_file, self._p2p_active_data)
                write_data_to_file(write_file, self._netinit_phase_data)
                close_file(write_file)


class GetP2PTableProcess(AbstractMultiConnectionProcess):

    def __init__(self, connection_selector, width, height, save, load):
        super(GetP2PTableProcess, self).__init__(connection_selector)
        self._width = width
        self._height = height
        self._p2p_column_data = [None] * width
        self._save = save
        self._load = load

        if load:
            p2p_data = open_file("record/p2p.dat", "rb")
            for column in range(width):
                col_length = struct.unpack("<I", p2p_data.read(4))[0]
                col_data = p2p_data.read(col_length)
                col_offset = struct.unpack("<I", p2p_data.read(4))[0]
                self._p2p_column_data[column] = (col_data, col_offset)
            close_file(p2p_data)

    def _receive_p2p_data(self, column, scp_read_response):
        self._p2p_column_data[column] = (
            scp_read_response.data, scp_read_response.offset)

    def get_p2p_table(self):
        if self._load:
            return P2PTable(self._width, self._height, self._p2p_column_data)

        # Get the P2P table - 8 entries are packed into each 32-bit word
        p2p_column_bytes = P2PTable.get_n_column_bytes(self._height)
        for column in range(self._width):
            offset = P2PTable.get_column_offset(column)
            self._send_request(
                ReadMemory(
                    x=0, y=0,
                    base_address=(ROUTER_REGISTER_P2P_ADDRESS + offset),
                    size=p2p_column_bytes),
                functools.partial(self._receive_p2p_data, column))
        self._finish()
        self.check_for_error()
        if save:
            p2p_data = open_file("record/p2p.dat", "wb")
            for column in range(self._width):
                data, offset = self._p2p_column_data[column]
                p2p_data.write(struct.pack("<I", len(data)))
                p2p_data.write(data)
                p2p_data.write(struct.pack("<I", offset))
            close_file(p2p_data)
        return P2PTable(self._width, self._height, self._p2p_column_data)


class ReadBoardProcess(AbstractMultiConnectionProcess):
    """ A process for getting the machine details over a set of connections.
    """

    def __init__(
            self, eth_x, eth_y, width, height, p2p_table, connection_selector,
            core_counter, save, load):
        # pylint: disable=too-many-arguments
        super(ReadBoardProcess, self).__init__(
            connection_selector, n_retries=10, timeout=5.0, n_channels=1,
            intermediate_channel_waits=0)
        self._eth_x = eth_x
        self._eth_y = eth_y
        self._width = width
        self._height = height
        self._p2p_table = p2p_table
        self._core_counter = core_counter
        self._save = save
        self._load = load
        self._thread = Thread(target=self._read_board)
        self._data = list()
        self._start_time = time.time()
        if load:
            read_file = open_file(
                "record/board_{}_{}.dat".format(eth_x, eth_y), "rb")
            self._data = read_data_from_file(read_file)
            close_file(read_file)

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def _receive_chip_info(self, scp_read_chip_info_response):
        chip_info = scp_read_chip_info_response.chip_info
        self._process_chip_info(chip_info.x, chip_info.y, chip_info.n_cores)

    def _process_chip_info(self, x, y, n_cores):
        if self._save:
            self._data.append(((time.time() - self._start_time),
                               (x, y, n_cores)))
        self._core_counter.add_cores(
            self._eth_x, self._eth_y, x, y, n_cores)

    def _read_board(self):
        if self._load:
            index = 0
            while index < len(self._data):
                timestamp, (b_x, b_y, n_cores) = self._data[index]
                index += 1
                sleep_time = timestamp - (time.time() - self._start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._process_chip_info(b_x, b_y, n_cores)
            return

        # Get the set of chips to read
        chips = [((x + self._eth_x) % self._width,
                  (y + self._eth_y) % self._height)
                 for x in range(0, 8) for y in range(0, 8)
                 if (x, y) not in Machine.BOARD_48_CHIP_GAPS]
        chips = filter(
            lambda coords: self._p2p_table.is_route(coords[0], coords[1]),
            chips)
        shuffle(chips)

        # Get the chip information for each chip
        for (x, y) in chips:
            self._send_request(GetChipInfo(x, y), self._receive_chip_info)
        self._finish()
        try:
            self.check_for_error()
        except Exception as e:
            warn(e)
        if self._save:
            write_file = open_file(
                "record/board_{}_{}.dat".format(
                    self._eth_x, self._eth_y), "wb")
            write_data_to_file(write_file, self._data)
            close_file(write_file)


class CoreCounter(object):

    def __init__(self, width, height):
        self._total_cores = 0
        self._update_lock = RLock()
        self._ready = False
        self._notify_lock = Condition()

        self._width = width
        self._height = height
        self._image_data = numpy.zeros((height, width, 3), dtype="uint8")
        self._ax = None
        self._text = None
        self._fig = None
        self._max_value = [175, 150, 125, 100]

    def run(self):
        matplotlib.rcParams['toolbar'] = 'None'
        self._fig, self._ax = pyplot.subplots()
        self._fig.canvas.set_window_title("SpiNNaker")
        self._image = self._ax.imshow(
            self._image_data, origin="lower")
        self._text = self._ax.text(
            self._width // 2, self._height // 2, "0,000,000",
            ha="center", va="center", color="#FFFFFF", fontsize=40)
        self._text.set_alpha(0.9)
        self._ani = animation.FuncAnimation(
            self._fig, self._update, interval=40, blit=True)
        pyplot.subplots_adjust(0.05, 0.05, 1.0, 1.0)
        self._fig.canvas.mpl_connect('key_release_event', self._key_press)
        self._fig.canvas.mpl_connect('close_event', self._close)
        pyplot.show()

    def _close(self, _event):
        with self._notify_lock:
            self._ready = True
            self._notify_lock.notify_all()

    def _key_press(self, _event):
        with self._notify_lock:
            self._ready = True
            self._notify_lock.notify_all()

    def wait_until_ready(self):
        with self._notify_lock:
            while not self._ready:
                self._notify_lock.wait()

    def _update(self, _frame):
        self._image.set_array(self._image_data)
        bbox = self._ax.get_window_extent().transformed(
            self._fig.dpi_scale_trans.inverted())
        self._text.set_fontsize(bbox.width * 12.5)
        self._text.set_text("{:08,d}".format(self._total_cores))
        return [self._image, self._text]

    def _get_max_value(self, x, y):
        index = 0
        if (x, y) == (0, 0):
            index = 0
        elif (x % 8) == 0 and (y % 8) == 0:
            index = 3
        elif (x % 8) == 0:
            index = 1
        elif (y % 8) == 0:
            index = 2
        return self._max_value[index]

    def add_ethernet(self, x, y):
        colour = self._get_max_value(x, y)
        self._image_data[y, x] = [colour, 0, 0]

    def set_ethernet_responding(self, x, y):
        colour = self._get_max_value(x, y)
        self._image_data[y, x] = [colour, colour, 0]

    def set_ethernet_booted(self, x, y):
        colour = self._get_max_value(x, y)
        self._image_data[y, x] = [colour, colour, 0]

    def set_ethernet_netinit_phase(self, x, y, phase):
        colour = self._get_max_value(x, y)
        self._image_data[y, x] = [colour, colour, phase]

    def set_ethernet_n_routes(self, x, y, n_routes):
        colour = self._get_max_value(x, y)
        max_n_routes = self._width * self._height
        value = int(round((float(colour) / max_n_routes) * n_routes))
        self._image_data[y, x] = [255 - value, 0, 255 - value]

    def add_cores(self, eth_x, eth_y, x, y, n_cores):
        colour = self._get_max_value(eth_x, eth_y)
        with self._update_lock:
            self._total_cores += n_cores
            value = int(round((float(colour) / 18.0) * n_cores))
            self._image_data[y, x] = [0, value, 0]


class MainThread(object):
    def __init__(self, core_counter, job, save, load):
        self._done = False
        self._thread = Thread(
            target=self.run, args=[core_counter, job, save, load])
        self._close_lock = RLock()
        self._boot_threads = list()
        self._processes = list()

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def close(self):
        with self._close_lock:
            self._done = True
            for thread in self._boot_threads:
                thread.close()

    def run(self, core_counter, job, save, load):
        job_connections = list()
        for (x, y), ip_address in iteritems(job.connections):
            job_connections.append((x, y, ip_address))
        job_connections.sort(key=operator.itemgetter(0, 1))
        with job:
            warn("Waiting for user to start process")
            core_counter.wait_until_ready()
            warn("Process starting")

            # Get a list of connections to the machine and start a thread for
            # each listening for the boot to be done
            connections = list()
            root_connection = None
            boot_connection = None
            with self._close_lock:
                if not self._done:
                    for x, y, ip_address in job_connections:
                        connection = None
                        connection = SCAMPConnection(
                            x, y, remote_host=ip_address)
                        connections.append(connection)
                        core_counter.add_ethernet(x, y)
                        if (x, y) == (0, 0):
                            root_connection = connection
                            if not load:
                                boot_connection = BootConnection(
                                    remote_host=ip_address)
                        reader = ReadNetinitPhaseProcess(
                            x, y, RoundRobinConnectionSelector([connection]),
                            core_counter, save, load)
                        reader.start()
                        self._boot_threads.append(reader)

            # Connect to the machine and boot it
            boot_done = False
            if load:
                boot_done = True
            tries = 3
            while not self._done and not boot_done and tries > 0:
                warn("Booting machine", boot_connection.remote_ip_address)
                boot_messages = SpinnakerBootMessages(board_version=5)
                for boot_message in boot_messages.messages:
                    boot_connection.send_boot_message(boot_message)
                time.sleep(2.0)
                try:
                    warn("Waiting for boot to complete")
                    version_read_ok = False
                    while not boot_done and not self._done:
                        try:
                            get_version = GetVersionProcess(
                                RoundRobinConnectionSelector(
                                    [root_connection]))
                            version = get_version.get_version(0, 0, 0)
                            version_read_ok = True
                            core_counter.set_ethernet_booted(0, 0)
                            if (version.x !=
                                    AbstractSCPRequest.DEFAULT_DEST_X_COORD and
                                    version.y !=
                                    AbstractSCPRequest.DEFAULT_DEST_Y_COORD):
                                boot_done = True
                                core_counter.set_ethernet_booted(0, 0)
                            else:
                                time.sleep(1.0)
                        except Exception:
                            if not version_read_ok:
                                six.reraise(*sys.exc_info())
                except Exception:
                    warn("Boot failed, retrying")
                    tries -= 1
            if not boot_done and not self._done:
                raise Exception("Could not boot machine")

            for thread in self._boot_threads:
                thread.join()

            # Read the P2P table to know which chips should exist
            if boot_done:
                warn("Reading P2P Table")
                p2p_process = GetP2PTableProcess(
                    RoundRobinConnectionSelector(
                        [root_connection]), job.width, job.height, save, load)
                p2p_table = p2p_process.get_p2p_table()

                # Create a reader thread for each connection,
                # and read the cores
                for connection in connections:
                    process = ReadBoardProcess(
                        connection.chip_x, connection.chip_y,
                        job.width, job.height, p2p_table,
                        RoundRobinConnectionSelector([connection]),
                        core_counter, save, load)
                    process.start()
                    self._processes.append(process)

                # Wait for everything to finish
                for process in self._processes:
                    process.join()


class MockJob(object):
    def __init__(self, width, height):
        self._boards = list()
        self._height = height
        self._width = width

    def add_board(self, x, y):
        self._boards.append((x, y))

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    @property
    def n_boards(self):
        return len(self._boards)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def connections(self):
        return {(x, y): "127.0.0.1" for (x, y) in self._boards}

    def destroy(self):
        pass

    def wait_for_state_change(self, state):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save = False
    load = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            save = True
        elif sys.argv[1] == "--load":
            load = True

    job = None
    if load:
        load_file = open_file("record/job.dat", "rb")
        width, height, n_boards = struct.unpack("<III", load_file.read(12))
        job = MockJob(width, height)
        for _ in range(n_boards):
            x, y = struct.unpack("<II", load_file.read(8))
            job.add_board(x, y)
        print(job.width, job.height)
        close_file(load_file)
    else:
        job = Job(1200, hostname="spinn-test.cs.man.ac.uk",
                  owner="SpiNNaker Start")
    try:
        job.wait_for_state_change(JobState.queued)
        if save:
            save_file = open_file("record/job.dat", "wb")
            save_file.write(struct.pack(
                "<III", job.width, job.height, len(job.connections)))
            for (x, y), _ in iteritems(job.connections):
                save_file.write(struct.pack("<II", x, y))
            close_file(save_file)

        # Create GUI
        core_counter = CoreCounter(job.width, job.height)

        # Run task in thread
        main_thread = MainThread(core_counter, job, save, load)
        main_thread.start()

        # Run GUI
        core_counter.run()

        # Close and wait for completion
        main_thread.close()
        main_thread.join()
    except Exception:
        traceback.print_exc()
        job.destroy()

from __future__ import print_function

import logging
import struct
import traceback
import numpy
import matplotlib
import time
import operator
import sys

from six import iteritems
from threading import Thread, RLock, Condition
from matplotlib import pyplot, animation
from numpy.random.mtrand import shuffle
from collections import defaultdict

from spinnman.constants import ROUTER_REGISTER_P2P_ADDRESS, \
    SYSTEM_VARIABLE_BASE_ADDRESS, address_length_dtype
from spinnman.messages.scp.impl import ReadMemory, GetChipInfo
from spinnman.model import P2PTable
from spinnman.processes.get_version_process import GetVersionProcess
from spinnman.messages.sdp.sdp_flag import SDPFlag
from spinnman.messages.spinnaker_boot import SystemVariableDefinition
from spinnman.messages.scp.impl.read_memory import _SCPReadMemoryResponse
from spinnman.exceptions import SpinnmanTimeoutException, SpinnmanIOException
from spinn_utilities.ordered_set import OrderedSet
from spinnman.messages.scp.enums.scp_result import SCPResult
from spalloc.states import JobState
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

from spinn_machine.machine import Machine

print_lock = RLock()


def warn(*args):
    with print_lock:
        print(*args)


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


class SCPComms(object):

    def __init__(self):
        self._sequence = 0
        self._callback_args = dict()

    def _send_scp(
            self, connection, x, y, ip_address, message, callback, *args):
        message.scp_request_header.sequence = self._sequence
        self._callback_args[ip_address, self._sequence] =\
            (message, callback, args)
        self._sequence = (self._sequence + 1) % 65536
        try:
            connection.send_scp_request_to(message, x, y, ip_address)
        except SpinnmanIOException as e:
            print("Error sending to", ip_address, e)

    def _process_scp(self, connection):
        result, sequence, data, offset, ip_address, _port = \
            connection.receive_scp_response_with_address(timeout=0.5)
        stored = self._callback_args.get((ip_address, sequence), None)
        if stored is not None:
            message, callback, args = stored
            response = message.get_scp_response()
            try:
                response.read_bytestring(data, offset)
                callback(response, *args)
            except Exception:
                etype, value, tb = sys.exc_info()
                self._on_error(
                    message, response, result, etype, value, tb, *args)

    def _on_error(self, message, response, result, etype, value, tb, *args):
        traceback.print_exception(etype, value, tb)


class ReadNetinitPhaseProcess(SCPComms):

    def __init__(self, connections, core_counter):
        super(ReadNetinitPhaseProcess, self).__init__()
        self._connections = connections
        self._last_x = defaultdict(lambda: None)
        self._last_y = defaultdict(lambda: None)
        self._last_max_width = defaultdict(lambda: None)
        self._last_max_height = defaultdict(lambda: None)
        self._last_phase = defaultdict(lambda: None)
        self._last_p2p_guess_x = defaultdict(lambda: None)
        self._last_p2p_guess_y = defaultdict(lambda: None)
        self._last_count = defaultdict(lambda: None)
        self._first_response = defaultdict(lambda: True)
        self._core_counter = core_counter
        self._finished = defaultdict(lambda: False)
        self._done = False
        self._n_to_go = len(connections)
        self._index = 0
        self._n_to_read = 0

    def close(self):
        self._done = True

    def running(self):
        return not self._done

    def _process_p2p_guess(self, response, x, y):
        p2p_guess_y, p2p_guess_x = struct.unpack_from(
            "<hh", response.data, response.offset)
        last_guess_x = self._last_p2p_guess_x[(x, y)]
        last_guess_y = self._last_p2p_guess_y[(x, y)]
        if p2p_guess_x != last_guess_x or p2p_guess_y != last_guess_y:
            self._last_p2p_guess_x[(x, y)] = p2p_guess_x
            self._last_p2p_guess_y[(x, y)] = p2p_guess_y
            warn("{:03d} {:03d} {:05d} thinks it is {} {}".format(
                x, y, self._index, p2p_guess_x, p2p_guess_y))
            self._index += 1

    def _process_p2p_active_response(self, response, x, y):
        count = struct.unpack_from("<H", response.data, response.offset)[0]
        if count != self._last_count[x, y]:
            warn("{:03d} {:03d} {:05d} has {} active P2P entries".format(
                x, y, self._index, count))
            self._last_count[x, y] = count
            self._core_counter.set_ethernet_n_routes(x, y, count)
        self._index += 1

    def _process_p2p_dims_response(self, response, x, y):
        guess_y, guess_x, max_height, max_width = struct.unpack_from(
            "<BBBB", response.data, response.offset)
        if guess_x != self._last_x[(x, y)] or guess_y != self._last_y[(x, y)]:
            warn("{:03d} {:03d} {:05d} thinks it is {} {}".format(
                x, y, self._index, guess_x, guess_y))
            self._last_x[(x, y)] = guess_x
            self._last_y[(x, y)] = guess_y
            self._index += 1
        if (max_width != self._last_max_width[(x, y)] or
                max_height != self._last_max_height[(x, y)]):
            warn("{:03d} {:03d} {:05d} thinks machine is {} x {}".format(
                x, y, self._index, max_width, max_height))
            self._last_max_width[(x, y)] = max_width
            self._last_max_height[(x, y)] = max_height
            self._index += 1

    def _process_netinit_phase_response(
            self, response, connection, x, y, ip_address):
        value = struct.unpack_from("<B", response.data, response.offset)[0]
        if self._first_response:
            self._core_counter.set_ethernet_responding(x, y)
            self._first_response = False

        if self._last_phase[(x, y)] != value:
            self._core_counter.set_ethernet_netinit_phase(x, y, value)
            warn("{:03d} {:03d} {:05d} moved to phase {:03d}".format(
                x, y, self._index, value))
            self._index += 1
            self._last_phase[(x, y)] = value
        if value == 0xFF:
            if not self._finished[(x, y)]:
                self._finished[(x, y)] = True
                self._n_to_go -= 1
                if self._n_to_go <= 0:
                    self._done = True
        elif value == 3:
            self._send_scp(
                connection, x, y, ip_address,
                ReadSV(
                    x, y,
                    SystemVariableDefinition.n_active_peer_to_peer_addresses),
                self._process_p2p_active_response, x, y)
            self._n_to_read += 1

    def run(self):
        connection = SCAMPConnection()
        while not self._done:
            timeout = False
            try:
                for x, y, ip_address in self._connections:
                    if not self._finished[x, y]:
                        self._send_scp(
                            connection, x, y, ip_address,
                            ReadSV(x, y,
                                   SystemVariableDefinition.netinit_phase),
                            self._process_netinit_phase_response,
                            connection, x, y, ip_address)
                        self._n_to_read += 1
                while self._n_to_read > 0:
                    self._process_scp(connection)
                    self._n_to_read -= 1
            except SpinnmanTimeoutException:
                timeout = True
            except Exception:
                traceback.print_exc()
            if not self._done and not timeout:
                time.sleep(0.5)


class GetP2PTableProcess(SCPComms):

    def __init__(self, width, height, x, y, ip_address, connection):
        super(GetP2PTableProcess, self).__init__()
        self._width = width
        self._height = height
        self._x = x
        self._y = y
        self._ip_address = ip_address
        self._connection = connection
        self._p2p_column_data = [None] * width
        self._cols_to_go = width
        self._done = False

    def _receive_p2p_data(self, scp_read_response, column):
        if self._p2p_column_data[column] is None:
            self._p2p_column_data[column] = (
                scp_read_response.data, scp_read_response.offset)
            self._cols_to_go -= 1
            if self._cols_to_go == 0:
                self._done = True

    def get_p2p_table(self):

        # Get the P2P table - 8 entries are packed into each 32-bit word
        p2p_column_bytes = P2PTable.get_n_column_bytes(self._height)
        while not self._done:
            n_sent = 0
            for column in range(self._width):
                if self._p2p_column_data[column] is None:
                    offset = P2PTable.get_column_offset(column)
                    self._send_scp(
                        self._connection, self._x, self._y, self._ip_address,
                        ReadMemory(
                            x=self._x, y=self._y,
                            base_address=(
                                ROUTER_REGISTER_P2P_ADDRESS + offset),
                            size=p2p_column_bytes),
                        self._receive_p2p_data, column)
                    n_sent += 1
            try:
                for _ in range(n_sent):
                    self._process_scp(self._connection)
            except SpinnmanTimeoutException:
                pass
        return P2PTable(self._width, self._height, self._p2p_column_data)


class ReadBoardProcess(SCPComms):
    """ A process for getting the machine details over a set of connections.
    """

    def __init__(
            self, width, height, p2p_table, job_connections, core_counter):
        super(ReadBoardProcess, self).__init__()
        self._core_counter = core_counter
        self._done = False
        self._n_sends = 0

        # Keep a set of chips to query by Ethernet adapter
        self._chips_to_ask = dict()
        for eth_x, eth_y, ip_address in job_connections:

            # Get the set of chips to read
            chips = [((x + eth_x) % width, (y + eth_y) % height)
                     for x in range(0, 8) for y in range(0, 8)
                     if (x, y) not in Machine.BOARD_48_CHIP_GAPS]
            chips = filter(
                lambda coords: p2p_table.is_route(coords[0], coords[1]),
                chips)
            shuffle(chips)
            self._chips_to_ask[eth_x, eth_y, ip_address] = OrderedSet(chips)

    def close(self):
        self._done = True

    def running(self):
        return not self._done

    def _receive_chip_info(self, scp_read_chip_info_response,
                           connection, eth_x, eth_y, ip_address):
        chip_info = scp_read_chip_info_response.chip_info
        chips = self._chips_to_ask.get((eth_x, eth_y, ip_address), None)
        if chips is not None:
            if (chip_info.x, chip_info.y) in chips:
                chips.remove((chip_info.x, chip_info.y))
                self._core_counter.add_cores(
                    eth_x, eth_y, chip_info.x, chip_info.y, chip_info.n_cores)
            if len(chips) == 0:
                self._chips_to_ask.pop((eth_x, eth_y, ip_address))
            else:
                x, y = next(iter(chips))
                self._send_scp(
                    connection, eth_x, eth_y, ip_address,
                    GetChipInfo(x, y), self._receive_chip_info, connection,
                    eth_x, eth_y, ip_address)
                self._n_sends += 1

    def _on_error(
            self, message, response, result, etype, value, tb, connection,
            eth_x, eth_y, ip_address):
        x, y = (message.sdp_header.destination_chip_x,
                message.sdp_header.destination_chip_y)
        if result == SCPResult.RC_OK:
            traceback.print_exception(etype, value, tb)
        else:
            warn(result, "when speaking to", x, y, "from",
                 eth_x, eth_y, ip_address)
        chips = self._chips_to_ask.get((eth_x, eth_y, ip_address), None)
        if chips is not None:
            if (x, y) in chips:
                chips.remove((x, y))
                chips.add((x, y))

    def run(self):
        connection = SCAMPConnection()
        while not self._done:
            timeout = False
            try:
                items_to_send = [(eth_x, eth_y, ip_address, chips)
                                 for (eth_x, eth_y, ip_address), chips in
                                 iteritems(self._chips_to_ask)]
                shuffle(items_to_send)
                self._n_sends = 0
                for eth_x, eth_y, ip_address, chips in items_to_send:
                    x, y = next(iter(chips))
                    self._send_scp(
                        connection, eth_x, eth_y, ip_address,
                        GetChipInfo(x, y), self._receive_chip_info, connection,
                        eth_x, eth_y, ip_address)
                    self._n_sends += 1
                while self._n_sends > 0:
                    self._process_scp(connection)
                    self._n_sends -= 1
            except SpinnmanTimeoutException:
                timeout = True
            except Exception:
                traceback.print_exc()
            if len(self._chips_to_ask) == 0:
                self._done = True
            elif not timeout:
                time.sleep(0.5)

        for (eth_x, eth_y, ip_address), chips in iteritems(
                self._chips_to_ask):
            warn("No reply from", ip_address, eth_x, eth_y, sorted(chips))


class CoreCounter(object):

    def __init__(self, width, height, job):
        self._total_cores = 0
        self._update_lock = RLock()
        self._ready = False
        self._notify_lock = Condition()

        self._width = width
        self._height = height
        self._job = job
        self._image_data = numpy.zeros((height, width, 3), dtype="uint8")
        self._ax = None
        self._text = None
        self._fig = None
        self._max_value = [175, 150, 125, 100]

    def run(self):
        matplotlib.rcParams['toolbar'] = 'None'
        self._fig, self._ax = pyplot.subplots()
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

    def _key_press(self, _event):
        with self._notify_lock:
            self._ready = True
            self._notify_lock.notify_all()

    def _close(self, _event):
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

    def __init__(self, core_counter, job):
        self._thread = Thread(target=self._main, args=[core_counter, job])
        self._finished = False
        self._finished_lock = RLock()
        self._net_init = None
        self._read_board_process = None
        self._done = False

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def close(self):
        with self._finished_lock:
            self._finished = True
            if self._net_init is not None:
                self._net_init.close()
            if self._read_board_process is not None:
                self._read_board_process.close()
            self._done = True

    def _main(self, core_counter, job):
        with job:
            job_connections = list()
            for (x, y), ip_address in iteritems(job.connections):
                job_connections.append((x, y, ip_address))
            job_connections.sort(key=operator.itemgetter(0, 1))

            # Get a list of connections to the machine
            root_connection = None
            boot_connection = None
            for x, y, ip_address in job_connections:
                core_counter.add_ethernet(x, y)
                if (x, y) == (0, 0):
                    root_connection = SCAMPConnection(
                        x, y, remote_host=ip_address)
                    boot_connection = BootConnection(remote_host=ip_address)

            core_counter.wait_until_ready()

            # Connect to the machine and boot it
            extra_boot_values = {
                SystemVariableDefinition.netinit_bc_wait_time: 255
            }
            version_read_ok = False
            tries = 3
            while not version_read_ok and tries > 0 and not self._done:
                warn("Booting machine", boot_connection.remote_ip_address)
                boot_messages = SpinnakerBootMessages(
                    board_version=5, extra_boot_values=extra_boot_values)
                for boot_message in boot_messages.messages:
                    boot_connection.send_boot_message(boot_message)
                time.sleep(2.0)
                try:
                    get_version = GetVersionProcess(
                        RoundRobinConnectionSelector([root_connection]))
                    get_version.get_version(0, 0, 0)
                    version_read_ok = True
                except Exception:
                    warn("Boot failed, retrying")
                    tries -= 1
            if not version_read_ok and not self._done:
                raise Exception("Could not boot machine")

            with self._finished_lock:
                self._net_init = ReadNetinitPhaseProcess(
                    job_connections, core_counter)
            if not self._done:
                warn("Waiting for boot to complete")
                self._net_init.run()

            if not self._done:
                warn("Waiting for 0, 0 to complete")
            boot_done = False
            while not self._done and not boot_done:
                try:
                    get_version = GetVersionProcess(
                        RoundRobinConnectionSelector([root_connection]))
                    version = get_version.get_version(0, 0, 0)
                    if version.x == 0 and version.y == 0:
                        boot_done = True
                except Exception:
                    pass

            # Read the P2P table to know which chips should exist
            if not self._done:
                connection = SCAMPConnection()
                warn("Reading P2P Table")
                p2p_process = GetP2PTableProcess(
                    job.width, job.height, 0, 0, job.hostname, connection)
                p2p_table = p2p_process.get_p2p_table()
                warn(len(p2p_table._routes), "chips")

            # Finally count the cores
            with self._finished_lock:
                if not self._done:
                    self._read_board_process = ReadBoardProcess(
                        job.width, job.height, p2p_table, job_connections,
                        core_counter)
            if not self._done:
                self._read_board_process.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    job = Job(1200, hostname="spinn-test.cs.man.ac.uk",
              owner="SpiNNaker Start")

    # Create GUI
    job.wait_for_state_change(JobState.queued)
    core_counter = CoreCounter(job.width, job.height, job)

    # Run task in thread
    main_thread = MainThread(core_counter, job)
    main_thread.start()

    # Run GUI
    core_counter.run()
    main_thread.close()

    # Wait for completion
    main_thread.join()

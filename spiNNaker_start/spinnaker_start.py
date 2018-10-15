from __future__ import print_function
import logging

from spinn_utilities.log import FormatAdapter

from spinnman.constants import ROUTER_REGISTER_P2P_ADDRESS
from spinnman.processes import AbstractMultiConnectionProcess
from spinnman.messages.scp.impl import ReadMemory, GetChipInfo
from spinnman.model import P2PTable
from spinnman.processes.get_version_process import GetVersionProcess
from spinnman.messages.sdp.sdp_flag import SDPFlag
import six
import sys
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

from six import iteritems
from threading import Thread, RLock, Condition
from matplotlib import pyplot, animation
from numpy.random.mtrand import shuffle
import functools
import numpy
import matplotlib
import time
import operator

logger = FormatAdapter(logging.getLogger(__name__))


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


class ReadVersionProcess(AbstractMultiConnectionProcess):

    def __init__(self, x, y, connection_selector, core_counter):
        super(ReadVersionProcess, self).__init__(
            connection_selector, timeout=0.5, n_retries=0)
        self._x = x
        self._y = y
        self._core_counter = core_counter
        self._thread = Thread(target=self._get_version)
        self._done = False

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def _get_version_response(self, version_response):
        version = version_response.version_info
        self._core_counter.set_ethernet_responding(self._x, self._y)
        if (version.x != AbstractSCPRequest.DEFAULT_DEST_X_COORD and
                version.y != AbstractSCPRequest.DEFAULT_DEST_Y_COORD):
            self._core_counter.set_ethernet_booted(version.x, version.y)
            self._done = True

    def _get_version(self):
        while not self._done:
            self._send_request(
                GetVersion(self._x, self._y),
                callback=self._get_version_response)
            self._finish()
            if not self._done:
                time.sleep(0.1)


class GetP2PTableProcess(AbstractMultiConnectionProcess):

    def __init__(self, connection_selector, width, height):
        super(GetP2PTableProcess, self).__init__(connection_selector)
        self._width = width
        self._height = height
        self._p2p_column_data = [None] * width

    def _receive_p2p_data(self, column, scp_read_response):
        self._p2p_column_data[column] = (
            scp_read_response.data, scp_read_response.offset)

    def get_p2p_table(self):

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
        return P2PTable(self._width, self._height, self._p2p_column_data)


class ReadBoardProcess(AbstractMultiConnectionProcess):
    """ A process for getting the machine details over a set of connections.
    """

    def __init__(
            self, eth_x, eth_y, width, height, p2p_table, connection_selector,
            core_counter):
        # pylint: disable=too-many-arguments
        super(ReadBoardProcess, self).__init__(connection_selector)
        self._eth_x = eth_x
        self._eth_y = eth_y
        self._width = width
        self._height = height
        self._p2p_table = p2p_table
        self._core_counter = core_counter
        self._thread = Thread(target=self._read_board)

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def _receive_chip_info(self, scp_read_chip_info_response):
        chip_info = scp_read_chip_info_response.chip_info
        self._core_counter.add_cores(
            self._eth_x, self._eth_y, chip_info.x, chip_info.y,
            chip_info.n_cores)

    def _read_board(self):

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
        except Exception:
            print("Warning:", x, y, "missing when asked from",
                  self._eth_x, self._eth_y)


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
        pyplot.show()

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

    def add_cores(self, eth_x, eth_y, x, y, n_cores):
        colour = self._get_max_value(eth_x, eth_y)
        with self._update_lock:
            self._total_cores += n_cores
            value = int(round((float(colour) / 18.0) * n_cores))
            self._image_data[y, x] = [0, value, 0]


def main(core_counter, job):
    job_connections = list()
    for (x, y), ip_address in iteritems(job.connections):
        job_connections.append((x, y, ip_address))
    job_connections.sort(key=operator.itemgetter(0, 1))
    core_counter.wait_until_ready()

    # Get a list of connections to the machine and start a thread for each
    # listening for the boot to be done
    connections = list()
    boot_threads = list()
    root_connection = None
    boot_connection = None
    for x, y, ip_address in job_connections:
        connection = SCAMPConnection(x, y, remote_host=ip_address)
        connections.append(connection)
        core_counter.add_ethernet(x, y)
        if (x, y) == (0, 0):
            root_connection = connection
            boot_connection = BootConnection(remote_host=ip_address)
        else:
            read_version = ReadVersionProcess(
                x, y, RoundRobinConnectionSelector([connection]), core_counter)
            read_version.start()
            boot_threads.append(read_version)

    # Connect to the machine and boot it
    boot_done = False
    tries = 3
    while not boot_done and tries > 0:
        print("Booting machine", boot_connection.remote_ip_address)
        boot_messages = SpinnakerBootMessages(board_version=5)
        for boot_message in boot_messages.messages:
            boot_connection.send_boot_message(boot_message)
        time.sleep(2.0)
        try:
            print("Waiting for boot to complete")
            version_read_ok = False
            while not boot_done:
                try:
                    get_version = GetVersionProcess(
                        RoundRobinConnectionSelector([root_connection]))
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
            print("Boot failed, retrying")
            tries -= 1
    if not boot_done:
        raise Exception("Could not boot machine")

    for thread in boot_threads:
        thread.join()
    print("Machine booted!")

    # Read the P2P table to know which chips should exist
    print("Reading P2P Table")
    p2p_process = GetP2PTableProcess(
        RoundRobinConnectionSelector([root_connection]), job.width, job.height)
    p2p_table = p2p_process.get_p2p_table()

    # Create a reader thread for each connection, and read the cores
    processes = list()
    for connection in connections:
        process = ReadBoardProcess(
            connection.chip_x, connection.chip_y,
            job.width, job.height, p2p_table,
            RoundRobinConnectionSelector([connection]), core_counter)
        process.start()
        processes.append(process)

    # Wait for everything to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get the machine and wait for it to be ready
    print("Waiting for machine to power on")
    with Job(570, hostname="spinnaker.cs.man.ac.uk",
             owner="SpiNNaker Start") as job:

        # Create GUI
        core_counter = CoreCounter(job.width, job.height)

        # Run task in thread
        main_thread = Thread(target=main, args=[core_counter, job])
        main_thread.start()

        # Run GUI
        core_counter.run()

        # Wait for completion
        main_thread.join()

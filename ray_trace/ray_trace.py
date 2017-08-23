import spinnaker_graph_front_end as g
from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.machine.machine_vertex import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.sdram_resource import SDRAMResource
from spinnman.messages.sdp.sdp_message import SDPMessage
from spinnman.messages.sdp.sdp_header import SDPHeader
from spinnman.messages.sdp.sdp_flag import SDPFlag
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint
from spinn_front_end_common.utilities.utility_objs.executable_start_type \
    import ExecutableStartType
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource
from pacman.model.resources.iptag_resource import IPtagResource

from threading import Thread
import sys
import subprocess
import os
import struct

position = (-220.0, 50.0, 0.0)
look = (1.0, 0.0, 0.0)
up = (0.0, 1.0, 0.0)

horizontalFieldOfView = 60.0
verticalFieldOfView = 50.0
frameHeight = 600
antialiasing = 10
cores = 59


class Aggregator(MachineVertex, AbstractHasAssociatedBinary):

    def __init__(self):
        MachineVertex.__init__(
            self, label="Aggregator",
            constraints=[ChipAndCoreConstraint(0, 0, 1)])

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "aggregator.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableStartType.RUNNING

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return ResourceContainer(
            dtcm=DTCMResource(0), sdram=SDRAMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0), iptags=[
                IPtagResource(".", 17894, strip_sdp=False, tag=1)])


class Tracer(
        MachineVertex, AbstractHasAssociatedBinary):

    seen_chips = set()

    def __init__(self):
        MachineVertex.__init__(
            self, label="Tracer")

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "tracer.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableStartType.RUNNING

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return ResourceContainer(
            dtcm=DTCMResource(0), sdram=SDRAMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))


def read_output(drawer, out):
    while drawer.poll() is None:
        line = out.readline()
        if line:
            print line
    print "Drawer exited - quitting"
    try:
        g.stop()
    except:
        pass
    os._exit(0)


drawer = None
if sys.platform.startswith("win32"):
    drawer = "drawer.exe"
elif sys.platform.startswith("darwin"):
    drawer = "drawer_osx"
elif sys.platform.startswith("linux"):
    drawer = "drawer_linux"
else:
    raise Exception("Unknown platform {}".format(sys.platform))
drawer = os.path.abspath(os.path.join(os.path.dirname(__file__), drawer))
print "Executing", drawer
drawer_exec = subprocess.Popen(
    args=[drawer, str(frameHeight)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
Thread(target=read_output, args=[drawer_exec, drawer_exec.stdout]).start()

g.setup()
for i in range(cores):
    tracer = Tracer()
    g.add_machine_vertex_instance(tracer)
aggregator = Aggregator()
g.add_machine_vertex_instance(aggregator)
g.run(None)

# Build the trigger message
(camx, camy, camz) = [int(pos) * 512 for pos in position]
(lookx, looky, lookz) = [int(l) * 65536 for l in look]
(upx, upy, upz) = [int(u) * 65536 for u in up]

frameWidth = int(frameHeight * horizontalFieldOfView / verticalFieldOfView)

# Pack them up...
data = struct.pack(
    "<HHIIIiiiiiiiiiiiiiiiii",
    4, 0, 0, 0, 0,
    camx, camy, camz,
    lookx, looky, lookz,
    upx, upy, upz,
    frameWidth, frameHeight,
    int(horizontalFieldOfView * 65536), int(verticalFieldOfView * 65536),
    antialiasing, 0, 0, 16)


# Send a trigger message
transceiver = g.transceiver()
placements = g.placements()
placement = placements.get_placement_of_vertex(aggregator)
transceiver.send_sdp_message(SDPMessage(
    sdp_header=SDPHeader(
        flags=SDPFlag.REPLY_NOT_EXPECTED, destination_port=1,
        destination_cpu=placement.p, destination_chip_x=placement.x,
        destination_chip_y=placement.y),
    data=data, offset=0))

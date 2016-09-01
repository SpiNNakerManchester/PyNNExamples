import spinnaker_graph_front_end as g
from pacman.model.graphs.machine.impl.machine_vertex import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.sdram_resource import SDRAMResource
from spinnman.messages.sdp.sdp_message import SDPMessage
from spinnman.messages.sdp.sdp_header import SDPHeader
from spinnman.messages.sdp.sdp_flag import SDPFlag
from pacman.model.constraints.placer_constraints\
    .placer_chip_and_core_constraint import PlacerChipAndCoreConstraint
from spinn_front_end_common.abstract_models.abstract_starts_synchronized import AbstractStartsSynchronized
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
cores = 64


class HeatDemo(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractStartsSynchronized):

    seen_chips = set()

    def __init__(self):
        MachineVertex.__init__(
            self, ResourceContainer(
                dtcm=DTCMResource(0), sdram=SDRAMResource(0),
                cpu_cycles=CPUCyclesPerTickResource(0),
                iptags=[IPtagResource("localhost", 17894, False, tag=1)]),
            label="Tracer")

    def get_binary_file_name(self):
        return "heat_demo.aplx"


def read_output(visualiser, out):
    while visualiser.poll() is None:
        line = out.readline()
        if line:
            print line
    print "Visualiser exited - quitting"
    try:
        g.stop()
    except:
        pass
    os._exit(0)


g.setup()
hostname = g._spinnaker._hostname

visualiser = None
if sys.platform.startswith("win32"):
    visualiser = "visualiser.exe"
elif sys.platform.startswith("darwin"):
    visualiser = "visualiser_osx"
elif sys.platform.startswith("linux"):
    visualiser = "visualiser_linux"
else:
    raise Exception("Unknown platform {}".format(sys.platform))
visualiser = os.path.abspath(os.path.join(os.path.dirname(__file__), visualiser))
print "Executing", visualiser
vis_exec = subprocess.Popen(
    args=[visualiser, "-c", "heatmap2x2.ini", "-ip", hostname],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
Thread(target=read_output, args=[vis_exec, vis_exec.stdout]).start()

for i in range(cores):
    heat_demo = HeatDemo()
    g.add_machine_vertex_instance(heat_demo)
g.run(None)

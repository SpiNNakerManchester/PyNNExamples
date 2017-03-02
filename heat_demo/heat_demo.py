import spinnaker_graph_front_end as g

from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.machine.impl.machine_vertex import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.sdram_resource import SDRAMResource
from pacman.model.resources.cpu_cycles_per_tick_resource \
    import CPUCyclesPerTickResource
from pacman.model.resources.iptag_resource import IPtagResource

from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.utilities.utility_objs.executable_start_type \
    import ExecutableStartType


from threading import Thread
import sys
import subprocess
import os
import platform

position = (-220.0, 50.0, 0.0)
look = (1.0, 0.0, 0.0)
up = (0.0, 1.0, 0.0)

horizontalFieldOfView = 60.0
verticalFieldOfView = 50.0
frameHeight = 600
antialiasing = 10


class HeatDemo(
        MachineVertex, AbstractHasAssociatedBinary):

    seen_chips = set()

    def __init__(self):
        MachineVertex.__init__(self, label="Tracer")

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "heat_demo.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableStartType.SYNC

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return ResourceContainer(
            dtcm=DTCMResource(0), sdram=SDRAMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0),
            iptags=[IPtagResource("localhost", 17894, False, tag=1)])


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
machine = g.machine()
cores = 0

for processor in machine.get_chip_at(0, 0).processors:
    if not processor.is_monitor:
        cores += 1
for processor in machine.get_chip_at(0, 1).processors:
    if not processor.is_monitor:
        cores += 1
for processor in machine.get_chip_at(1, 0).processors:
    if not processor.is_monitor:
        cores += 1
for processor in machine.get_chip_at(1, 1).processors:
    if not processor.is_monitor:
        cores += 1

if hostname is None:
    raise Exception(
        "You must use a local machine for this to work correctly.")

visualiser = None
if sys.platform.startswith("win32"):
    visualiser = "visualiser.exe"
elif sys.platform.startswith("darwin"):
    visualiser = "visualiser_osx"
elif sys.platform.startswith("linux"):
    if platform.machine() == "x86_64":
        visualiser = "visualiser_linux"
    elif platform.machine() == "i386":
        visualiser = "visualiser_linux"
    elif platform.machine() is None:
        print "Cant diagnose the bit size of the machine. " \
              "Running 32 bit visualiser."
        visualiser = "visualiser_linux"
    else:
        print "I do not recognise the bit size of the machine. " \
              "Will use 32 bit visualiser."
        visualiser = "visualiser_linux"
else:
    raise Exception("Unknown platform {}".format(sys.platform))

visualiser = os.path.abspath(os.path.join(
    os.path.dirname(__file__), visualiser))

print "Executing", visualiser
vis_exec = subprocess.Popen(
    args=[visualiser, "-c", "heatmap2x2.ini", "-ip", hostname],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
Thread(target=read_output, args=[vis_exec, vis_exec.stdout]).start()

for i in range(cores):
    heat_demo = HeatDemo()
    g.add_machine_vertex_instance(heat_demo)
g.run(None)

#############################################################
#
# A PyNN Sudoku model
# Steve Furber, November 2015
#
#############################################################
import spynnaker8 as p
from pyNN.random import RandomDistribution
import spynnaker8.external_devices as ext
import subprocess
import os
import sys
import traceback
from utils import puzzles, get_rates

run_time = 20000                        # run time in milliseconds
neurons_per_digit = 5                   # number of neurons per digit
fact = float(neurons_per_digit) / 10.0  # number of neurons per digit / 10
ms_per_bin = 100
running = False
ended = False


# Run the visualiser
def read_output(visualiser, out):
    result = visualiser.poll()
    while result is None:
        line = out.readline()
        if line:
            print(line)
        result = visualiser.poll()
    print("Visualiser exited: {} - quitting".format(result))
    global running
    global ended
    if running and not ended:
        p.end()
    os._exit(0)


def activate_visualiser(old_vis):
    vis_exe = None
    if old_vis:
        if sys.platform.startswith("win32"):
            vis_exe = "sudoku.exe"
        elif sys.platform.startswith("darwin"):
            vis_exe = "sudoku_osx"
        elif sys.platform.startswith("linux"):
            vis_exe = "sudoku_linux"
        else:
            raise Exception("Unknown platform: {}".format(sys.platform))
        vis_exe = [os.path.abspath(os.path.join(
            os.path.dirname(__file__), vis_exe))]
        neur_per_num_opt = "-neurons_per_number"
        ms_per_bin_opt = "-ms_per_bin"
    else:
        vis_exe = ["spynnaker_sudoku"]
        neur_per_num_opt = "--neurons_per_number"
        ms_per_bin_opt = "--ms_per_bin"
    try:
        return subprocess.Popen(
            args=vis_exe + [neur_per_num_opt, str(neurons_per_digit),
                            ms_per_bin_opt, str(ms_per_bin)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        # Thread(target=read_output,
        #        args=[visualiser, visualiser.stdout]).start()
    except Exception:
        if not old_vis:
            print("This example depends on https://github.com/"
                  "SpiNNakerManchester/sPyNNakerVisualisers")
            traceback.print_exc()
            print("trying old visualiser")
            activate_visualiser(old_vis=True)
        else:
            raise


vis_process = activate_visualiser(old_vis=("OLD_VIS" in os.environ))

p.setup(timestep=1.0)
print("Creating Sudoku Network...")
n_cell = int(90 * fact)   # total number of neurons in a cell
n_stim = n_cell           # number of neurons in each stimulation source
n_N = n_cell // 9         # number of neurons per value in cell

# total number of neurons
n_total = n_cell * 9 * 9
n_stim_total = n_stim * 9 * 9

# global distributions & parameters
weight_cell = 0.2
weight_stim = 1.8
dur_nois = RandomDistribution("uniform", [30000.0, 30001.0])
weight_nois = 1.8
delay = 2.0
puzzle = 5


# initialise non-zeros
# NB use as init[8-y][x] -> cell[x][y]
init = puzzles[puzzle]

# Dream problem - no input!
# init = [[0 for x in range(9)] for y in range(9)]
corr = init

p.set_number_of_neurons_per_core(p.IF_curr_exp, 200)

#
# set up the 9x9 cell array populations
#
cell_params_lif = {
    'cm': 0.25,         # nF membrane capacitance
    'i_offset': 0.5,    # nA    bias current
    'tau_m': 20.0,      # ms    membrane time constant
    'tau_refrac': 2.0,  # ms    refractory period
    'tau_syn_E': 5.0,   # ms    excitatory synapse time constant
    'tau_syn_I': 5.0,   # ms    inhibitory synapse time constant
    'v_reset': -70.0,   # mV    reset membrane potential
    'v_rest': -65.0,    # mV    rest membrane potential
    'v_thresh': -50.0,  # mV    firing threshold voltage
}

print("Creating Populations...")
cells = p.Population(n_total, p.IF_curr_exp, cell_params_lif, label="Cells",
                     additional_parameters={"spikes_per_second": 200})
# cells.record("spikes")
ext.activate_live_output_for(cells, tag=1, port=17897)

#
# add a noise source to each cell
#
print("Creating Noise Sources...")
default_rate = 35.0
max_rate = 100.0
rates = get_rates(init, n_total, n_cell, n_N, default_rate, max_rate)
noise = p.Population(
    n_total, p.SpikeSourcePoisson,
    {"rate": rates},
    label="Noise")

p.Projection(noise, cells, p.OneToOneConnector(),
             synapse_type=p.StaticSynapse(weight=weight_nois))

p.external_devices.add_poisson_live_rate_control(
    noise, database_notify_port_num=19990)


#
# set up the cell internal inhibitory connections
#
print("Setting up cell inhibition...")
connections = list()
for x in range(9):
    for y in range(9):
        base = ((y * 9) + x) * n_cell

        # full constant matrix of weight_cell apart from n_N squares on
        # diagonal
        connections_cell = [
            (i + base, j + base,
             0.0 if i // n_N == j // n_N else weight_cell, delay)
            for i in range(n_cell) for j in range(n_cell)
        ]
        connections.extend(connections_cell)


#
# set up the inter-cell inhibitory connections
#
def interCell(x, y, r, c, connections):
    """ Inhibit same number: connections are n_N squares on diagonal of
        weight_cell() from cell[x][y] to cell[r][c]
    """
    base_source = ((y * 9) + x) * n_cell
    base_dest = ((c * 9) + r) * n_cell
    connections_intC = [
        (i + base_source, j + base_dest, weight_cell, delay)
        for i in range(n_cell)
        for j in range(n_N * (i // n_N), n_N * (i // n_N + 1))]

    connections.extend(connections_intC)


print("Setting up inhibition between cells...")
for x in range(9):
    for y in range(9):
        for r in range(9):
            if r != x:
                interCell(x, y, r, y, connections)  # by row...
        for c in range(9):
            if c != y:
                interCell(x, y, x, c, connections)  # by column...
        for r in range(3 * (x // 3), 3 * (x // 3 + 1)):
            for c in range(3 * (y // 3), 3 * (y // 3 + 1)):
                if r != x and c != y:
                    interCell(x, y, r, c, connections)  # & by square
conn_intC = p.FromListConnector(connections)
p.Projection(cells, cells, conn_intC, receptor_type="inhibitory")

# initialise the network, run, and get results
cells.initialize(v=RandomDistribution("uniform", [-65.0, -55.0]))

set_window = subprocess.Popen((sys.executable, "-m", "set_numbers",
                               str(n_total), str(n_cell), str(n_N),
                               str(default_rate), str(max_rate), str(puzzle)))

p.external_devices.run_forever()

# spikes = cells.getSpikes()
# f, axarr = pylab.subplots(9, 9)
# for y in range(9):
#     for x in range(9):
#         base = ((y * 9) + x) * n_cell
#         next_base = base + n_cell
#         ids = spikes[:, 0]
#         cell_spikes = spikes[numpy.where((ids >= base) & (ids < next_base))]
#         axarr[8 - y][x].plot(
#             [i[1] for i in cell_spikes],
#             [i[0] - base for i in cell_spikes], ".")
#         axarr[8 - y][x].axis([0, run_time, -1, n_cell + 1])
#         # axarr[8 - y][x].axis('off')
# pylab.show()
# pylab.savefig("sudoku.png")

vis_process.wait()
set_window.wait()

p.end()
ended = True

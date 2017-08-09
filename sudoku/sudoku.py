#############################################################
#
# A PyNN Sudoku model
# Steve Furber, November 2015
#
#############################################################
try:
    import pyNN.spiNNaker as p
except Exception as e:
    import spynnaker7.pyNN as p
from pyNN.random import RandomDistribution
import spynnaker7.pyNN.external_devices as ext
import subprocess
from threading import Thread
import os
import sys

run_time = 20000                        # run time in milliseconds
neurons_per_digit = 5                   # number of neurons per digit
fact = float(neurons_per_digit) / 10.0  # number of neurons per digit / 10
ms_per_bin = 100
running = False
ended = False


# Run the visualiser
def read_output(visualiser, out):
    while visualiser.poll() is None:
        line = out.readline()
        if line:
            print line
    print "Visualiser exited - quitting"
    global running
    global ended
    if running and not ended:
        p.end()
    os._exit(0)


vis_exe = None
if sys.platform.startswith("win32"):
    vis_exe = "sudoku.exe"
elif sys.platform.startswith("darwin"):
    vis_exe = "sudoku_osx"
elif sys.platform.startswith("linux"):
    vis_exe = "sudoku_linux"
else:
    raise Exception("Unknown platform: {}".format(sys.platform))
vis_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), vis_exe))
print "Executing", vis_exe
visualiser = subprocess.Popen(args=[
    vis_exe,
    "-neurons_per_number", str(neurons_per_digit),
    "-ms_per_bin", str(ms_per_bin)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
Thread(target=read_output, args=[visualiser, visualiser.stdout]).start()

p.setup(timestep=1.0)
print "Creating Sudoku Network..."
n_cell = int(90 * fact)   # total number of neurons in a cell
n_stim = 30               # number of neurons in each stimulation source
n_N = n_cell / 9          # number of neurons per value in cell

# total number of neurons
n_total = n_cell * 9 * 9
n_stim_total = n_stim * 9 * 9

# global distributions & parameters
weight_cell = 0.2
weight_stim = 0.2
dur_nois = RandomDistribution("uniform", [30000.0, 30001.0])
weight_nois = 1.4
delay = 2.0


# Diabolical problem:

init = [[0, 0, 1, 0, 0, 8, 0, 7, 3],  # initialise non-zeros
        [0, 0, 5, 6, 0, 0, 0, 0, 1],  # NB use as init[8-y][x] -> cell[x][y]
        [7, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 9, 0, 8, 1, 0, 0, 0, 0],
        [5, 3, 0, 0, 0, 0, 0, 4, 6],
        [0, 0, 0, 0, 6, 5, 0, 3, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 4],
        [8, 0, 0, 0, 0, 9, 3, 0, 0],
        [9, 4, 0, 5, 0, 0, 7, 0, 0]]

# init = [[2, 0, 0, 0, 0, 6, 0, 3, 0],  # initialise non-zeros
#         [4, 8, 0, 0, 1, 9, 0, 0, 0],  # NB use as init[8-y][x] -> cell[x][y]
#         [0, 0, 7, 0, 2, 0, 9, 0, 0],
#         [0, 0, 0, 3, 0, 0, 0, 9, 0],
#         [7, 0, 8, 0, 0, 0, 1, 0, 5],
#         [0, 4, 0, 0, 0, 7, 0, 0, 0],
#         [0, 0, 4, 0, 9, 0, 6, 0, 0],
#         [0, 0, 0, 6, 4, 0, 0, 1, 9],
#         [0, 5, 0, 1, 0, 0, 0, 0, 8]]

# Dream problem - no input!
# init = [[0 for x in range(9)] for y in range(9)]
corr = init

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
    'spikes_per_second': 200
}

print "Creating Populations..."
cells = p.Population(n_total, p.IF_curr_exp, cell_params_lif, label="Cells")
cells.record()
ext.activate_live_output_for(cells, tag=1, port=17897)

#
# add a noise source to each cell
#
print "Creating Noise Sources..."
noise = p.Population(
    n_total, p.SpikeSourcePoisson,
    {"rate": 20.0},
    label="Noise")
p.Projection(noise, cells, p.OneToOneConnector(weight_nois))


#
# set up the cell internal inhibitory connections
#
print "Setting up cell inhibition..."
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


print "Setting up inhibition between cells..."
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
p.Projection(cells, cells, conn_intC, target="inhibitory")

#
# set up & connect the initial (stimulation) conditions
#
print "Fixing initial numbers..."
s = 0
connections_stim = []
for x in range(9):
    for y in range(9):
        if init[8 - y][x] != 0:
            base_stim = ((y * 9) + x) * n_stim
            base = ((y * 9) + x) * n_cell
            for i in range(n_stim):

                # one n_N square on diagonal
                for j in range(
                        n_N * (init[8 - y][x] - 1), n_N * init[8 - y][x]):
                    connections_stim.append(
                        (i + base_stim, j + base, weight_stim, delay))

            s += 1

if len(connections_stim) > 0:
    stim = p.Population(
        n_stim_total, p.SpikeSourcePoisson,
        {"rate": 10.0}, label="Stim")
    conn_stim = p.FromListConnector(connections_stim)
    p.Projection(stim, cells, conn_stim, target="excitatory")
#
# initialise the network, run, and get results
cells.initialize("v", RandomDistribution("uniform", [-65.0, -55.0]))

running = True
p.run(run_time)

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

p.end()
ended = True

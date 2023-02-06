# Copyright (c) 2017-2021 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Injecting time-varying current into a cell.

There are four "standard" current sources in PyNN:

    - DCSource
    - ACSource
    - StepCurrentSource
    - NoisyCurrentSource

Any other current waveforms could be implemented using StepCurrentSource.

Script from http://neuralensemble.org/docs/PyNN/examples/current_injection.html

"""

import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from quantities import mV

sim.setup(timestep=1.0)

# === Create four cells and inject current into each one =====================
cells = sim.Population(4, sim.IF_curr_exp(
    v_thresh=-55.0, tau_refrac=5.0, tau_m=10.0))

current_sources = [sim.DCSource(amplitude=0.5, start=50.0, stop=400.0),
                   sim.StepCurrentSource(times=[50.0, 210.0, 250.0, 410.0],
                                         amplitudes=[0.4, 0.6, -0.2, 0.2]),
                   sim.ACSource(start=50.0, stop=450.0, amplitude=0.2,
                                offset=0.1, frequency=10.0, phase=180.0),
                   sim.NoisyCurrentSource(mean=0.5, stdev=0.2, start=50.0,
                                          stop=450.0, dt=1.0)]

for cell, current_source in zip(cells, current_sources):
    cell.inject(current_source)

cells.record('v')

# === Run the simulation =====================================================
sim.run(500.0)


# === Save the results, optionally plot a figure =============================
vm = cells.get_data().segments[0].filter(name="v")[0]
sim.end()

Figure(
    Panel(vm, y_offset=-10 * mV, xticks=True, yticks=True,
          xlabel="Time (ms)", ylabel="Membrane potential (mV)",
          ylim=(-96, -59)),
    title="Current injection example",
    annotations="Simulated with {}".format(sim.name())
)

plt.show()

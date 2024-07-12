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

"""
Injecting time-varying current into a cell.

There are four "standard" current sources in PyNN:

    - DCSource
    - ACSource
    - StepCurrentSource
    - NoisyCurrentSource

Any other current waveforms could be implemented using StepCurrentSource.

Script from
https://neuralensemble.org/docs/PyNN/examples/current_injection.html

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
    annotations=f"Simulated with {sim.name()}"
)

plt.show()

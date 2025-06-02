# Copyright (c) 2019 The University of Manchester
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
import pyNN.spiNNaker as pynn
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel


runtime = 10000
pynn.setup(1.0)

neuron_params = {
    "v": 0,
    "i_offset": 1,
    "v_rest": 0
    }

neuron = pynn.Population(1,
                         pynn.extra_models.EPropAdaptive(**neuron_params),
                         label='eprop_pop')

neuron.record('all')

pynn.run(runtime)

res = neuron.get_data('all')

Figure(
    Panel(res.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    Panel(res.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=neuron.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)

plt.show()

pynn.end()
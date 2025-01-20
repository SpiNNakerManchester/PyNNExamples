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

import pyNN.spiNNaker as pynn

import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel

simulator_Name = 'spiNNaker'
# exec('import pyNN.%s as pynn' % simulator_Name)


def poisson_generator(_rate, _rng, _t_start=0.0, _t_stop=1000.0, _debug=False):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a numpy array of sorted spikes is returned,
                  rather than a SpikeTrain object.

    Examples:
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)

    See also:
        inh_poisson_generator, inh_gamma_generator,
        inh_adaptingmarkov_generator
    """

    n = (_t_stop - _t_start) / 1000.0 * _rate
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    if number > 0:
        isi = _rng.exponential(1.0 / _rate, number) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += _t_start
    i = np.searchsorted(spikes, _t_stop)

    extra_spikes = []
    if i == len(spikes):
        # Interspike interval buffer overrun

        t_last = spikes[-1] + _rng.exponential(1.0 / _rate, 1)[0] * 1000.0

        while (t_last < _t_stop):
            extra_spikes.append(t_last)
            t_last += _rng.exponential(1.0 / _rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))

        if _debug:
            print(f"ISI buf overrun handled. {len(spikes)=}, "
                  f"{len(extra_spikes)=}")

    else:
        spikes = np.resize(spikes, (i,))

    if _debug:
        return spikes, extra_spikes
    else:
        return [round(x) for x in spikes]


# Total number of neurons
Neurons = 1000
sim_time = 1000.0
g = 5.0
eta = 2.0
delay = 2.0
epsilon = 0.1

tau_m = 20.0  # ms (20ms will give a FR of 20hz)
tau_ref = 2.0
v_reset = 10.0
V_th = 20.0
v_rest = 0.0
tauSyn = 1.0

N_E = int(round(Neurons * 0.8))
N_I = int(round(Neurons * 0.2))

C_E = N_E * 0.1
C_I = N_I * 0.1

# Excitatory and inhibitory weights
J_E = 0.1
J_I = -g * J_E

# The firing rate of a neuron in the external pop
# is the product of eta time the threshold rate
# the steady state firing rate which is
# needed to bring a neuron to threshold.
nu_ex = eta * V_th / (J_E * C_E * tau_m)

# population rate of the whole external population.
# With CE neurons the pop rate is simply the product
# nu_ex*C_E  the factor 1000.0 changes the units from
# spikes per ms to spikes per second.
p_rate = 1000.0 * nu_ex * C_E
print(f"Rate is: {p_rate / 1000} HZ")

# Neural Parameters
pynn.setup(timestep=0.1, min_delay=0.1)

if simulator_Name == "spiNNaker":

    # Makes it easy to scale up the number of cores
    pynn.set_number_of_neurons_per_core(pynn.IF_curr_exp, 64)
    pynn.set_number_of_neurons_per_core(pynn.SpikeSourcePoisson, 64)

exc_cell_params = {
    'cm': 1.0,
    'tau_m': tau_m,
    'tau_refrac': tau_ref,
    'v_rest': v_rest,
    'v_reset': v_reset,
    'v_thresh': V_th,
    'tau_syn_E': tauSyn,
    'tau_syn_I': tauSyn,
    'i_offset': 0.9
}

inh_cell_params = {
    'cm': 1.0,
    'tau_m': tau_m,
    'tau_refrac': tau_ref,
    'v_rest': v_rest,
    'v_reset': v_reset,
    'v_thresh': V_th,
    'tau_syn_E': tauSyn,
    'tau_syn_I': tauSyn,
    'i_offset': 0.9
}

# Set-up pynn Populations
E_pop = pynn.Population(
    N_E, pynn.IF_curr_exp(**exc_cell_params), label="E_pop", seed=1)

I_pop = pynn.Population(
    N_I, pynn.IF_curr_exp(**inh_cell_params), label="I_pop", seed=2)

Poiss_ext_E = pynn.Population(
    N_E, pynn.SpikeSourcePoisson(rate=10.0), label="Poisson_pop_E",
    additional_parameters={"seed": 3})
Poiss_ext_I = pynn.Population(
    N_I, pynn.SpikeSourcePoisson(rate=10.0), label="Poisson_pop_I",
    additional_parameters={"seed": 4})

# Connectors
E_conn = pynn.FixedProbabilityConnector(epsilon)
I_conn = pynn.FixedProbabilityConnector(epsilon)

# Use random delays for the external noise and
# set the initial membrane voltage below the resting potential
# to avoid the overshoot of activity in the beginning of the simulation
delay_distr = RandomDistribution('uniform', low=0.1, high=12.8)
Ext_conn = pynn.OneToOneConnector()

uniformDistr = RandomDistribution('uniform', low=-10, high=0)
E_pop.initialize(v=uniformDistr)
I_pop.initialize(v=uniformDistr)

# Projections
E_E = pynn.Projection(
    E_pop, E_pop, E_conn, receptor_type="excitatory",
    synapse_type=pynn.StaticSynapse(weight=J_E, delay=delay))
I_E = pynn.Projection(
    I_pop, E_pop, I_conn, receptor_type="inhibitory",
    synapse_type=pynn.StaticSynapse(weight=J_I, delay=delay))
E_I = pynn.Projection(
    E_pop, I_pop, E_conn, receptor_type="excitatory",
    synapse_type=pynn.StaticSynapse(weight=J_E, delay=delay))
I_I = pynn.Projection(
    I_pop, I_pop, I_conn, receptor_type="inhibitory",
    synapse_type=pynn.StaticSynapse(weight=J_I, delay=delay))

Ext_E = pynn.Projection(
    Poiss_ext_E, E_pop, Ext_conn, receptor_type="excitatory",
    synapse_type=pynn.StaticSynapse(weight=J_E * 10, delay=delay_distr))
Ext_I = pynn.Projection(
    Poiss_ext_I, I_pop, Ext_conn, receptor_type="excitatory",
    synapse_type=pynn.StaticSynapse(weight=J_E * 10, delay=delay_distr))

# Record stuff
E_pop.record("spikes")

pynn.run(sim_time)

esp = None
isp = None
pe = None
pi = None
v_esp = None

esp = E_pop.get_data("spikes")


Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(esp.segments[0].spiketrains,
          yticks=True, markersize=1, xlim=(0, sim_time)),
    title="Brunnel example",
    annotations=f"Simulated with {pynn.name()}"
)
plt.show()

pynn.end()

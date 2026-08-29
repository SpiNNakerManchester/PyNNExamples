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
LIF neuron F-I curve demonstration.

Five IF_curr_exp (LIF) neurons are each driven by a different constant DC
current. The example records membrane voltage and spikes then produces:

  1. Voltage traces — showing integrate-to-threshold, spike, and reset for
     each current level.
  2. F-I curve — measured firing rate vs. injected current overlaid with
     the analytical LIF prediction.

LIF parameters used (defaults for IF_curr_exp):
  v_rest   = -65 mV   v_thresh = -50 mV   v_reset  = -65 mV
  tau_m    =  20 ms   tau_refrac =  2 ms  cm       =  1 nF

Threshold current: I_thresh = (v_thresh - v_rest) * cm / tau_m = 0.75 nA
  - neurons driven below this level stay at rest (subthreshold)
  - neurons driven above fire at a rate that increases with current
"""

import numpy as np
import matplotlib.pyplot as plt
import pyNN.spiNNaker as sim

# ── Simulation parameters ────────────────────────────────────────────────────
SIMTIME = 1000.0   # ms  — total simulation duration
TIMESTEP = 1.0     # ms
STIM_START = 100.0  # ms  — DC current onset
STIM_STOP = 900.0   # ms  — DC current offset

# DC current levels to sweep (nA).  Two below threshold, three above.
CURRENTS = [0.3, 0.6, 0.9, 1.2, 1.5]
N = len(CURRENTS)

# LIF neuron parameters (explicit for clarity)
LIF_PARAMS = dict(
    v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
    tau_m=20.0, tau_refrac=2.0, cm=1.0,
    tau_syn_E=5.0, tau_syn_I=5.0,
)

# ── Analytical F-I curve ─────────────────────────────────────────────────────


def lif_fi_theory(currents, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
                  tau_m=20.0, tau_refrac=2.0, cm=1.0):
    """Return theoretical LIF firing rates (Hz) for each current (nA)."""
    R_m = tau_m / cm          # MΩ  (tau_m in ms, cm in nF → mV/nA)
    rates = []
    for I in currents:
        V_inf = v_rest + R_m * I   # steady-state voltage (mV)
        if V_inf <= v_thresh:
            rates.append(0.0)
        else:
            # time from v_reset to v_thresh
            t_spike = tau_m * np.log(
                (V_inf - v_reset) / (V_inf - v_thresh))  # ms
            rates.append(1000.0 / (tau_refrac + t_spike))   # Hz
    return np.array(rates)


# ── Build and run network ────────────────────────────────────────────────────
sim.setup(timestep=TIMESTEP, min_delay=TIMESTEP)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

neurons = sim.Population(
    N, sim.IF_curr_exp(**LIF_PARAMS), label="LIF neurons")

for cell, amp in zip(neurons, CURRENTS):
    cell.inject(sim.DCSource(amplitude=amp, start=STIM_START, stop=STIM_STOP))

neurons.record(["v", "spikes"])

sim.run(SIMTIME)

neo = neurons.get_data(variables=["v", "spikes"])
v_data = neo.segments[0].filter(name="v")[0]        # shape: (time, N)
spike_trains = neo.segments[0].spiketrains           # list of N SpikeTrains

sim.end()

# ── Compute measured firing rates ────────────────────────────────────────────
stim_duration_s = (STIM_STOP - STIM_START) / 1000.0   # seconds
measured_rates = []
for train in spike_trains:
    in_window = [s for s in train
                 if STIM_START < float(s) < STIM_STOP]
    measured_rates.append(len(in_window) / stim_duration_s)

theory_currents = np.linspace(0, max(CURRENTS) + 0.2, 200)
theory_rates = lif_fi_theory(theory_currents, **LIF_PARAMS)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("LIF Neuron — F-I Curve Demonstration", fontsize=13)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, N))

# Panel 1: voltage traces
for i in range(N):
    ax1.plot(v_data.times, v_data[:, i],
             color=colors[i], label=f"I = {CURRENTS[i]:.1f} nA")
ax1.axhline(LIF_PARAMS["v_thresh"], color="red",
            linestyle="--", linewidth=1.0, label=f"Threshold ({LIF_PARAMS['v_thresh']} mV)")
ax1.axhline(LIF_PARAMS["v_rest"], color="gray",
            linestyle=":", linewidth=0.8, label=f"V_rest ({LIF_PARAMS['v_rest']} mV)")
ax1.axvspan(STIM_START, STIM_STOP, alpha=0.08, color="yellow", label="Stimulus window")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane potential (mV)")
ax1.set_title("Membrane Voltage Traces")
ax1.legend(loc="upper right", fontsize=8, ncol=2)
ax1.set_xlim(0, SIMTIME)

# Panel 2: F-I curve
ax2.plot(theory_currents, theory_rates,
         color="tomato", linewidth=1.5, linestyle="--", label="Analytical LIF")
ax2.plot(CURRENTS, measured_rates,
         "o", color="steelblue", markersize=9, zorder=5, label="Simulated")
ax2.axvline(0.75, color="gray", linestyle=":", linewidth=1.0,
            label=r"$I_{thresh}$ = 0.75 nA")
ax2.set_xlabel("Injected current (nA)")
ax2.set_ylabel("Firing rate (Hz)")
ax2.set_title("F-I Curve (Firing Rate vs. Injected Current)")
ax2.legend(fontsize=9)
ax2.set_xlim(0, max(CURRENTS) + 0.2)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lif_fi_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Console summary ──────────────────────────────────────────────────────────
print(f"\n{'Current (nA)':<14} {'Measured (Hz)':<16} {'Theory (Hz)':<14} Status")
print("-" * 58)
theory_at_points = lif_fi_theory(CURRENTS, **LIF_PARAMS)
for I, meas, theo in zip(CURRENTS, measured_rates, theory_at_points):
    status = "subthreshold" if meas == 0 else "firing"
    print(f"{I:<14.1f} {meas:<16.1f} {theo:<14.1f} {status}")

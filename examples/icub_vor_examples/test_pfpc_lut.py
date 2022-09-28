# Copyright (c) 2019-2022 The University of Manchester
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

import matplotlib.pyplot as plt
from spynnaker.pyNN.models.neuron.plasticity.stdp.common \
    import write_pfpc_lut

t_peak = 100

_comp_times, out_float, out_fixed, t = write_pfpc_lut(
    spec=None, peak_time=t_peak, lut_size=256, shift=0, time_probe=t_peak,
    kernel_scaling=0.8)

plt.plot(t, out_float, label='float')
plt.legend()
plt.title("pf-PC LUT")
plt.savefig("figures/write_pfpc_lut.png")

plt.plot(t, out_fixed, label='fixed int16')
plt.legend()
plt.title("pf-PC LUT")
plt.savefig("figures/write_pfpc_lut_final_exp_fix.png")

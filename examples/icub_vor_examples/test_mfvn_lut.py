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
    import write_mfvn_lut

beta = 10
sigma = 200

_comp_times, out_float, plot_times = write_mfvn_lut(
    spec=None, sigma=sigma, beta=beta, lut_size=256, shift=0, time_probe=22,
    kernel_scaling=0.8)

plt.plot(plot_times, out_float, label='float')
# plt.plot(t,out_fixed, label='fixed')
plt.legend()
plt.title("mf-VN LUT")
plt.savefig("figures/write_mfvn_lut.png")

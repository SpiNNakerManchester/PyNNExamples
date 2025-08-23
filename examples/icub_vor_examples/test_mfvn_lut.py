# Copyright (c) 2021 The University of Manchester
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

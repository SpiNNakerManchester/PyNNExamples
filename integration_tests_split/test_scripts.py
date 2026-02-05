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

import os
from spinnaker_testbase import ScriptChecker


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def setUp(self):
        os.chdir(os.path.dirname(__file__))

    def test_examples_extra_models_examples_IF_cond_exp_stoc(self):
        self.check_script("examples/extra_models_examples/IF_cond_exp_stoc.py",
                          use_script_dir=False)
        self.check_binary_used("IF_cond_exp_stoc_neuron.aplx")

    def test_examples_extra_models_examples_IF_curr_exp_ca2_adaptive(self):
        self.check_script("examples/extra_models_examples/IF_curr_exp_ca2_adaptive.py",
                          use_script_dir=False)
        self.check_binary_used("IF_curr_exp_ca2_adaptive_neuron.aplx")

    def test_examples_extra_models_examples_synfire_if_curr_dual_exp(self):
        self.check_script("examples/extra_models_examples/synfire_if_curr_dual_exp.py",
                          use_script_dir=False)
        self.check_binary_used("IF_curr_exp_dual_neuron.aplx")

    def test_examples_extra_models_examples_IF_curr_exp_sEMD(self):
        self.check_script("examples/extra_models_examples/IF_curr_exp_sEMD.py",
                          use_script_dir=False)
        self.check_binary_used("IF_curr_exp_sEMD_neuron.aplx")

    def test_examples_extra_models_examples_vogel_2011_vogels_2011_live(self):
        self.check_script("examples/extra_models_examples/vogel_2011/vogels_2011_live.py",
                          use_script_dir=False)
        # test does not produce spikes in either mode
        self.check_binary_used("synapses_stdp_mad_vogels_2011_additive.aplx")

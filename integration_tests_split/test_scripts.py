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
from spynnaker.pyNN.data import SpynnakerDataView


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

    def _print_binaries(self):
        targets = SpynnakerDataView.get_executable_targets()
        binaries = set()
        for target in targets.binaries:
            _, file = os.path.split(target)
            binaries.add(file)
        print(binaries)

    def test_examples_extra_models_examples_IF_cond_exp_stoc(self):
        self.check_script("examples/extra_models_examples/IF_cond_exp_stoc.py",
                          use_script_dir=False)
        self._print_binaries()

    def test_examples_extra_models_examples_IF_curr_exp_ca2_adaptive(self):
        self.check_script("examples/extra_models_examples/IF_curr_exp_ca2_adaptive.py",
                          use_script_dir=False)
        self._print_binaries()


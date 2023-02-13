# Copyright (c) 2019-2023 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from spinnaker_testbase import ScriptChecker
from unittest import SkipTest  # pylint: disable=unused-import


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """

    def test_balanced_random_balanced_random(self):
        self.check_script("balanced_random/balanced_random.py")

    def test_learning_random_dist(self):
        self.check_script("learning/random_dist.py")

    def test_learning_stdp(self):
        self.check_script("learning/stdp.py")

    def test_learning_simple(self):
        raise SkipTest("I hate simple")
        self.check_script("learning/simple.py")

    def test_synfire_synfire(self):
        self.check_script("synfire/synfire.py")

    def test_synfire_synfire_collab(self):
        self.check_script("synfire/synfire_collab.py")

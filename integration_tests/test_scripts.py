# Copyright (c) 2019-2021 The University of Manchester
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

from spinnaker_testbase import ScriptChecker


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to githuib for reference only.
    """
    def test_balanced_random_balanced_random(self):
        self.check_script("balanced_random/balanced_random.py")

    def test_learning_random_dist(self):
        self.check_script("learning/random_dist.py")

    def test_learning_stdp(self):
        self.check_script("learning/stdp.py")

    def test_learning_simple(self):
        self.check_script("learning/simple.py")

    def test_synfire_synfire(self):
        self.check_script("synfire/synfire.py")

    def test_synfire_synfire_collab(self):
        self.check_script("synfire/synfire_collab.py")

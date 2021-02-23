# Copyright (c) 2017-2019 The University of Manchester
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

from spinnaker_testbase import RootScriptBuilder


class ScriptBuilder(RootScriptBuilder):
    """
    This file will recreate the test_scripts.py file

    To skip the too_long scripts run this script with a parameter
    """

    def build_intro_labs_scripts(self):
        # These scripts raise a SkipTest with the reasons given
        exceptions = {}
        exceptions["pushbot_ethernet_example.py"] = "Needs a physical pushbot"

        # For branches these raise a SkipTest quoting the time given
        # For cron and manual runs these just and a warning
        too_long = {}
        too_long["stdp_triplet.py"] = "10 minutes"
        too_long["balanced_random_live_rate.py"] = "125 seconds"
        too_long["stdp_curve.py"] = "118 seconds"
        too_long["stdp_curve_cond.py"] = "121 seconds"
        too_long["vogels_2011.py"] = "698 seconds"

        self.create_test_scripts(["examples"], too_long, exceptions)


if __name__ == '__main__':
    builder = ScriptBuilder()
    builder.build_intro_labs_scripts()

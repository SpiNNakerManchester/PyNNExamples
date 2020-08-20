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
from examples.partitioner_examples.splitter_objects.common_splitter import (
    CommonSplitter)
from spynnaker.pyNN.utilities.constants import \
    MAX_TIMER_TICS_SUPPORTED_PER_BLOCK


class SplitterBySlice(CommonSplitter):

    def __init__(self):
        CommonSplitter.__init__(self)

    def create_machine_vertices(self, resource_tracker, machine_graph):
        return True

    def get_out_going_slices(self):
        pass

    def get_in_coming_slices(self):
        pass

    def get_pre_vertices(self):
        pass

    def get_post_vertices(self):
        pass

    def machine_vertices_for_recording(self, variable_to_record):
        pass

    def can_support_delays_up_to(self, max_delay):
        if max_delay > MAX_TIMER_TICS_SUPPORTED_PER_BLOCK:
            return False
        else:
            return True

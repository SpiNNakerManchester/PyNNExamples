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
from six import add_metaclass
from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class CommonSplitter(object):

    __slots__ = [
        "_governed_app_vertex"
    ]

    def __init__(self):
        self._governed_app_vertex = None

    def set_governed_app_vertex(self, app_vertex):
        self._governed_app_vertex = app_vertex

    def split(self, resource_tracker, machine_graph):
        """ executes splitting

        :param resource_tracker:
        :param machine_graph:
        :return:
        """
        success = self.create_machine_vertices(resource_tracker, machine_graph)
        if success:
            return self._do_delays(resource_tracker, machine_graph)
        else:
            return False

    def _do_delays(self, resource_tracker, machine_graph):
        """
        common delay code (NEEDS FILLING IN)
        :return: bool
        """
        return True

    @abstractmethod
    def create_machine_vertices(self, resource_tracker, machine_graph):
        """ method for specific splitter objects to use.

        :param resource_tracker:
        :param machine_graph:
        :return: bool true if successful, false otherwise
        """

    @abstractmethod
    def get_out_going_slices(self):
        """ allows a application vertex to control the set of slices for \
        outgoing application edges
        :return: list of Slices and bool of estimate or not
        """

    @abstractmethod
    def get_in_coming_slices(self):
        """ allows a application vertex to control the set of slices for \
        incoming application edges
        :return: the slices incoming to this vertex
        """

    @abstractmethod
    def get_pre_vertices(self):
        """

        :return: list of machine vertices
        """

    @abstractmethod
    def get_post_vertices(self):
        """

        :return: list of machine vertices
        """

    @abstractmethod
    def machine_vertices_for_recording(self, variable_to_record):
        """

        :param variable_to_record:
        :return: list of machine vertices
        """

    @abstractmethod
    def can_support_delays_up_to(self, max_delay):
        """

        :param max_delay:
        :return: bool
        """

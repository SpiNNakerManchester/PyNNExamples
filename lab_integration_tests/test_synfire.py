import os
import unittest


class TestSynfire(unittest.TestCase):

    def test_synfire(self):
        import spinnaker_graph_front_end.examples.hello_world as hw_dir
        class_file = hw_dir.__file__
        path = os.path.dirname(os.path.abspath(class_file))
        print path
        os.chdir(path)
        import spinnaker_graph_front_end.examples.hello_world.hello_world\
            as hw
        hw.__file__

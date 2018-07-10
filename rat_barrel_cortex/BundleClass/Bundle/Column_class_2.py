# coding=utf-8
import spynnaker8 as p
import numpy as np
import math
import unittest as unitt
from pyNN.utility.plotting import Figure, Panel, plot_spiketrains
from pyNN.random import NumpyRNG, RandomDistribution
from neo.core import Unit, Segment
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from L4.Layer_4 import Layer4
from L23.Layer_23 import Layer23

# L23 = Layer23(450, 79, 0.1, 0.19, 1.5, 0.75)


class Column:

    def __init__(self, src_Thal):

        self.Thal_src = src_Thal

        # === Instantiate Layer 4 ====

        self.L4 = Layer4(347, 61, 0.1, 0.19, 1.5, 0.75)

        # === Instantite Layer 4 connections ===

        self.L4.Thalconnector(self.Thal_src, 0.2, 1.0, 0.25)
        self.L4.Intralayerconnector(0.1, 0.1, 1.5, 1.5)
        self.L4.Interlayerconnector(self.L23.L23_exc_cells, self.L23.L23_inh_cells, 0.1, 1.5)

        # === Instantiate Layer 2/3 ====

        self.L23 = Layer23(450, 79, 0.1, 0.19, 1.5, 0.75)

        # === Instantiate Layer 2/3 connections ===

        self.L23.Intralayerconnector(0.1, 0.1)

    def DataRecording(self):

        self.L4.L4_exc_cells.record('spikes')
        self.L4.L4_inh_cells.record('spikes')

        self.L23.L23_exc_cells.record('spikes')
        self.L23.L23_inh_cells.record('spikes')

        self.L4_exc_data = self.L4.L4_exc_cells.get_data()
        self.L4_inh_data = self.L4.L4_inh_cells.get_data()
        self.L23_exc_data = self.L23.L23_exc_cells.get_data()
        self.L23_inh_data = self.L23.L23_inh_cells.get_data()

    def DataHandling(self):

        # === Append lists of exitatory and inhibitory spiketrains =====

        self.L4ei = self.L4_exc_data.segments[0].spiketrains + self.L4_inh_data.segments[0].spiketrains
        self.L23ei = self.L23_exc_data.segments[0].spiketrains + self.L23_inh_data.segments[0].spiketrains

        # === Create arrays shaped to size of E/I spiketrain lists to apply raster colour ===

        self.L4_exc_col = np.array([0, 0, 1] * self.L4.L4_exc_n).reshape(self.L4.L4_exc_n, -1)
        self.L4_inh_col = np.array([1, 0, 0] * self.L4.L4_inh_n).reshape(self.L4.L4_inh_n, -1)

        self.L23_exc_col = np.array([0, 0, 1] * self.L23.L23_exc_n).reshape(self.L23.L23_exc_n, -1)
        self.L23_inh_col = np.array([1, 0, 0] * self.L23.L23_inh_n).reshape(self.L23.L23_exc_n, -1)

        # === Append arrays of E/I colour codes

        self.L4_colorCodes = np.append(self.L4_exc_col, self.L4_inh_col, axis=0)
        self.L23_colorCodes = np.append(self.L23_exc_col, self.L23_inh_col, axis=0)



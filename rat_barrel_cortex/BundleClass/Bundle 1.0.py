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
from Bundle.Column_class_2 import Column


# Set simulator timestep & runtime
p.setup(0.1)
runtime = 1000

# Frequency and duration of Thalamic Stimuli
rate = 6  # Frequency of random stimulation [Hz]
stim_dur = 700  # Duration of random stimulation [ms]

Thal_n = 79

src_Thal = p.Population(
                    Thal_n, p.SpikeSourcePoisson(rate=rate, duration=stim_dur),
                    label="expoisson")

Col = Column(src_Thal)
Col.DataRecording()

p.run(runtime)

Col.DataHandling()
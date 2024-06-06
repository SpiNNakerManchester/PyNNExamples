# Copyright (c) 2023 The University of Manchester
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

import numpy
import math
from time import time, sleep
from functools import partial
import pyNN.spiNNaker as p
from matplotlib import pyplot, colors
from matplotlib.patches import Rectangle
from spynnaker.pyNN.utilities.utility_calls import get_n_bits
from threading import Thread

running = True


class CSVLine():
    def __init__(self, line):
        if len(line) == 0:
            raise EOFError
        parts = [int(part.strip()) for part in line.split(",")]
        if len(parts) < 4:
            raise EOFError
        self.x, self.y, self.p, send_time = parts
        self.send_time = send_time / 1000.0

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y}, send_time: {self.send_time})"

    def get_pos(self, min_x, min_y):
        return self.x - min_x, self.y - min_y, self.send_time

    def get_spike(self, width, min_x, min_y):
        return ((self.y - min_y) * width) + (self.x - min_x)


def make_kernel_circle(r, k_sz, weight, kernel):
    var = int((k_sz+1)/2-1)
    a = numpy.arange(0, 2 * math.pi, 0.01)
    dx = numpy.round(r * numpy.sin(a)).astype("uint32")
    dy = numpy.round(r * numpy.cos(a)).astype("uint32")
    kernel[var + dx, var + dy] = weight



def read_csv_line(f):
    line = f.readline()
    if not f:
        return None
    try:
        return CSVLine(line)
    except EOFError:
        return None


def send_spikes(width, height, min_x, min_y, run_time, label, connection):
    global running
    start_time = None
    max_x = min_x + width
    max_y = min_y + height
    y_shift = get_n_bits(width)
    with open("spikes.csv") as f:
        first_time = -1
        line = read_csv_line(f)
        while line and running:
            send_time = line.send_time
            if start_time is not None and send_time - start_time > run_time:
                return
            same_time_lines = []
            next_line = line
            while next_line and next_line.send_time == send_time:
                same_time_lines.append(next_line)
                next_line = read_csv_line(f)
            line = next_line

            filtered_lines = [
                l for l in same_time_lines
                if (l.x >= min_x and l.x < max_x and l.y >= min_y and
                    l.y < max_y)]

            if not filtered_lines:
                continue

            spikes = [s.get_spike(width, min_x, min_y) for s in filtered_lines]
            spikes = spikes[:20]

            if first_time == -1:
                first_time = send_time
                start_time = time()

            sleep_time = (time() - start_time) - (send_time - first_time)
            if sleep_time > 0:
                sleep(sleep_time)
            connection.send_spikes(label, spikes, send_full_keys=True)
    running = False


WIDTH = 120
HEIGHT = 120
MIN_X = 450
MIN_Y = 400
PER_CORE_WIDTH = 16
PER_CORE_HEIGHT = 16
SPIF_IP = "spif-01"
SPIF_PORT = 3332
POP_LABEL = "target"
SEND_POP_LABEL = "source"
RUN_TIME = 5000
CHIP = (0, 0)

scaler = 0.1
k_sz = 39
pos_w = 0.8
neg_w = -1.0

kernel = numpy.zeros((k_sz, k_sz))
make_kernel_circle(0.46*k_sz, k_sz, pos_w*scaler, kernel)
make_kernel_circle(0.41*k_sz, k_sz, neg_w*scaler, kernel)
make_kernel_circle(0.36*k_sz, k_sz, pos_w*scaler, kernel)
make_kernel_circle(0.26*k_sz, k_sz, neg_w*scaler, kernel)

pyplot.imshow(kernel, interpolation='nearest')
pyplot.savefig("kernel.png")

convolution = p.ConvolutionConnector(kernel_weights=kernel)
out_width, out_height = convolution.get_post_shape((WIDTH, HEIGHT))

print(f"Output {out_width} x {out_height}")


pyplot.ion()
colours = ["black", "g"]
cmap = colors.ListedColormap(colours)
image_data = numpy.zeros((WIDTH, HEIGHT))
fig, axes = pyplot.subplots(figsize=(8, 8))
plot = axes.imshow(image_data, interpolation="nearest", cmap="Greens", vmin=0,
                   vmax=100)
fig.canvas.draw()
fig.canvas.flush_events()
rect_pos = None
rect_count = None

redraw = False


def recv(label, time, spikes):
    global redraw, image_data
    np_spikes = numpy.array(spikes)
    ys, xs = numpy.divmod(np_spikes, WIDTH)
    image_data[xs, ys] += 100
    redraw = True


def recv_conv(label, time, spikes):
    global redraw, image_data, rect_pos, rect_count
    np_spikes = numpy.array(spikes)
    square_row, rem = numpy.divmod(np_spikes, out_width * PER_CORE_WIDTH)
    square_col, rem = numpy.divmod(rem, PER_CORE_WIDTH * PER_CORE_HEIGHT)
    in_square_y, in_square_x = numpy.divmod(rem, PER_CORE_WIDTH)
    xs = (k_sz//2) + square_row * PER_CORE_HEIGHT + in_square_x
    ys = (k_sz//2) + square_col * PER_CORE_WIDTH + in_square_y
    rect_pos = (numpy.amin(xs), numpy.amin(ys), numpy.amax(xs), numpy.amax(ys))
    rect_count = 4


conn = p.external_devices.SpynnakerLiveSpikesConnection(
    receive_labels=[SEND_POP_LABEL, POP_LABEL], send_labels=[SEND_POP_LABEL], local_port=None)
conn.add_receive_callback(SEND_POP_LABEL, recv)
conn.add_receive_callback(POP_LABEL, recv_conv)
conn.add_start_callback(
    SEND_POP_LABEL, partial(send_spikes, WIDTH, HEIGHT, MIN_X, MIN_Y, RUN_TIME / 1000.0))


p.setup(1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp,
                                 (PER_CORE_WIDTH, PER_CORE_HEIGHT))

retina = p.Population(
    WIDTH * HEIGHT, p.external_devices.SpikeInjector(),
    structure=p.Grid2D(WIDTH / HEIGHT), label=SEND_POP_LABEL)

target_pop = p.Population(
    out_width * out_height, p.IF_curr_exp(),
    structure=p.Grid2D(out_width / out_height), label=POP_LABEL)

p.Projection(retina, target_pop, convolution, p.Convolution())
p.external_devices.activate_live_output_for(
    retina, database_notify_port_num=conn.local_port)
p.external_devices.activate_live_output_for(
    target_pop, database_notify_port_num=conn.local_port)


def do_run():
    global running
    p.external_devices.run_forever()
    running = False
    p.end()


t = Thread(target=do_run)
t.start()

rect = None
while running and fig.get_visible():
    try:
        plot.set_array(image_data)
        if rect_count is not None:
            rect_count -= 1
            if rect_count == 0:
                rect_pos = None
                rect_count = None
                rect.set_visible(False)
        if rect_pos is not None:
            x_min, y_min, x_max, y_max = rect_pos
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            if rect is None:
                rect = Rectangle((x_min, y_min), width, height, linewidth=2,
                                 edgecolor='r', facecolor='r')
                axes.add_patch(rect)
            else:
                rect.set_visible(True)
                rect.set_bounds(x_min, y_min, width, height)
        fig.canvas.draw()
        fig.canvas.flush_events()
        image_data *= 0.5
        sleep(0.1)
    except Exception:
        break

p.external_devices.request_stop()
t.join()

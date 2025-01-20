# Copyright (c) 2018 The University of Manchester
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

import tkinter as tk
import tkinter.font as tkFont
import sys
import functools
import pyNN.spiNNaker as p
from sudoku.utils import puzzles, get_rates


class GUI(object):
    """
    A Gui for setting the numbers in the puzzle.
    """

    def __init__(self, n_total, n_cell, n_N, default_rate, max_rate, puzzle):
        """
        :param int n_total:
        :param int n_cell:
        :param int n_N:
        :param float default_rate:
        :param float max_rate:
        :param int puzzle:
        """
        self._n_total = n_total
        self._n_cell = n_cell
        self._n_N = n_N
        self._after_id = None
        self._puzzle = puzzle
        self._connection = \
            p.external_devices.SpynnakerPoissonControlConnection(
                poisson_labels=["Noise"], local_port=None)
        print(f"Listening on {self._connection.local_port}")
        sys.stdout.flush()
        self._connection.add_start_callback("Noise", self.on_start)
        self._root = tk.Tk()
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14)
        self._root.option_add("*Font", default_font)
        self._root.title("Set Numbers")
        self._numbers = [[None for _ in range(9)] for _ in range(9)]
        self._numbers_txt = [[None for _ in range(9)] for _ in range(9)]
        init = puzzles[puzzle]
        self._rates = get_rates(
            init, n_total, n_cell, n_N, default_rate, max_rate)
        for x in range(9):
            for y in range(9):
                row = x if x < 3 else x + 1 if x < 6 else x + 2
                col = y if y < 3 else y + 1 if y < 6 else y + 2
                var = tk.StringVar(self._root)
                var.set(str(init[x][y]))
                item = tk.Spinbox(
                    self._root, from_=0, to=9, width=5, textvariable=var)
                item.grid(row=row, column=col, padx=5, pady=5)
                self._numbers[x][y] = item
                self._numbers_txt[x][y] = var

        tk.Frame(self._root, height=10).grid(row=3, column=0, columnspan=11)
        tk.Frame(self._root, height=10).grid(row=7, column=0, columnspan=11)
        tk.Frame(self._root, width=20).grid(row=0, column=3, rowspan=11)
        tk.Frame(self._root, width=20).grid(row=0, column=7, rowspan=11)

        frame = tk.Frame(self._root)
        for i in range(len(puzzles)):
            button = tk.Button(
                frame, text=f"Puzzle {i + 1}", width=10,
                command=functools.partial(self.set_puzzle, i))
            button.grid(row=i, column=0, pady=5)
        button = tk.Button(frame, text="Dream", width=10,
                           command=self.set_dream)
        button.grid(row=i + 1, column=0, pady=5)
        frame.grid(row=0, column=11, rowspan=11, padx=5)

        def_rate_frame = tk.Frame(self._root)
        def_rate_label = tk.Label(def_rate_frame, text="Noise Rate")
        def_rate_label.grid(row=0, column=0, padx=5)
        def_rate = tk.StringVar(def_rate_frame)
        def_rate.set(str(default_rate))
        self._default_rate = tk.Spinbox(
            def_rate_frame, from_=0, to=100, width=5, textvariable=def_rate)
        self._default_rate.grid(row=0, column=1, padx=5)
        def_rate_frame.grid(row=11, column=0, columnspan=11, pady=5)

        max_rate_frame = tk.Frame(self._root)
        max_rate_label = tk.Label(
            max_rate_frame, text="Fixed Number Rate", padx=5, pady=5)
        max_rate_label.grid(row=0, column=0, padx=5)
        maximum_rate = tk.StringVar(max_rate_frame)
        maximum_rate.set(str(max_rate))
        self._max_rate = tk.Spinbox(
            max_rate_frame, from_=0, to=max_rate, width=5,
            textvariable=maximum_rate)
        self._max_rate.grid(row=0, column=1, padx=5)
        max_rate_frame.grid(row=12, column=0, columnspan=11, pady=5)

        button_frame = tk.Frame(self._root)
        self._button = tk.Button(
            button_frame, text='Set Numbers', width=25,
            command=self.set_values, state="disabled")
        self._button.grid(row=0, column=0, padx=5)
        self._cycle_button = tk.Button(
            button_frame, text="Cycle Puzzles", width=25,
            command=self.cycle_puzzles, state="disabled")
        self._cycle_button.grid(row=0, column=1, padx=5)
        button_frame.grid(row=13, column=0, columnspan=11, pady=5)

    def start(self):
        """
        Starts the main loop running

        """
        self._root.mainloop()

    def on_start(self, label, connection):
        """
        set the button and cycle button to normal
        """
        # pylint: disable=unused-argument
        self._button["state"] = "normal"
        self._cycle_button["state"] = "normal"

    def cycle_puzzles(self):
        """
        Cycle through the puzzles at 30 seconds intervals
        """
        self._after_id = self._root.after(30000, self.cycle_puzzles)
        print("Cycling after 30 seconds")
        self._puzzle = (self._puzzle + 1) % len(puzzles)
        print("Updating puzzle to", self._puzzle)
        self.set_puzzle(self._puzzle, automatic=True)

    def set_dream(self):
        """
        Sets the numbers to zero and based on that the coonections and rates
        """
        for x in range(9):
            for y in range(9):
                self._numbers_txt[x][y].set(0)
        self.set_values()

    def set_puzzle(self, puzzle, automatic=False):
        """
        Sets the puzzel number and based on that the numbers, coonections \
        and rates

        :param int puzzle:
        :param bool automatic:
        """
        self._puzzle = puzzle
        for x in range(9):
            for y in range(9):
                self._numbers_txt[x][y].set(puzzles[puzzle][x][y])
        self.set_values(automatic)

    def set_values(self, automatic=False):
        """
        Stes the connections and rates

        :param bool automatic:
        :return:
        """
        if not automatic:
            if self._after_id is not None:
                print("Cancelling")
                self._root.after_cancel(self._after_id)
            self._after_id = self._root.after(120000, self.cycle_puzzles)
            print("Setting timeout to 120 seconds")
        new_values = [[0 for _ in range(9)] for _ in range(9)]
        for x in range(9):
            for y in range(9):
                new_values[x][y] = int(self._numbers[x][y].get())
        new_rates = get_rates(
            new_values, self._n_total, self._n_cell, self._n_N,
            int(self._default_rate.get()), int(self._max_rate.get()))
        updated_rates = list()
        for i, rate in enumerate(new_rates):
            if rate != self._rates[i]:
                updated_rates.append((i, rate))
        self._connection.set_rates("Noise", updated_rates)
        self._rates = new_rates


if __name__ == "__main__":
    gui = GUI(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
              float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
    gui.start()

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

puzzles = list()
puzzles.append(
    # Diabolical problem:
    [[0, 0, 1,  0, 0, 8,  0, 7, 3],
     [0, 0, 5,  6, 0, 0,  0, 0, 1],
     [7, 0, 0,  0, 0, 1,  0, 0, 0],

     [0, 9, 0,  8, 1, 0,  0, 0, 0],
     [5, 3, 0,  0, 0, 0,  0, 4, 6],
     [0, 0, 0,  0, 6, 5,  0, 3, 0],

     [0, 0, 0,  1, 0, 0,  0, 0, 4],
     [8, 0, 0,  0, 0, 9,  3, 0, 0],
     [9, 4, 0,  5, 0, 0,  7, 0, 0]])
puzzles.append(
    [[2, 0, 0,  0, 0, 6,  0, 3, 0],
     [4, 8, 0,  0, 1, 9,  0, 0, 0],
     [0, 0, 7,  0, 2, 0,  9, 0, 0],

     [0, 0, 0,  3, 0, 0,  0, 9, 0],
     [7, 0, 8,  0, 0, 0,  1, 0, 5],
     [0, 4, 0,  0, 0, 7,  0, 0, 0],

     [0, 0, 4,  0, 9, 0,  6, 0, 0],
     [0, 0, 0,  6, 4, 0,  0, 1, 9],
     [0, 5, 0,  1, 0, 0,  0, 0, 8]])
puzzles.append(
    [[0, 0, 3,  2, 0, 0,  0, 7, 0],
     [0, 0, 5,  0, 0, 0,  3, 0, 0],
     [0, 0, 8,  9, 7, 0,  0, 5, 0],

     [0, 0, 0,  8, 9, 0,  0, 0, 0],
     [0, 5, 0,  0, 0, 0,  0, 2, 0],
     [0, 0, 0,  0, 6, 1,  0, 0, 0],

     [0, 1, 0,  0, 2, 5,  6, 0, 0],
     [0, 0, 4,  0, 0, 0,  8, 0, 0],
     [0, 9, 0,  0, 0, 7,  5, 0, 0]])
puzzles.append(
    [[0, 1, 0,  0, 0, 0,  0, 0, 2],
     [8, 7, 0,  0, 0, 0,  5, 0, 4],
     [5, 0, 2,  0, 0, 0,  0, 9, 0],

     [0, 5, 0,  4, 0, 9,  0, 0, 1],
     [0, 0, 0,  7, 3, 2,  0, 0, 0],
     [9, 0, 0,  5, 0, 1,  0, 4, 0],

     [0, 2, 0,  0, 0, 0,  4, 0, 8],
     [4, 0, 6,  0, 0, 0,  0, 1, 3],
     [1, 0, 0,  0, 0, 0,  0, 2, 0]])
puzzles.append(
    [[8, 9, 0,  2, 0, 0,  0, 7, 0],
     [0, 0, 0,  0, 8, 0,  0, 0, 0],
     [0, 4, 1,  0, 3, 0,  5, 0, 0],

     [2, 5, 8,  0, 0, 0,  0, 0, 6],
     [0, 0, 0,  0, 0, 0,  0, 0, 0],
     [6, 0, 0,  0, 0, 0,  1, 4, 7],

     [0, 0, 7,  0, 1, 0,  4, 3, 0],
     [0, 0, 0,  0, 2, 0,  0, 0, 0],
     [0, 2, 0,  0, 0, 7,  0, 5, 1]])
puzzles.append(
    # "World's hardest sudoku":
    # http://www.telegraph.co.uk/news/science/science-news/9359579/\
    # Worlds-hardest-sudoku-can-you-crack-it.html
    [[8, 0, 0,  0, 0, 0,  0, 0, 0],
     [0, 0, 3,  6, 0, 0,  0, 0, 0],
     [0, 7, 0,  0, 9, 0,  2, 0, 0],

     [0, 5, 0,  0, 0, 7,  0, 0, 0],
     [0, 0, 0,  0, 4, 5,  7, 0, 0],
     [0, 0, 0,  1, 0, 0,  0, 3, 0],

     [0, 0, 1,  0, 0, 0,  0, 6, 8],
     [0, 0, 8,  5, 0, 0,  0, 1, 0],
     [0, 9, 0,  0, 0, 0,  4, 0, 0]])
puzzles.append(
    [[1, 0, 0,  4, 0, 0,  0, 0, 0],
     [7, 0, 0,  5, 0, 0,  6, 0, 3],
     [0, 0, 0,  0, 3, 0,  4, 2, 0],

     [0, 0, 9,  0, 0, 0,  0, 3, 5],
     [0, 0, 0,  3, 0, 5,  0, 0, 0],
     [6, 3, 0,  0, 0, 0,  1, 0, 0],

     [0, 2, 6,  0, 5, 0,  0, 0, 0],
     [9, 0, 4,  0, 0, 6,  0, 0, 7],
     [0, 0, 0,  0, 0, 8,  0, 0, 2]])


def get_rates(values, n_total, n_cell, n_N, default_rate, max_rate):
    rates = [default_rate] * n_total
    for x in range(9):
        for y in range(9):
            if values[8 - y][x] != 0:
                base = ((y * 9) + x) * n_cell
                for j in range(n_N * (values[8 - y][x] - 1),
                               n_N * values[8 - y][x]):
                    rates[j + base] = max_rate
    return rates

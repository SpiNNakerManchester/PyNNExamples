# Copyright (c) 2019 The University of Manchester
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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio
import os

def load_connections(npy_label, pop_size, rec=True):
    in_conn = [list(ele) for ele in np.load(npy_label+' in.npy').tolist()]
    if rec:
        rec_conn = [list(ele) for ele in np.load(npy_label+' rec.npy').tolist()]
    out_conn = [list(ele) for ele in np.load(npy_label+' out.npy').tolist()]
    for ndx in range(len(in_conn)):
        if in_conn[ndx][3] == 16 and in_conn[ndx][0] == 0:
            in_conn[ndx][3] = 0
    if rec:
        for ndx in range(len(rec_conn)):
            if rec_conn[ndx][3] == 16 and rec_conn[ndx][0] == 0:
                rec_conn[ndx][3] = 0
    for ndx in range(len(out_conn)):
        if out_conn[ndx][3] == 16 and out_conn[ndx][0] == 0:
            out_conn[ndx][3] = 0
    checking_delays = [[] for i in range(pop_size)]
    list_to_check = in_conn
    if rec:
        list_to_check = in_conn+rec_conn
    for [pre, post, weight, delay] in list_to_check:
        if delay not in checking_delays[post]:
            checking_delays.append(delay)
        else:
            print("delays are overlapped")
            Exception
    if not rec:
        rec_conn = []
    return in_conn, rec_conn, out_conn

def add_edges(conn_list, G, pre, post):
    for conn in conn_list:
        G.add_edge(pre+'{}'.format(conn[0]), post+'{}'.format(conn[1]), weight=conn[2])
    return G

def create_lists(conn_list, pre, post):
    edge_list = []
    weight_list = []
    colour_list = []
    for conn in conn_list:
        # G.add_edge(pre+'{}'.format(conn[0]), post+'{}'.format(conn[1]), weight=conn[2])
        edge_list.append([pre+'{}'.format(conn[0]), post+'{}'.format(conn[1])])
        weight_list.append(conn[2]*4)
        colour_list.append(conn[2]*4)
    return edge_list, weight_list, colour_list

def condense_inputs(conn_list):
    new_list = []
    for [pre, post, weight, delay] in conn_list:
        new_list.append([np.floor(pre / 10), post, weight, delay])
    combined_weights = [{} for i in range(4)]
    for [pre, post, weight, delay] in new_list:
        if '{}'.format(post) in combined_weights[int(pre)]:
            combined_weights[int(pre)]['{}'.format(post)] += weight
        else:
            combined_weights[int(pre)]['{}'.format(post)] = weight
    condensed_list = []
    for dict in combined_weights:
        for post in dict:
            condensed_list.append([combined_weights.index(dict), int(post), dict[post]/10.])
    return condensed_list

def create_pos(in_size, rec_size, out_size):
    pos = {}
    for node in range(in_size):
        pos['in{}'.format(node)] = np.array([0, (node * (100. / in_size) + (100. / (in_size * 2.)))])
    # for node in range(in_size):
    #     x = 0.5 - 0.5*(np.sin((np.pi*(node/float(in_size))/2) + np.pi/4.))
    #     y = (node * (150. / in_size) + (150. / (in_size * 2.))) - 25
    #     pos['in{}'.format(node)] = np.array([x, y])
    for node in range(rec_size):
        pos['h{}'.format(node)] = np.array([1, (node * (100. / rec_size) + (100. / (rec_size * 2.)))])
    # for node in range(rec_size):
    #     x = 1 + 0.5*(np.sin(2.*np.pi*(node/float(rec_size))))
    #     y = 50 + 50*(np.cos(2.*np.pi*(node/float(rec_size))))
    #     pos['h{}'.format(node)] = np.array([x, y])
    for node in range(out_size):
        pos['out{}'.format(node)] = np.array([2, (node * (100. / out_size) + (100. / (out_size * 2.)))])
    return pos

def draw_graph_from_file(address_string, test_label, rec_flag, save_flag=False):
    G = nx.Graph()

    from_list_in, from_list_rec, from_list_out = load_connections(address_string+test_label, 20, rec=rec_flag)
    from_list_in = condense_inputs(from_list_in)
    G = add_edges(from_list_in, G, 'in', 'h')
    G = add_edges(from_list_rec, G, 'h', 'h')
    G = add_edges(from_list_out, G, 'h', 'out')
    all_edges = []
    all_weights = []
    all_colours = []
    edges, weights, colours = create_lists(from_list_in, 'in', 'h')
    all_edges += edges
    all_weights += weights
    all_colours += colours
    edges, weights, colours = create_lists(from_list_rec, 'h', 'h')
    all_edges += edges
    all_weights += weights
    all_colours += colours
    edges, weights, colours = create_lists(from_list_out, 'h', 'out')
    all_edges += edges
    all_weights += weights
    all_colours += colours

    hidden_nodes = int(np.amax(from_list_in, axis=0)[1] + 1)

    pos = create_pos(4, hidden_nodes, 2)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=all_weights, edge_color=all_colours)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    # plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # print(manager.window.maxsize())
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    plt.tight_layout()
    if save_flag:
        plt.savefig(address_string+test_label+".png", bbox_inches='tight') # save as png
    # plt.show() # display
    plt.close()

def draw_graph_from_list(from_list_in, from_list_rec, from_list_out, address_string=None, test_label=None, rec_flag=False, save_flag=False, plot_flag=False):
    G = nx.Graph()

    # from_list_in, from_list_rec, from_list_out = load_connections(address_string+test_label, 20, rec=rec_flag)
    from_list_in = condense_inputs(from_list_in)
    G = add_edges(from_list_in, G, 'in', 'h')
    G = add_edges(from_list_rec, G, 'h', 'h')
    G = add_edges(from_list_out, G, 'h', 'out')
    all_edges = []
    all_weights = []
    all_colours = []
    edges, weights, colours = create_lists(from_list_in, 'in', 'h')
    all_edges += edges
    all_weights += weights
    all_colours += colours
    edges, weights, colours = create_lists(from_list_rec, 'h', 'h')
    all_edges += edges
    all_weights += weights
    all_colours += colours
    edges, weights, colours = create_lists(from_list_out, 'h', 'out')
    all_edges += edges
    all_weights += weights
    all_colours += colours

    hidden_nodes = int(np.amax(from_list_in, axis=0)[1] + 1)

    pos = create_pos(4, hidden_nodes, 2)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=all_weights, edge_color=all_colours)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    # plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    # print(manager.window.maxsize())
    plt.tight_layout()
    if save_flag:
        plt.savefig(address_string+test_label+" SNN.png", bbox_inches='tight') # save as png
    if plot_flag:
        plt.show() # display
    plt.close()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curve(correct_or_not, cycle_error, address_string, test_label, save_flag=False, cue_break=[]):
    # if not isinstance(data[0], list):
    #     data = [data]
    num_repeats = len(correct_or_not)
    ave_corr10 = moving_average(correct_or_not, 10)
    ave_corr64 = moving_average(correct_or_not, 64)

    fig, axs = plt.subplots(4, 1)
    fig.suptitle(test_label, fontsize=16)
    axs[0].scatter([i for i in range(len(correct_or_not))], correct_or_not)
    axs[0].set_xlim([0, num_repeats])
    for iteration_break in cue_break:
        axs[0].axvline(x=iteration_break, color='b')
    axs[0].set_title('Thresholded performance')
    axs[1].scatter([i for i in range(len(cycle_error))], cycle_error)
    axs[1].set_xlim([0, num_repeats])
    axs[1].plot([0, len(cycle_error)], [75, 75], 'r')
    for iteration_break in cue_break:
        axs[1].axvline(x=iteration_break, color='b')
    axs[1].set_title('Iteration error')
    axs[2].plot([i + 5 for i in range(len(ave_corr10))], ave_corr10)
    axs[2].plot([0, num_repeats], [0.9, 0.9], 'r')
    axs[2].plot([0, num_repeats], [0.95, 0.95], 'g')
    for iteration_break in cue_break:
        axs[2].axvline(x=iteration_break-5, color='b')
    axs[2].set_xlim([0, num_repeats])
    axs[2].set_title('Averaged performance over 10 cycles')
    axs[3].plot([i + 32 for i in range(len(ave_corr64))], ave_corr64)
    axs[3].plot([0, num_repeats], [0.9, 0.9], 'r')
    axs[3].plot([0, num_repeats], [0.95, 0.95], 'g')
    axs[3].set_xlim([0, num_repeats])
    for iteration_break in cue_break:
        axs[3].axvline(x=iteration_break-32, color='b')
    axs[3].set_title('Averaged performance over 64 cycles')

    # plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # manager.full_screen_toggle()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    # print(manager.window.maxsize())
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_flag:
        plt.savefig(address_string+test_label+" learning curve.png", bbox_inches='tight')
    # plt.show()
    plt.close()


def create_video(top_directory, base_label, string_end=' SNN.png'):
    files_for_video = [[], [], [], []]
    for root, dirs, files in os.walk(top_directory):
        for filename in files:
            if base_label in filename and string_end in filename:
                if 'c-1' in filename:
                    iteration = filename.replace(string_end, '')
                    for i in range(2, 5, 1):
                        if iteration[-i] == ' ':
                            break
                    iteration = int(iteration[-i+1:])
                    files_for_video[0].append([filename, iteration])
                elif 'c-3' in filename:
                    iteration = filename.replace(string_end, '')
                    for i in range(2, 5, 1):
                        if iteration[-i] == ' ':
                            break
                    iteration = int(iteration[-i+1:])
                    files_for_video[1].append([filename, iteration])
                elif 'c-5' in filename:
                    iteration = filename.replace(string_end, '')
                    for i in range(2, 5, 1):
                        if iteration[-i] == ' ':
                            break
                    iteration = int(iteration[-i+1:])
                    files_for_video[2].append([filename, iteration])
                elif 'c-7' in filename:
                    iteration = filename.replace(string_end, '')
                    for i in range(2, 5, 1):
                        if iteration[-i] == ' ':
                            break
                    iteration = int(iteration[-i+1:])
                    files_for_video[3].append([filename, iteration])
                else:
                    print("incorrect file name", filename)
    files_for_video[0] = sorted(files_for_video[0], key=lambda l: l[1])
    files_for_video[1] = sorted(files_for_video[1], key=lambda l: l[1])
    files_for_video[2] = sorted(files_for_video[2], key=lambda l: l[1])
    files_for_video[3] = sorted(files_for_video[3], key=lambda l: l[1])

    print("creating video")
    images = []
    for filenames in files_for_video:
        for filename in filenames:
            images.append(imageio.imread(top_directory+'/'+filename[0]))
    imageio.mimsave(top_directory+'/videos/'+base_label+string_end+'.gif', images)

if __name__ == '__main__':
    directory = '/localhome/mbaxrap7/eprop_python3/PyNN8Examples/eprop_testing/big_with_labels'
    # label = 'eta-0.1_0.05 - size-40_100 - rec-False'
    # label = 'eta-0.03_0.015 - size-40_100 - rec-False'
    label = 't35 eta-0.03_0.015 - size-40_100 - rec-False'
    create_video(directory, label,
                 # string_end=' learning curve.png')
                 string_end=' SNN.png')
    create_video(directory, label,
                 string_end=' learning curve.png')
                 # string_end=' SNN.png')

    print("done")

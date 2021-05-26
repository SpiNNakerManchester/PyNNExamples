import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio
import os
import seaborn as sn
import pandas as pd

def load_connections(npy_label, pop_size, rec=True):
    in_conn = [list(ele) for ele in np.load(npy_label + ' in.npy').tolist()]
    if rec:
        rec_conn = [list(ele) for ele in np.load(npy_label + ' rec.npy').tolist()]
    out_conn = [list(ele) for ele in np.load(npy_label + ' out.npy').tolist()]
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
        list_to_check = in_conn + rec_conn
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
        G.add_edge(pre + '{}'.format(conn[0]), post + '{}'.format(conn[1]), weight=conn[2])
    return G


def create_lists(conn_list, pre, post):
    edge_list = []
    weight_list = []
    colour_list = []
    for conn in conn_list:
        # G.add_edge(pre+'{}'.format(conn[0]), post+'{}'.format(conn[1]), weight=conn[2])
        edge_list.append([pre + '{}'.format(conn[0]), post + '{}'.format(conn[1])])
        weight_list.append(conn[2] * 4)
        colour_list.append(conn[2] * 4)
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
            condensed_list.append([combined_weights.index(dict), int(post), dict[post] / 10.])
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

    from_list_in, from_list_rec, from_list_out = load_connections(address_string + test_label, 20, rec=rec_flag)
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
        plt.savefig(address_string + test_label + ".png", bbox_inches='tight')  # save as png
    # plt.show() # display
    plt.close()


def draw_graph_from_list(from_list_in, from_list_rec, from_list_out, address_string=None, test_label=None,
                         rec_flag=False, save_flag=False, plot_flag=False):
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
        plt.savefig(address_string + test_label + " SNN.png", bbox_inches='tight')  # save as png
    if plot_flag:
        plt.show()  # display
    plt.close()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def graph_weights(all_weights, address_string, test_label, max_weight, plot_flag=False):
    print("processing weights")
    all_weight_list = []
    for weights in all_weights:
        weight_list = []
        for conn in weights:
            if conn:
                weight_list.append(conn[2])
        all_weight_list.append(weight_list)
    fig, axs = plt.subplots(1, len(all_weight_list))
    for idx, weights in enumerate(all_weight_list):
        axs[idx].scatter([i for i in range(len(weights))], weights, s=2)
        axs[idx].set_ylim([-max_weight, max_weight])

    plt.suptitle(test_label, fontsize=16)
    print("plotting weights")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    # print(manager.window.maxsize())
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(address_string + "weights " + test_label + ".png", bbox_inches='tight', dpi=200)
    if plot_flag:
        plt.show()
    print("weights plotted")
    plt.close()

def plot_learning_curve(correct_or_not, cycle_error, confusion_matrix, final_confusion_matrix,
                        output_size,
                        address_string, test_label, save_flag=False, cue_break=[], plot_flag=False,
                        learning_threshold=0.75, no_classes=10):
    fig, axs = plt.subplots(2, 2)
    df_cm = pd.DataFrame(confusion_matrix, range(output_size+1), range(output_size+1))
    f_df_cm = pd.DataFrame(final_confusion_matrix, range(output_size+1), range(output_size+1))
    ave_err10 = moving_average(cycle_error, 10)
    ave_err100 = moving_average(cycle_error, 100)
    ave_err1000 = moving_average(cycle_error, 1000)
    axs[0][0].scatter([i for i in range(len(cycle_error))], cycle_error)
    axs[0][0].plot([i + 5 for i in range(len(ave_err10))], ave_err10, 'r')
    axs[0][0].plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][0].plot([i + 500 for i in range(len(ave_err1000))], ave_err1000, 'g')
    axs[0][0].plot([0, len(cycle_error)], [900, 900], 'g')
    axs[0][0].set_xlim([0, len(cycle_error)])
    axs[0][0].set_ylim([0, 1000])
    axs[0][0].set_title("cycle error")
    for iteration_break in cue_break:
        axs[0][0].axvline(x=iteration_break, color='b')
    ave_corr10 = moving_average(correct_or_not, 10)
    ave_err100 = moving_average(correct_or_not, 100)
    ave_err1000 = moving_average(correct_or_not, 1000)
    axs[0][1].scatter([i for i in range(len(correct_or_not))], correct_or_not)
    axs[0][1].plot([i + 5 for i in range(len(ave_corr10))], ave_corr10, 'r')
    axs[0][1].plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][1].plot([i + 500 for i in range(len(ave_err1000))], ave_err1000, 'g')
    axs[0][1].plot([0, len(correct_or_not)], [0.5, 0.5], 'r')
    random_chance = 1. / float(no_classes)
    axs[0][1].plot([0, len(correct_or_not)], [random_chance, random_chance], 'r')
    axs[0][1].plot([0, len(correct_or_not)], [learning_threshold, learning_threshold], 'g')
    if len(ave_err1000) > 0:
        axs[0][1].plot([0, len(correct_or_not)], [ave_err1000[-1], ave_err1000[-1]], 'b')
    for iteration_break in cue_break:
        axs[0][1].axvline(x=iteration_break, color='b')
    axs[0][1].set_xlim([0, len(correct_or_not)])
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_title("classification error")
    for iteration_break in cue_break:
        axs[0][1].axvline(x=iteration_break, color='b')
    axs[1][0] = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, ax=axs[1][0])  # font size
    axs[1][0].set_title("full confusion matrix")
    axs[1][1] = sn.heatmap(f_df_cm, annot=True, annot_kws={"size": 8}, ax=axs[1][1])  # font size
    axs[1][1].set_title("window confusion matrix")
    plt.suptitle(test_label, fontsize=16)
    # plt.title(experiment_label)
    if plot_flag:
        plt.show()

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
        plt.savefig(address_string + test_label + " learning curve.png", bbox_inches='tight', dpi=200)
    # plt.show()
    plt.close()


def create_video(graph_directory, gif_directory, base_label, string_match=' SNN.png', repeat_best=20, not_match='empty'):
    print("\nStarting video creation of ", base_label)
    files_for_video = []
    for root, dirs, files in os.walk(graph_directory):
        for filename in files:
            if base_label in filename and string_match in filename and not_match not in filename:
                iteration = filename.replace(string_match, '')
                iteration = iteration.replace(base_label, '')
                files_for_video.append([filename, int(iteration)])
    files_for_video = sorted(files_for_video, key=lambda l: l[1])
    corrupt = True
    invalid_count = 0
    while corrupt and len(files_for_video) > 0:
        try:
            test = imageio.imread(graph_directory + '/' + files_for_video[-1][0])
            files_for_video.append([files_for_video[-1][0], files_for_video[-1][1]])
            corrupt = False
        except:
            print("invalid final file - deleting and repeating", invalid_count)
            del files_for_video[-1]

    if not len(files_for_video):
        print("no files for video creation of", base_label)
        return 0

    files_for_video[-1][1] = 0
    for i in range(repeat_best):
        files_for_video.append(files_for_video[-1])
    files_for_video = sorted(files_for_video, key=lambda l: l[1])

    print("creating video of length", files_for_video[-1][1])

    # with imageio.get_writer(gif_directory + '/gifs/' + base_label + '.gif', mode='I') as writer:
    #     for filename in files_for_video:
    #         image = imageio.imread(graph_directory + '/' + filename[0])
    #         writer.append_data(image)

    images = []
    for filename in files_for_video:
        images.append(imageio.imread(graph_directory + '/' + filename[0]))
    imageio.mimsave(gif_directory + '/gifs/' + base_label + '.gif', images)
    print("Finished", base_label)


if __name__ == '__main__':
    gif_directory = '/localhome/mbaxrap7/eprop_python3/PyNN8Examples/eprop_testing/shd_graphs'
    graph_directory = '/data/mbaxrap7/shd_graphs'

    h_eta = [0.03]
    r_eta = [0.001, 0.0006, 0.0004, 0.0002]
    var_v = [0.]
    var_f = [0.01]
    w_fb = [3]
    fb_m = [30., 100., 300., 1000.]

    processes = []
    logs = []
    for h in h_eta:
        for r in r_eta:
            for vf in var_f:
                for vv in var_v:
                    for fb in w_fb:
                        for m in fb_m:
                            h = r
                            # label = 'lc norm8 eta(1) h{}o{} - b0.3-0.5 - tr10-{} - vmem{} - fb{}x{}' \
                            #         ' - in0.0 out0.0 rec0.0False (1x256)'.format(h, r, vf, vv, fb, m)
                            # 'lc reg-Lfb eta(3) h1.0o0.0003 - b0.3-0.5 - tr95-0.001 - vmem0.0 - fb4 - in0.0 out0.0 rec0.0False (1x256) 290 learning curve.png'
                            # lc norm8 eta(1) h0.0006o0.0006 - b0.3-0.5 - tr10-0.0 - vmem0.6 - fb3x3.0 - in0.0 out0.0 rec0.0False (1x256)
                            label = 'lc no rec10 eta(1) h{}o{} - b0.3-0.5 - tr5-{} - vmem{} - fb{}x{}' \
                                    ' - in0.0 out0.0 rec0.0False (1x256)'.format(h, r, vf, vv, fb, m)
                            create_video(graph_directory, gif_directory, label,
                                         # string_match='learning curve.png',
                                         string_match='.png',
                                         not_match='shd_graphs')

    print("done")
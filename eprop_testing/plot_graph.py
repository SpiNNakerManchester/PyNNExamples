import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import imageio
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
    if save_flag:
        plt.savefig(address_string+test_label+".png") # save as png
    plt.show() # display
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
    if save_flag:
        plt.savefig(address_string+test_label+" SNN.png") # save as png
    if plot_flag:
        plt.show() # display
    plt.close()

def plot_learning_curve(data, address_string, test_label, save_flag=False):
    # if not isinstance(data[0], list):
    #     data = [data]

    fig, axs = plt.subplots(len(data), 1)
    axs[0].set_title(test_label)
    axs[0].scatter([i for i in range(len(data[0]))], data[0])
    axs[1].scatter([i for i in range(len(data[1]))], data[1])
    axs[1].plot([0, len(data[1])], [75, 75], 'r')
    if save_flag:
        plt.savefig(address_string+test_label+" learning curve.png")
    # plt.show()
    plt.close()


def create_video(top_directory, base_label):
    for root, dirs, files in os.walk(top_directory):
        if base_label in files:
            print("save files")
    print("creating video")
    # for filename in filenames:
    #     images.append(imageio.imread(filename))
    # imageio.mimsave(top_directory+'/videos/'+base_label+'.gif', images)

# draw_graph_from_file('/home/adampcloth/PycharmProjects/PyNN8Examples/eprop_testing/connection_lists/',
#                      'full 0nd 1 cue 20n recF',
#                      False,
#                      save_flag=False)
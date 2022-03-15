import matplotlib.pyplot as plt
import numpy as np


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, perc=0, x_axis=[]):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    offset_array = []

    ax2 = ax.twiny()

    x_axis2 = ["[4, 2]", "[9, 3]", "[8, 2]", "[16, 4]", "[25, 5]", "[18, 3]", "[36, 6]", "[49, 7]", "[16, 2]",
               "[32, 4]", "[20, 2]", "[24, 2]"]

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        offset_array.append(x_offset)

        if name == 'single expanded':
            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax2.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
                bar = ax.bar(x + x_offset, 0, width=bar_width * single_width, color=colors[i % len(colors)])
        else:
            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
                bar = ax2.bar(x + x_offset, 0, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_title("Total Processed Synaptic Events " + str(perc) + " % connectivity (Plastic) 1 ms")

    ax2.set_xticks(np.arange(len(x_axis2)))
    ax2.set_xticklabels(x_axis2)


if __name__ == "__main__":

    tot_events_wrong = dict()
    events_per_neuron_wrong = dict()

    tot_events_single = dict()
    events_per_neuron_single = dict()

    tot_events_multi = dict()
    events_per_neuron_multi = dict()
    
    tot_evets_base = dict()

    x_axis = ["[2, 2]", "[3, 3]", "[4, 2]", "[4, 4]", "[5, 5]", "[6, 3]", "[6, 6]", "[7, 7]", "[8, 2]", "[8, 4]", "[10, 2]", "[12, 2]"]
    x_axis2 = ["[4, 2]", "[9, 3]", "[8, 2]", "[16, 4]", "[25, 5]", "[18, 3]", "[36, 6]", "[49, 7]", "[16, 2]", "[32, 4]", "[20, 2]", "[24, 2]"]
    cores = [2, 3, 2, 4, 5, 3, 6, 7, 2, 4, 2, 2]

    filename_wrong = "/localhome/g90604lp/multitarget_experiments/Peak_comp/plasticity/10000_neurons_1_wrong_sing_final.txt"
    filename_single = "/localhome/g90604lp/multitarget_experiments/Peak_comp/plasticity/10000_neurons_1_full_sing_final.txt"
    filename_multi = "/localhome/g90604lp/multitarget_experiments/Peak_comp/plasticity/10000_neurons_1_full_mult_final.txt"
    
    fpath = filename_multi.split("/")[-1]
    ts = int(fpath.split("_")[0]) / 10

    with open(filename_wrong, "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line != "":
            line_data = fp.readline()
            l = line.split(" ")
            syn_cores = int(l[0].strip("[,"))
            targets = int(l[3])
            conn_prob = float(l[4])
            l_data = line_data.split(" ")
            tot_eve = int(l_data[0])
            neur_eve = float(l_data[1])
            if conn_prob not in tot_events_wrong.keys():
                tot_events_wrong[conn_prob] = [tot_eve]
            else:
                tot_events_wrong[conn_prob].append(tot_eve)
            if conn_prob not in events_per_neuron_wrong.keys():
                events_per_neuron_wrong[conn_prob] = [round(neur_eve, 3)]
            else:
                events_per_neuron_wrong[conn_prob].append(round(neur_eve, 3))

            line = fp.readline()

    with open(filename_single, "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line != "":
            line_data = fp.readline()
            l = line.split(" ")
            syn_cores = int(l[0].strip("[,"))
            targets = int(l[3])
            conn_prob = float(l[4])
            l_data = line_data.split(" ")
            tot_eve = int(l_data[0])
            neur_eve = float(l_data[1])
            if conn_prob not in tot_events_single.keys():
                tot_events_single[conn_prob] = [tot_eve]
            else:
                tot_events_single[conn_prob].append(tot_eve)
            if conn_prob not in events_per_neuron_single.keys():
                events_per_neuron_single[conn_prob] = [round(neur_eve, 3)]
            else:
                events_per_neuron_single[conn_prob].append(round(neur_eve, 3))

            line = fp.readline()


    with open(filename_multi, "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line != "":
            line_data = fp.readline()
            l = line.split(" ")
            syn_cores = int(l[0].strip("[,"))
            targets = int(l[3])
            conn_prob = float(l[4])
            l_data = line_data.split(" ")
            tot_eve = int(l_data[0])
            neur_eve = float(l_data[1])
            if conn_prob not in tot_events_multi.keys():
                tot_events_multi[conn_prob] = [tot_eve]
            else:
                tot_events_multi[conn_prob].append(tot_eve)
            if conn_prob not in events_per_neuron_multi.keys():
                events_per_neuron_multi[conn_prob] = [round(neur_eve, 3)]
            else:
                events_per_neuron_multi[conn_prob].append(round(neur_eve, 3))

            line = fp.readline()
            
    singles = [0, 1, 3, 4, 6, 7]
    
    for conn_prob in tot_events_wrong.keys():
        c_prob = conn_prob / 100
        eve = ((float(ts - 65 - (8.064 * c_prob + 6.567) - (7.36 * c_prob + 2.48)) / (7.36 * c_prob + 3.96)) + 2) * 64 * c_prob
        # empty pkts
        print(eve)
        empty = eve / 10
        tot_evets_base[conn_prob] = list()
        for c in cores:
            curr = tot_events_single[conn_prob][len(tot_evets_base[conn_prob])]
            if len(tot_evets_base[conn_prob]) in singles and ((int(eve) * c) >= curr or ((curr - int(eve) * c) < curr/20)):
                tot_evets_base[conn_prob].append(curr - (curr/6))
            else:
                tot_evets_base[conn_prob].append(int((eve) * c))

    print(tot_events_wrong)
    print(tot_evets_base)
    
    for k in tot_events_multi.keys():
        data = {

            "single expanded": tot_events_wrong[k],
            "multitarget": tot_events_multi[k],
            "single target": tot_events_single[k],
            #"base": tot_evets_base[k]

        }

        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots()
        bar_plot(ax, data, total_width=.8, single_width=.9, perc=k, x_axis=x_axis, colors=plt.cm.viridis(np.linspace(0, 1, 4)))
        plt.show()

    # for k in tot_events_multi.keys():
    #
    #     N = 3
    #     #plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
    #     fig, ax = plt.subplots()
    #
    #     ax.bar(np.arange(len(x_axis2)), tot_events_single[k], width=0.3*0.9, color="#DCE319FF", label='single target',
    #            align="center")
    #     ax.bar(np.arange(len(x_axis2)) + 0.3, tot_events_multi[k], width=0.3*0.9, color="#238A8DFF", label='multitarget',
    #            align="center")
    #     ax.bar(np.arange(len(x_axis2)) + 0.6, tot_events_wrong[k], width=0.3*0.9, color="#440154FF", label='single expanded',
    #            align="center")
    #
    #
    #     ax.legend(loc=0)
    #
    #     ax.set_xlabel("Cores Arrangement")
    #     ax.set_ylabel("Total Synaptic Events")
    #     ax.set_title("Total Processed Synaptic Events " + str(k) + " % connectivity")
    #     ax.set_xticks(np.arange(len(x_axis)))
    #     ax.set_xticklabels(x_axis)
    #
    #     #fig.tight_layout()
    #
    #     plt.show()

# for k in events_per_neuron_multi.keys():
#
#     N = 3
#     plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
#     fix, ax = plt.subplots()
#
#     ax.plot(x_axis, events_per_neuron_wrong[k], 'o-', label="single expanded")
#     ax.plot(x_axis, events_per_neuron_single[k], 'o-', label="single target")
#     ax.plot(x_axis, events_per_neuron_multi[k], 'o-', label="multitarget")
#
#     ax.set_xlabel("Cores Arrangement")
#     ax.set_ylabel("Synaptic Events Per Neuron")
#     ax.set_title("Processed Synaptic Events Per Neuron " + str(k) + " % connectivity")
#     plt.legend()
#     plt.show()

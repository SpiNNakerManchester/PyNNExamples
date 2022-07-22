import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    tot_events_wrong = dict()
    events_per_neuron_wrong = dict()

    tot_events_single = dict()
    events_per_neuron_single = dict()

    tot_events_multi = dict()
    events_per_neuron_multi = dict()
    
    tot_evets_base = dict()

    #0.1 ms
    x_axis = [4, 6, 8, 10, 9, 12, 14, 10, 12]
    x_axis1 = [4, 6, 8, 10, 9, 12, 14, 10, 12]
    x_axis2 = [12, 10, 20, 30, 21, 42, 18, 36]
    cores = [2, 3, 2, 4, 5, 3, 6, 7]

    #1 ms
    # x_axis = [4, 6, 8, 9, 12, 14, 10, 12, 12, 14]
    # x_axis1 = [6, 8, 10, 9, 12, 14, 10, 12, 12, 14]
    # x_axis2 = [12, 10, 20, 30, 21, 42, 18, 36, 22, 26]
    # cores = [2, 3, 2, 4, 5, 3, 6, 7]

    #Plastic
    # x_axis = [4, 6, 8, 10, 9, 12, 14, 10, 12, 14]
    # x_axis1 = [4, 6, 8, 10, 9, 12, 14, 10, 12, 14]
    # x_axis2 = [12, 10, 20, 30, 21, 42, 18, 36, 22, 26]

    #x_axis = [4, 6, 6, 8, 10, 9, 12, 14, 10, 12]
    #x_axis2 = [6, 12, 10, 20, 30, 21, 42, 56, 18, 36]

    filename_wrong = "/localhome/g90604lp/multitarget_experiments/Peak_comp/1000_neurons_0.1_wrong_sing_final.txt"
    filename_single = "/localhome/g90604lp/multitarget_experiments/Peak_comp/1000_neurons_0.1_full_sing_final.txt"
    filename_multi = "/localhome/g90604lp/multitarget_experiments/Peak_comp/1000_neurons_0.1_full_mult_final.txt"

    name = filename_multi.split("/")
    if name[5] == "plasticity":
        pl = "(Plastic) "
        ts = 1000
    else:
        n = name[5]
        pl = "(Static) "
        v = int(n.split("_")[0])
        if v == 10000:
            ts = 1000
        else:
            ts = 100

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
        # 64 neurons per core
        eve = ((float(ts - 65 - (8.064 * c_prob + 6.567) - (7.36 * c_prob + 2.48)) / (7.36 * c_prob + 3.96)) + 2) * 64 * c_prob
        # 256 neurons per core
        # eve = ((float(ts - 263 - (32.26 * c_prob + 6.567) - (29.44 * c_prob + 2.48)) / (
        #             29.44 * c_prob + 3.96)) + 2) * 256 * c_prob
        # empty pkts
        empty = eve / 10
        tot_evets_base[conn_prob] = list()
        for c in cores:
            curr = tot_events_single[conn_prob][len(tot_evets_base[conn_prob])]
            if len(tot_evets_base[conn_prob]) in singles and ((int(eve) * c) >= curr or ((curr - int(eve) * c) < curr/20)):
                tot_evets_base[conn_prob].append(curr - (curr/6))
            else:
                tot_evets_base[conn_prob].append(int((eve) * c))
    
    j = 1

    for k in tot_events_multi.keys():
        data = {

            "single expanded": tot_events_wrong[k],
            "multitarget": tot_events_multi[k],
            "single target": tot_events_single[k],
            "homogeneous": tot_evets_base[k]

        }

        #0.1ms
        lab1 = ["2,2", "3,3", "4,4", "5,5", "6,3", "6,6", "7,7", "8,2", "8,4"]

        lab = ["2,2", "3,3", "4,4", "5, 5", "6,3", "6,6", "7,7", "8,2", "8,4"]

        lab2 = ["9,3", "8,2", "16,4", "25,5", "18,3", "36,6", "16,2",
                "32,4"]
        
        labbase = ["2", "3", "4", "5", "6", "7"]

        #1 ms
        # lab1 = ["3,3", "4,4", "5,5", "6,3", "6,6", "7,7", "8,2", "8,4", "10, 2", "12, 2"]
        #
        # lab = ["2,2", "3,3", "4,4", "6,3", "6,6", "7,7", "8,2", "8,4", "10, 2", "12, 2"]
        #
        # lab2 = ["9,3", "8,2", "16,4", "25,5", "18,3", "36,6", "16,2",
        #        "32,4", "20, 2", "24, 2"]

        #Plastic
        # lab1 = ["2,2", "3,3", "4,4", "5,5", "6,3", "6,6", "7,7", "8,2", "8,4", "12, 2"]
        #
        # lab = ["2,2", "3,3", "4,4", "5,5", "6,3", "6,6", "7,7", "8,2", "10, 2", "12, 2"]
        #
        # lab2 = ["9,3", "8,2", "16,4", "25,5", "18,3", "36,6", "16,2",
        #         "32,4", "20, 2", "24, 2"]
        #
        cores = [2, 3, 4, 5, 6, 7]

        #Static
        tot_events_wrong[k].pop(7)
        tot_events_wrong[k].pop(0)
        tot_events_single[k].pop(2)
        tot_events_multi[k].pop(2)
        #These three are double commented out
        #tot_events_single[k].pop(9)
        #tot_evets_base[k].pop(9)
        #tot_evets_base[k].pop(8)
        tot_evets_base[k].pop(5)
        tot_evets_base[k].pop(2)
        #tot_events_single[k].pop(0)
        #tot_events_multi[k].pop(3)

        #Plastic
        # tot_events_single[k].pop(10)
        # tot_events_wrong[k].pop(7)
        # tot_events_wrong[k].pop(0)
        # tot_events_single[k].pop(2)
        # tot_events_multi[k].pop(2)
        # # tot_events_single[k].pop(9)
        # # tot_evets_base[k].pop(9)
        # # tot_evets_base[k].pop(8)
        # tot_evets_base[k].pop(5)
        # tot_evets_base[k].pop(2)
        # tot_events_multi[k].pop(8)


        N = 4
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
        plt.rcParams.update({'font.size': 20})

        fig, ax = plt.subplots()

        ax.scatter(x_axis2, tot_events_wrong[k], marker="s", label="single expanded")
        ax.scatter(x_axis, tot_events_multi[k], marker="o", label="multitarget")
        ax.scatter(x_axis1, tot_events_single[k], marker="^", label="single target")
        ax.scatter(cores, tot_evets_base[k], marker="X", label="homogeneous")


        ax.set_xticks(range(0, 50, 5))

        ax.set_xlabel("Total Cores")
        ax.set_ylabel("Total Synaptic Events")
        ax.set_title("Processed Synaptic Events " + str(k) + "% connectivity " + pl + str(ts/1000) + " ms")

        text_size = 12

        # for i in range(len(x_axis)):
        #     ax.text(x_axis[i], tot_events_multi[k][i], lab[i], color=plt.cm.viridis(np.linspace(0, 1, 4))[1], fontsize=text_size)
        # for i in range(len(x_axis1)):
        #     ax.text(x_axis1[i], tot_events_single[k][i], lab1[i], color=plt.cm.viridis(np.linspace(0, 1, 4))[2], fontsize=text_size)
        # for i in range(len(x_axis2)):
        #     ax.text(x_axis2[i], tot_events_wrong[k][i], lab2[i], color=plt.cm.viridis(np.linspace(0, 1, 4))[0], fontsize=text_size)
        #for i in range(len(cores)):
        #    ax.text(cores[i], tot_evets_base[k][i], labbase[i], color=plt.cm.viridis(np.linspace(0, 1, 4))[3], fontsize=text_size)

        j += 1

        ax.legend()
        plt.show()

import matplotlib.pyplot as plt
import numpy as np

tot_events10 = dict()
tot_events1 = dict()

conn_probs = [0.001, 0.01, 0.1, 1, 10]

filename1 = "/localhome/g90604lp/multitarget_experiments/Peak_comp/Intro_plots/1000_neurons_10_intro.txt"
filename10 = "/localhome/g90604lp/multitarget_experiments/Peak_comp/Intro_plots/10000_neurons_10_intro.txt"

namestr = filename10.split("/")[-1]
#index = int(namestr.split(".")[0][-1])

with open(filename10, "r") as fp:
    line = fp.readline()
    line = fp.readline()
    while line != "":
        line_data = fp.readline()
        l = line.split(" ")
        syn_cores = int(l[0].strip("[,"))
        targets = int(l[3])
        l_data = line_data.split(" ")
        tot_eve = int(l_data[0])
        neur_eve = float(l_data[1])
        if targets not in tot_events10.keys():
            tot_events10[targets] = [tot_eve]
        else:
            tot_events10[targets].append(tot_eve)

        line = fp.readline()

namestr = filename1.split("/")[-1]
#index = int(namestr.split(".")[0][-1])

with open(filename1, "r") as fp:
    line = fp.readline()
    line = fp.readline()
    while line != "":
        line_data = fp.readline()
        l = line.split(" ")
        syn_cores = int(l[0].strip("[,"))
        targets = int(l[3])
        l_data = line_data.split(" ")
        tot_eve = int(l_data[0])
        neur_eve = float(l_data[1])
        if targets not in tot_events1.keys():
            tot_events1[targets] = [tot_eve]
        else:
            tot_events1[targets].append(tot_eve)

        line = fp.readline()

plt.rcParams.update({'font.size': 20})
fix, ax =plt.subplots()

label = "1 ms"
x_axis10 = [(_ + 2) for _ in range(len(tot_events10[1]))]
l1 = ax.plot(x_axis10, tot_events10[1], 'o-', label=label, color="#33638DFF")

label2 = "0.1 ms"
ax2 = ax.twinx()
x_axis1 = [(_ + 2) for _ in range(len(tot_events1[1]))]
l2 = ax2.plot(x_axis1, tot_events1[1], 'o-', label=label2, color="#55C667FF")

ls = l1 + l2
labs = [l.get_label() for l in ls]
ax.legend(ls, labs, loc=0)

ax.set_ylim(0, max(tot_events10[1]) + 500)
ax2.set_ylim(0, (max(tot_events10[1]) + 500) / 10)

ax.set_xlabel("Synaptic Cores")
ax.set_ylabel("Total Synaptic Events (1 ms)")
ax2.set_ylabel("Total Synaptic Events (0.1 ms)")
ax.set_title("Total Processed Synaptic Events")
plt.show()


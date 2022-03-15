import matplotlib.pyplot as plt
import numpy as np
import glob

tot_events = dict()
events_per_neuron = dict()

ytic = []

filename = "/localhome/g90604lp/multitarget_experiments/Sparsity_comp/10000_neurons_1.txt"

x_conv = {0.1: 0,
          1.0: 1,
          5.0: 2,
          10.0: 3,
          50.0: 4}

with open(filename, "r") as fp:
    line = fp.readline()
    line = fp.readline()

    while line != "":
        line2 = fp.readline()
        l = line.split(" ")
        syn_cores = int(l[0].strip("[,"))
        targets = int(l[3])
        prob = float(l[4])
        eve = line2.split(" ")
        tot_eve = int(eve[0])
        neur_eve = float(eve[1])
        if (syn_cores, targets) not in tot_events.keys():
            tot_events[(syn_cores, targets)] = [0 for _ in range(len(x_conv.keys()))]
            tot_events[(syn_cores, targets)][x_conv[prob]] = tot_eve
            ytic.append(tot_eve)
        else:
            tot_events[(syn_cores, targets)][x_conv[prob]] = tot_eve
            ytic.append(tot_eve)
        if (syn_cores, targets) not in events_per_neuron.keys():
            events_per_neuron[(syn_cores, targets)] = [0 for _ in range(len(x_conv.keys()))]
            events_per_neuron[(syn_cores, targets)][x_conv[prob]] = round(neur_eve, 3)
        else:
            events_per_neuron[(syn_cores, targets)][x_conv[prob]] = round(neur_eve, 3)

        line = fp.readline()

N = 7
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
plt.rcParams.update({'font.size': 20})
fix, ax =plt.subplots()

for k in tot_events.keys():
    label = str(k[0]) + " Syn cores to " + str(k[1]) + " targets"
    x_axis = [0.1, 1, 5, 10, 50]
    ax.plot(x_axis, tot_events[k], 'o-', label=label)

ax.set_xscale('log')
ax.set_xticks([0.1, 1, 5, 10, 50])
ax.set_xticklabels(["0.1", "1", "5", "10", "50"])
ax.set_xlabel("Connectivity (%)")
ax.set_ylabel("Total Synaptic Events")
ax.set_title("Total Processed Synaptic Events 1 ms timestep")
#ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.show()

N = 7
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax =plt.subplots()

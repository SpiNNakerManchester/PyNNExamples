import matplotlib.pyplot as plt
import numpy as np
import glob

tot_events = dict()
events_per_neuron = dict()

#filename = "/localhome/g90604lp/multitarget_experiments/10000_neurons_001.txt"
filepath = glob.iglob("/localhome/g90604lp/multitarget_experiments/Microseconds_res/*.txt")

x_axis = [0.01, 0.1, 1, 10]

for filename in filepath:
    name_file = filename.split("_")[-1]
    index = int(name_file.split(".")[0])

    with open(filename, "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line != "":
            line_data = fp.readline()
            line2 = fp.readline()
            l = line.split(" ")
            syn_cores = int(l[0].strip("[,"))
            targets = int(l[3])
            cmp = line2.split(",")[0]
            cmp = cmp.strip("[")
            if (line2 != "" and int(cmp) != syn_cores) or (line2 == ""):
                l_data = line_data.split(" ")
                tot_eve = int(l_data[0])
                neur_eve = float(l_data[1])
                if (syn_cores, targets) not in tot_events.keys():
                    tot_events[(syn_cores, targets)] = [0 for _ in range(len(x_axis))]
                    tot_events[(syn_cores, targets)][index] = tot_eve
                else:
                    tot_events[(syn_cores, targets)][index] = tot_eve
                if (syn_cores, targets) not in events_per_neuron.keys():
                    events_per_neuron[(syn_cores, targets)] = [0 for _ in range(len(x_axis))]
                    events_per_neuron[(syn_cores, targets)][index] = round(neur_eve, 3)
                else:
                    events_per_neuron[(syn_cores, targets)][index] = round(neur_eve, 3)

            line = line2

N = 13
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax =plt.subplots()

for k in tot_events.keys():
    label = str(k[0]) + " Syn cores to " + str(k[1]) + " targets"
    x_axis = [0.01, 0.1, 1, 10]
    ax.plot(x_axis, tot_events[k], 'o-', label=label)

ax.set_xticks([0.01, 0.1, 1, 10])
ax.set_xlabel("Connectivity (%)")
ax.set_ylabel("Total Synaptic Events")
ax.set_title("Total Processed Synaptic Events")
ax.set_xscale('log')
#ax.set_yscale('log')
plt.legend()
plt.show()

N = 13
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax =plt.subplots()

for k in events_per_neuron.keys():
    label = str(k[0]) + " Syn cores to " + str(k[1]) + " targets"
    x_axis = [0.01, 0.1, 1, 10]
    ax.plot(x_axis, events_per_neuron[k], 'o-', label=label)

ax.set_xticks([0.01, 0.1, 1, 10])
ax.set_xlabel("Connectivity (%)")
ax.set_ylabel("Synaptic Events Per Neuron")
ax.set_title("Processed Synaptic Events Per Neuron")
ax.set_xscale('log')
#ax.set_yscale('log')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

tot_events = dict()
events_per_neuron = dict()

conn_probs = [0.001, 0.01, 0.1, 1, 10]

filename = "/localhome/g90604lp/multitarget_experiments/Millisec_res/10000_neurons_10_3.txt"

namestr = filename.split("/")[-1]
index = int(namestr.split(".")[0][-1])

with open(filename, "r") as fp:
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
        if targets not in tot_events.keys():
            tot_events[targets] = [tot_eve]
        else:
            tot_events[targets].append(tot_eve)
        if targets not in events_per_neuron.keys():
            events_per_neuron[targets] = [round(neur_eve, 3)]
        else:
            events_per_neuron[targets].append(round(neur_eve, 3))

        line = fp.readline()

N = 12
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax =plt.subplots()

for k in tot_events.keys():
    label = str(k) + " targets"
    x_axis = [(_ + 2) for _ in range(len(tot_events[k]))]
    ax.plot(x_axis, tot_events[k], 'o-', label=label)

ax.set_xlabel("Synaptic Cores")
ax.set_ylabel("Total Synaptic Events")
ax.set_title("Total Processed Synaptic Events " + str(conn_probs[index]) + "% Connectivity")
plt.legend()
plt.show()

N = 12
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax =plt.subplots()

for k in events_per_neuron.keys():
    label = str(k) + " targets"
    x_axis = [(_ + 2) for _ in range(len(events_per_neuron[k]))]
    ax.plot(x_axis, events_per_neuron[k], 'o-', label=label)

ax.set_xlabel("Synaptic Cores")
ax.set_ylabel("Synaptic Events Per Neuron")
ax.set_title("Processed Synaptic Events Per Neuron " + str(conn_probs[index]) + "% Connectivity")
plt.legend()
plt.show()

import glob
import matplotlib.pyplot as plt
import numpy as np

files = {}
j = 0

for txtfile in glob.iglob("/localhome/g90604lp/ICPP_res/synapses/*.txt"):
    with open(txtfile, "r") as fp:
        i = 0
        partitions = []
        values = []
        for line in fp:
            if i == 0:
                #neurons = line.split(" ")[1][:-1]
                filename_split = txtfile.split("/")
                timestep = filename_split[len(filename_split)-1].split("_")[2]

            if i % 2 == 0:
                partitions.append(int(line.split(" ")[0]))
            else:
                syn_events = line.split(" ")[0]
                values.append(float(syn_events))

            i += 1
        neurons = txtfile.split("/")[5].split("_")[0]
    files[j] = values
    j += 1


minerr = []
maxerr = []
avgerr = []
for i in range(len(partitions)):
    li = []
    avg = 0
    for k in range(j):
        li.append(files[k][i])
        avg += files[k][i]
    finavg = float(avg) / 3
    avgerr.append(finavg)
    minerr.append(finavg - min(li))
    maxerr.append(max(li) - finavg)



plt.errorbar(partitions, avgerr, yerr=[minerr, maxerr], fmt="-o", capsize=7, elinewidth=3, linewidth=3, markersize=7)


plt.xlabel('synapse cores', size=28)
plt.ylabel("synaptic events per timestep", size=28)

plt.grid()
plt.xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], size=17)
plt.yticks(size=17)

    #plt.suptitle(neurons + " Neurons, timestep " + timestep + " ms")
plt.show()

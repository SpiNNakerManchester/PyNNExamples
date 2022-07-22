import matplotlib.pyplot as plt
import numpy as np

filename="/localhome/g90604lp/selfpred.txt"

upyr = []
va = []
vb = []
usst = []
utop = []

with open(filename, "r") as fp:

    line = fp.readline()
    linelist = line.split(" ")
    upyr.append(float(linelist[0].strip("[]")))
    for i in range(1, len(linelist[1:len(linelist)-1])):
        upyr.append(float(linelist[i].strip("mV[]")))

    line = fp.readline()
    linelist = line.split(" ")
    va.append(float(linelist[0].strip("[]")))
    for i in range(1, len(linelist[1:len(linelist) - 1])):
        va.append(float(linelist[i].strip("uS[]")))

    line = fp.readline()
    linelist = line.split(" ")
    vb.append(float(linelist[0].strip("[]")))
    for i in range(1, len(linelist[1:len(linelist) - 1])):
        vb.append(float(linelist[i].strip("uS[]")))

    line = fp.readline()
    linelist = line.split(" ")
    usst.append(float(linelist[0].strip("[]")))
    for i in range(1, len(linelist[1:len(linelist) - 1])):
        usst.append(float(linelist[i].strip("mV[]")))

    line = fp.readline()

    line = fp.readline()
    linelist = line.split(" ")
    utop.append(float(linelist[0].strip("[]")))
    for i in range(1, len(linelist[1:len(linelist) - 1])):
        utop.append(float(linelist[i].strip("mV[]")))

plt.rcParams.update({'font.size': 19.5})
fix, ax = plt.subplots(5, 1)

ax[0].plot([i for i in range(len(upyr))], upyr)

ax[0].set_ylabel("Pyramidal\nSomatic")
ax[0].yaxis.set_ticks(np.arange(0, 1, 0.2))

ax[1].plot([i for i in range(len(va))], va)

ax[1].set_ylabel("Apical")
ax[1].yaxis.set_ticks(np.arange(0, 0.8, 0.2))

ax[2].plot([i for i in range(len(vb))], vb)

ax[2].set_ylabel("Basal")
ax[2].yaxis.set_ticks(np.arange(0, 1.5, 0.25))

ax[3].plot([i for i in range(len(usst))], usst)

ax[3].set_ylabel("SST\nSomatic")
ax[3].yaxis.set_ticks(np.arange(0, 1.2, 0.2))

ax[4].plot([i for i in range(len(utop))], utop)

ax[4].set_ylabel("Top-down\nSomatic")
ax[4].yaxis.set_ticks(np.arange(0, 1.2, 0.2))

ax[4].set_xlabel("Time (ms)")

plt.show()